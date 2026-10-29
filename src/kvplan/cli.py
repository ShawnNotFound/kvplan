"""Command line: `kvplan check MODEL --gpu H100-80GB`."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .gpus import GPUS_MIB, gpu_total_bytes
from .kv import kv_layout, resolve_kv_cache_dtype
from .loader import load
from .model import UnsupportedModel
from .sources import SourceError, open_source
from .vllm import DEFAULT_UTILIZATION, GiB, ServeConfig, plan
from .weights import weight_bytes_per_rank


def _gib(b) -> str:
    return f"{b / GiB:.2f} GiB"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kvplan", description="Plan vLLM deployments without the GPU.")
    p.add_argument("--version", action="version", version=f"kvplan {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def model_args(sp):
        sp.add_argument("model", help="Hugging Face repo id (owner/name) or a local directory or file")
        sp.add_argument("--revision", default="main")
        sp.add_argument("--token", help="Hugging Face token (default: HF_TOKEN or the huggingface-cli login)")
        sp.add_argument("--gguf", nargs="?", const="", default=None, metavar="QUANT",
                        help="read a GGUF file instead of safetensors (optionally pick a quant, e.g. Q4_K_M)")
        sp.add_argument("--json", action="store_true", help="print JSON")

    c = sub.add_parser("check", help="will this vLLM config start, and how much KV cache will it get?")
    model_args(c)
    gpu = c.add_mutually_exclusive_group(required=True)
    gpu.add_argument("--gpu", help=f"GPU type ({', '.join(GPUS_MIB)})")
    gpu.add_argument("--gpu-mem-gib", type=float, help="exact total GPU memory in GiB, as CUDA reports it")
    c.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    c.add_argument("--max-model-len", type=int)
    c.add_argument("--max-num-seqs", type=int)
    c.add_argument("--max-num-batched-tokens", type=int)
    c.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_UTILIZATION)
    c.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])
    c.add_argument("--block-size", type=int, help="KV block size (default: vLLM's backend default)")
    c.add_argument("--enforce-eager", action="store_true")
    c.add_argument("--no-async-scheduling", action="store_true", help="as vLLM's --no-async-scheduling")
    c.add_argument("--other-gpu-mem-gib", type=float, default=0.0,
                   help="memory already used on the GPU by other processes")
    c.add_argument("--overhead-gib", type=float,
                   help="replace the estimated activation + CUDA-graph + runtime overhead with this value")

    i = sub.add_parser("inspect", help="show the model's attention geometry, weights and KV bytes per token")
    model_args(i)
    i.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    i.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])
    return p


def _print_spec(loaded, tp: int, kv_dtype: str) -> None:
    s = loaded.spec
    kinds = {k: s.layer_types.count(k) for k in dict.fromkeys(s.layer_types)}
    attn = "MLA (latent KV)" if s.is_mla else (
        f"{s.num_heads} heads, {s.num_kv_heads} KV heads, head_dim {s.head_dim}")
    print(f"  architecture   {s.model_type} (read from {s.source})")
    print(f"  layers         {s.num_layers}: " + ", ".join(f"{n} {k}" for k, n in kinds.items())
          + (f" (window {s.sliding_window})" if s.sliding_window else ""))
    print(f"  attention      {attn}")
    if s.num_experts:
        print(f"  experts        {s.num_experts} (all resident in GPU memory)")
    ctx = s.max_position_embeddings
    if ctx and s.default_max_model_len and s.default_max_model_len != ctx:
        print(f"  context        up to {ctx} (vLLM defaults to {s.default_max_model_len})")
    else:
        print(f"  context        {ctx or 'unknown'}")
    layout = kv_layout(s, tp, kv_dtype)
    per_token = sum(layer.page_bytes for layer in layout.layers) // layout.block_size
    print(f"  KV per token   {per_token:,} bytes per GPU at tp={tp} ({resolve_kv_cache_dtype(s, kv_dtype)})")


def cmd_inspect(args) -> int:
    loaded = load(open_source(args.model, args.revision, args.token), gguf=args.gguf)
    w = weight_bytes_per_rank(loaded.spec, loaded.tensors, args.tensor_parallel_size)
    if args.json:
        from dataclasses import asdict
        print(json.dumps({"spec": asdict(loaded.spec), "weights": asdict(w), "files": loaded.files}, indent=2))
        return 0
    print(args.model)
    _print_spec(loaded, args.tensor_parallel_size, args.kv_cache_dtype)
    print(f"  weights        {_gib(w.per_rank_bytes)} per GPU at tp={args.tensor_parallel_size} "
          f"(checkpoint {_gib(w.total_checkpoint_bytes)} in {len(loaded.files)} file(s))")
    for n in loaded.spec.notes + w.notes:
        print(f"  note           {n}")
    return 0


def cmd_check(args) -> int:
    loaded = load(open_source(args.model, args.revision, args.token), gguf=args.gguf)
    total = int(args.gpu_mem_gib * GiB) if args.gpu_mem_gib else gpu_total_bytes(args.gpu)
    cfg = ServeConfig(
        gpu_total_bytes=total, tp=args.tensor_parallel_size, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization, kv_cache_dtype=args.kv_cache_dtype,
        block_size=args.block_size, enforce_eager=args.enforce_eager,
        async_scheduling=not args.no_async_scheduling,
        other_process_bytes=int(args.other_gpu_mem_gib * GiB),
        overhead_bytes=int(args.overhead_gib * GiB) if args.overhead_gib is not None else None,
        gpu_name=args.gpu or "",
    )
    cfg.validate()
    w = weight_bytes_per_rank(loaded.spec, loaded.tensors, cfg.tp)
    p = plan(loaded.spec, w, cfg, args.model)
    if args.json:
        print(json.dumps(p.to_dict() | {"spec_notes": loaded.spec.notes}, indent=2))
        return 0 if p.starts else 1

    gpu_desc = args.gpu or f"{args.gpu_mem_gib} GiB GPU"
    print(f"{args.model} on {cfg.tp} x {gpu_desc} "
          f"(max_model_len {p.max_model_len}, max_num_seqs {p.max_num_seqs}, "
          f"gpu_memory_utilization {cfg.gpu_memory_utilization})")
    print()
    print(f"  budget         {_gib(p.requested_bytes)} of {_gib(p.gpu_total_bytes)} per GPU")
    print(f"  weights        {_gib(w.per_rank_bytes)} per GPU (exact, from the weight files)")
    o = p.overheads
    if o.user_override:
        print(f"  runtime        {_gib(o.total)} (your --overhead-gib)")
    else:
        print(f"  runtime        {_gib(o.total)} estimated: activations {_gib(o.activation_bytes)}, "
              f"sampler logits {_gib(o.logits_bytes)}, CUDA graphs {_gib(o.cudagraph_bytes)}, "
              f"other {_gib(o.non_torch_bytes)}")
    print(f"  KV cache       {_gib(max(p.available_kv_bytes, 0))} available "
          f"({p.kv_bytes_per_token:,} bytes per token per GPU, {p.kv_cache_dtype}, block size {p.block_size})")
    if p.kv is not None:
        print(f"  capacity       {p.kv.kv_tokens:,} tokens; max concurrency at {p.max_model_len:,} tokens per "
              f"request: {p.kv.max_concurrency:.2f}x")
    print()
    print("  RESULT         " + ("vLLM should start" if p.starts else "vLLM will not start"))
    for c in p.checks:
        tag = {"error": "error", "warning": "warning", "info": "note"}[c.severity]
        print(f"  {tag:<14} {c.message}")
    for s in p.suggestions:
        print(f"  try            {s}")
    for n in loaded.spec.notes + w.notes:
        print(f"  note           {n}")
    if not o.user_override:
        print("  note           runtime overheads are estimates until calibrated against real vLLM logs; "
              "weights and KV sizes are exact")
    return 0 if p.starts else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return cmd_inspect(args)
        return cmd_check(args)
    except (SourceError, UnsupportedModel, ValueError) as e:
        print(f"kvplan: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
