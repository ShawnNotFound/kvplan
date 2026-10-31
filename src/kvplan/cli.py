"""Command line: `kvplan check MODEL --gpu H100-80GB`."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

from . import __version__
from . import hub
from .gpus import GPUS_MIB, GPUDetectError, gpu_total_bytes, query_nvidia_smi, select_gpus
from .kv import kv_layout, resolve_kv_cache_dtype
from .loader import load
from .model import UnsupportedModel
from .recommend import NOT_VLLM_TASKS, PRIORITIES, TASKS, recommend
from .sources import RepoNotFoundError, SourceError, find_token, open_source
from .vllm import DEFAULT_UTILIZATION, LARGEST_MEASURED_ENCODER_PEAK, GiB, ServeConfig, plan
from .weights import weight_bytes_per_rank


def _gib(b) -> str:
    return f"{b / GiB:.2f} GiB"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kvplan", description="Plan vLLM deployments without the GPU.")
    p.add_argument("--version", action="version", version=f"kvplan {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def token_arg(sp):
        sp.add_argument("--token", help="Hugging Face token (default: HF_TOKEN, or the token saved by "
                                        "`kvplan login` or `huggingface-cli login`)")

    def model_args(sp, json_flag=True):
        sp.add_argument("model", help="Hugging Face repo id (owner/name), a model name to look up "
                                      "(e.g. Qwen2.5-7B-Instruct), or a local directory or file")
        sp.add_argument("--revision", default="main")
        token_arg(sp)
        sp.add_argument("--gguf", nargs="?", const="", default=None, metavar="QUANT",
                        help="read a GGUF file instead of safetensors (optionally pick a quant, e.g. Q4_K_M)")
        if json_flag:
            sp.add_argument("--json", action="store_true", help="print JSON")

    def gpu_args(sp):
        gpu = sp.add_mutually_exclusive_group()
        gpu.add_argument("--gpu", help=f"GPU type ({', '.join(GPUS_MIB)}), or 'auto' (the default) to read the "
                                       "local GPUs and the memory other processes already use from nvidia-smi")
        gpu.add_argument("--gpu-mem-gib", type=float, help="exact total GPU memory in GiB, as CUDA reports it")
        sp.add_argument("--gpu-index", help="with --gpu auto: nvidia-smi index(es) of the GPU(s) to plan for, "
                                            "comma-separated (default: the first tensor-parallel-size GPUs)")
        sp.add_argument("--other-gpu-mem-gib", type=float,
                        help="memory already used on the GPU by other processes (default: 0, or what nvidia-smi "
                             "reports with --gpu auto)")

    c = sub.add_parser("check", help="will this vLLM config start, and how much KV cache will it get?")
    model_args(c)
    gpu_args(c)
    c.add_argument("--download", action="store_true",
                   help="download the model when the check says vLLM should start")
    c.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    c.add_argument("--max-model-len", type=int)
    c.add_argument("--max-num-seqs", type=int)
    c.add_argument("--max-num-batched-tokens", type=int)
    c.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_UTILIZATION)
    c.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])
    c.add_argument("--block-size", type=int, help="KV block size (default: vLLM's backend default)")
    c.add_argument("--enforce-eager", action="store_true")
    c.add_argument("--no-async-scheduling", action="store_true", help="as vLLM's --no-async-scheduling")
    c.add_argument("--limit-mm-per-prompt", type=json.loads, default=None, metavar="JSON",
                   help='as vLLM\'s, e.g. \'{"video": 0}\'; a modality set to 0 is not profiled')
    c.add_argument("--mm-processor-kwargs", type=json.loads, default=None, metavar="JSON",
                   help='as vLLM\'s; kvplan reads max_pixels and min_pixels, e.g. \'{"max_pixels": 1003520}\'')
    c.add_argument("--skip-mm-profiling", action="store_true", help="as vLLM's --skip-mm-profiling")
    c.add_argument("--overhead-gib", type=float,
                   help="replace the estimated activation + CUDA-graph + runtime overhead with this value")

    i = sub.add_parser("inspect", help="show the model's attention geometry, weights and KV bytes per token")
    model_args(i)
    i.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    i.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])

    s = sub.add_parser("search", help="find models on Hugging Face by a loose name, e.g. 'llama 3.1 8b'")
    s.add_argument("query", nargs="+")
    s.add_argument("--limit", type=int, default=10)
    token_arg(s)
    s.add_argument("--json", action="store_true", help="print JSON")

    d = sub.add_parser("download", help="download a model's weights, configs and tokenizer for vLLM")
    model_args(d, json_flag=False)
    d.add_argument("--local-dir", help="download into this directory instead of the Hugging Face cache")

    lg = sub.add_parser("login", help="save a Hugging Face token so kvplan (and vLLM) can read gated models")
    lg.add_argument("--token", help="the token (default: ask for it)")

    r = sub.add_parser("recommend", help="popular models that vLLM should start on your GPUs, checked")
    gpu_args(r)
    r.add_argument("--gpus", help="how many GPUs to use: a number, or 'any' for the fewest that work "
                                  "(default: ask, or 1)")
    r.add_argument("--task", choices=list(TASKS) + list(NOT_VLLM_TASKS), help="default: ask, or chat")
    r.add_argument("--priority", choices=list(PRIORITIES), help="default: ask, or balanced")
    r.add_argument("--max-model-len", type=int, help="context length per request (default: ask, or 8192)")
    r.add_argument("--min-concurrency", type=float, default=1.0,
                   help="only models that fit at least this many full-length requests at once (default 1)")
    r.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_UTILIZATION)
    r.add_argument("--limit", type=int, default=10, help="how many models to list (default 10)")
    r.add_argument("--no-gated", action="store_true", help="leave out gated models even with a token")
    r.add_argument("--no-input", action="store_true", help="never ask questions; use the defaults")
    token_arg(r)
    r.add_argument("--json", action="store_true", help="print JSON")
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


def _inspect(args, loaded) -> int:
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


def _mm_limits(value) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('--limit-mm-per-prompt must be a JSON object such as \'{"image": 1, "video": 0}\'')
    # vLLM also accepts {"image": {"count": 1, ...}}.
    return {k: int(v["count"] if isinstance(v, dict) else v) for k, v in value.items()}


def spec_kind(spec) -> str:
    return spec.encoder.kind if spec.encoder else ""


def _parse_indices(text: str | None) -> list[int] | None:
    if text is None:
        return None
    try:
        indices = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"--gpu-index must be comma-separated GPU numbers such as 0 or 0,1, "
                         f"not '{text}'") from None
    if not indices or any(i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError(f"--gpu-index must list distinct GPU numbers such as 0 or 0,1, not '{text}'")
    return indices


def _resolve_gpu(args) -> tuple[int, str, int, str, list]:
    """(total bytes, GPU name, bytes used by other processes, description, detected GPUs)."""
    for flag, v in (("--gpu-mem-gib", args.gpu_mem_gib), ("--other-gpu-mem-gib", args.other_gpu_mem_gib),
                    ("--overhead-gib", args.overhead_gib)):
        if v is not None and not math.isfinite(v):
            raise ValueError(f"{flag} must be a number of GiB, not {v}")
    other = int(args.other_gpu_mem_gib * GiB) if args.other_gpu_mem_gib is not None else 0
    if args.gpu_index is not None and (args.gpu_mem_gib is not None or (args.gpu or "auto").lower() != "auto"):
        raise ValueError("--gpu-index only applies to --gpu auto")
    if args.gpu_mem_gib is not None:
        return int(args.gpu_mem_gib * GiB), "", other, f"{args.gpu_mem_gib} GiB GPU", []
    if args.gpu and args.gpu.lower() != "auto":
        return gpu_total_bytes(args.gpu), args.gpu, other, args.gpu, []
    chosen = select_gpus(_detect(args.gpu is None), args.tensor_parallel_size, _parse_indices(args.gpu_index))
    # vLLM sizes every rank like the smallest GPU; the busiest GPU decides whether it starts.
    total = min(g.cuda_total_mib for g in chosen) * 2**20
    if args.other_gpu_mem_gib is None:
        other = max(g.used_mib or 0 for g in chosen) * 2**20
    names = sorted({g.name for g in chosen})
    return total, names[0] if len(names) == 1 else "", other, " / ".join(names), chosen


def _detect(defaulted: bool):
    try:
        return query_nvidia_smi()
    except GPUDetectError as e:
        if defaulted:
            reason = str(e).split("; pass --gpu", 1)[0]
            raise GPUDetectError(f"no --gpu given, and this machine's GPUs could not be read ({reason}). Name the "
                                 "GPU you will run on, e.g. --gpu A6000, or give its memory with --gpu-mem-gib")\
                from None
        raise


def _interactive(args) -> bool:
    return not getattr(args, "json", False) and not getattr(args, "no_input", False) \
        and sys.stdin.isatty() and sys.stdout.isatty()


def _ask(question: str, default: str = "") -> str:
    try:
        answer = input(f"{question}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError:
        answer = ""
    return answer or default


def _fmt_count(n: float) -> str:
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(int(n))


def _format_matches(matches) -> str:
    lines = []
    for i, m in enumerate(matches, 1):
        size = f"{_fmt_count(m.total_parameters)} params" if m.total_parameters else "size unknown"
        lines.append(f"  {i:>2}. {m.id:<50} {size:>13}  {_fmt_count(m.downloads):>6} downloads"
                     + ("  (gated)" if m.gated else ""))
    return "\n".join(lines)


def _resolve_model(args) -> str:
    """The repo id to read: the argument itself, or for a bare name ('Qwen2.5-7B-Instruct') the most
    downloaded repo with exactly that name."""
    model = args.model
    if os.path.exists(model) or re.fullmatch(r"[\w.\-]+/[\w.\-]+", model):
        return model
    found, matches = hub.resolve_name(model, args.token)
    if found:
        print(f"kvplan: using {found}, the most downloaded repo named '{model}'", file=sys.stderr)
        return found
    return _pick_suggestion(args, f"no Hugging Face repo named '{model}'", matches)


def _pick_suggestion(args, problem: str, matches) -> str:
    if not matches:
        raise SourceError(f"{problem}, and a search found nothing similar; try `kvplan search` with fewer words")
    if not _interactive(args):
        raise SourceError(f"{problem}. Did you mean:\n{_format_matches(matches)}")
    print(f"{problem}. Similar models:")
    print(_format_matches(matches))
    answer = _ask("Pick a number to use it (Enter to stop)")
    if answer.isdigit() and 1 <= int(answer) <= len(matches):
        return matches[int(answer) - 1].id
    raise SourceError(problem)


def _load(args):
    """Load the model, offering similar names when the repo does not exist."""
    while True:
        try:
            return load(open_source(args.model, args.revision, args.token), gguf=args.gguf)
        except RepoNotFoundError as e:
            matches = hub.similar_models(e.repo_id, limit=8, token=args.token)
            args.model = _pick_suggestion(args, str(e), matches)


def _serve_command(args, target: str) -> str:
    parts = ["vllm serve", target]
    if args.tensor_parallel_size != 1:
        parts.append(f"--tensor-parallel-size {args.tensor_parallel_size}")
    if args.max_model_len:
        parts.append(f"--max-model-len {args.max_model_len}")
    if args.max_num_seqs:
        parts.append(f"--max-num-seqs {args.max_num_seqs}")
    if args.gpu_memory_utilization != DEFAULT_UTILIZATION:
        parts.append(f"--gpu-memory-utilization {args.gpu_memory_utilization}")
    if args.kv_cache_dtype != "auto":
        parts.append(f"--kv-cache-dtype {args.kv_cache_dtype}")
    if args.enforce_eager:
        parts.append("--enforce-eager")
    return " ".join(parts)


def cmd_check(args) -> int:
    # Resolve the GPU first: a typo or a missing driver should fail before any network access.
    total, gpu_name, other, gpu_desc, detected = _resolve_gpu(args)
    args.model = _resolve_model(args)
    loaded = _load(args)
    cfg = ServeConfig(
        gpu_total_bytes=total, tp=args.tensor_parallel_size, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization, kv_cache_dtype=args.kv_cache_dtype,
        block_size=args.block_size, enforce_eager=args.enforce_eager,
        async_scheduling=not args.no_async_scheduling,
        other_process_bytes=other,
        overhead_bytes=int(args.overhead_gib * GiB) if args.overhead_gib is not None else None,
        gpu_name=gpu_name,
        mm_limits=_mm_limits(args.limit_mm_per_prompt),
        mm_max_pixels=(args.mm_processor_kwargs or {}).get("max_pixels"),
        mm_min_pixels=(args.mm_processor_kwargs or {}).get("min_pixels"),
        skip_mm_profiling=args.skip_mm_profiling,
    )
    cfg.validate()
    w = weight_bytes_per_rank(loaded.spec, loaded.tensors, cfg.tp)
    p = plan(loaded.spec, w, cfg, args.model)
    if args.json:
        extra = {"spec_notes": loaded.spec.notes, "other_process_bytes": other}
        if detected:
            extra["detected_gpus"] = [g.__dict__ | {"cuda_total_mib": g.cuda_total_mib} for g in detected]
        print(json.dumps(p.to_dict() | extra, indent=2))
        if args.download and p.starts:
            _download(args, loaded, log=lambda m: print(m, file=sys.stderr))
        return _exit_code(p)

    print(f"{args.model} on {cfg.tp} x {gpu_desc} "
          f"(max_model_len {p.max_model_len}, max_num_seqs {p.max_num_seqs}, "
          f"gpu_memory_utilization {cfg.gpu_memory_utilization})")
    print()
    for g in detected:
        used = "unknown" if g.used_mib is None else f"{g.used_mib:,} MiB"
        reserved = f", {g.reserved_mib:,} reserved by the driver" if g.reserved_mib else ""
        print(f"  gpu {g.index:<10} {g.name}: {g.total_mib:,} MiB total{reserved}, {used} in use (nvidia-smi)")
    print(f"  budget         {_gib(p.requested_bytes)} of {_gib(p.gpu_total_bytes)} per GPU")
    print(f"  weights        {_gib(w.per_rank_bytes)} per GPU (exact, from the weight files)")
    o = p.overheads
    if o.user_override:
        print(f"  runtime        {_gib(o.total)} (your --overhead-gib)")
    else:
        print(f"  runtime        {_gib(o.total)} estimated: activation peak {_gib(o.activation_bytes)}, "
              f"torch.compile {_gib(o.compile_bytes)}, CUDA graphs {_gib(o.cudagraph_bytes)}, "
              f"other {_gib(o.non_torch_bytes)}"
              + (f", {spec_kind(loaded.spec)} encoder profiling {_gib(o.encoder_bytes)}"
                 + (" (approximate)" if o.encoder_approximate else "") if o.encoder_bytes else ""))
    print(f"  KV cache       {_gib(max(p.available_kv_bytes, 0))} available "
          f"({p.kv_bytes_per_token:,} bytes per token per GPU, {p.kv_cache_dtype}, block size {p.block_size})")
    if p.kv is not None:
        print(f"  capacity       {p.kv.kv_tokens:,} tokens; max concurrency at {p.max_model_len:,} tokens per "
              f"request: {p.kv.max_concurrency:.2f}x")
    print()
    result = p.verdict
    if p.starts is None:
        result += ": kvplan could not check " + "; ".join(p.unconfirmed)
        if p.start_margin_bytes is not None:
            result += f". Under kvplan's estimates {_gib(max(p.start_margin_bytes, 0))} would be left at the start check"
            if any("encoder" in u for u in p.unconfirmed):
                result += (f"; the largest encoder startup peak measured so far was "
                           f"{_gib(LARGEST_MEASURED_ENCODER_PEAK)}")
    print("  RESULT         " + result)
    for c in p.checks:
        tag = {"error": "error", "warning": "warning", "info": "note"}[c.severity]
        print(f"  {tag:<14} {c.message}")
    for s in p.suggestions:
        print(f"  try            {s}")
    for n in loaded.spec.notes + w.notes:
        print(f"  note           {n}")
    if o.encoder_note and not o.user_override:
        print(f"  note           {o.encoder_note}")
    if not o.user_override and not o.calibrated:
        print("  note           runtime overheads for this model are a rough estimate: they were calibrated on "
              "dense models only; pass --overhead-gib to override")
    if p.starts and not os.path.exists(args.model):
        print()
        target = args.model
        if args.download:
            where = _download(args, loaded)
            target = args.model if _has_hf_hub() else where
        else:
            print(f"  download       kvplan download {args.model}   (or add --download to this check)")
        print(f"  serve          {_serve_command(args, target)}")
    return _exit_code(p)


def _has_hf_hub() -> bool:
    import importlib.util
    return importlib.util.find_spec("huggingface_hub") is not None


def _download(args, loaded, log=print) -> str:
    if os.path.exists(args.model):
        raise ValueError(f"{args.model} is already on this machine")
    where = hub.download(args.model, loaded.files, loaded.repo_files, revision=args.revision, token=args.token,
                         local_dir=getattr(args, "local_dir", None), log=log)
    log(f"  downloaded     {args.model} to {where}")
    return where


def cmd_inspect_or_download(args) -> int:
    args.model = _resolve_model(args)
    loaded = _load(args)
    if args.command == "download":
        where = _download(args, loaded)
        print(f"serve it with: vllm serve {where if args.local_dir or not _has_hf_hub() else args.model}")
        return 0
    return _inspect(args, loaded)


def cmd_search(args) -> int:
    query = " ".join(args.query)
    matches = hub.search_models(query, limit=args.limit, token=args.token)
    if args.json:
        print(json.dumps([m.__dict__ for m in matches], indent=2))
        return 0
    if not matches:
        print(f"nothing on Hugging Face matches '{query}'; try fewer or shorter words", file=sys.stderr)
        return 1
    print(_format_matches(matches))
    print(f"\nnext: kvplan check {matches[0].id}")
    return 0


def cmd_login(args) -> int:
    token = args.token
    if not token:
        if os.environ.get("HF_TOKEN"):
            print("note: HF_TOKEN is set in your environment; it takes precedence over a saved token")
        import getpass
        print("Create a token at https://huggingface.co/settings/tokens (read access is enough).")
        token = getpass.getpass("Hugging Face token: ").strip()
    if not token:
        raise ValueError("no token given")
    name = hub.whoami(token)
    path = hub.save_token(token)
    print(f"logged in as {name}; token saved to {path}")
    print("kvplan, vLLM and huggingface-cli all read it from there. For gated models (Llama, Gemma), also accept "
          "the licence on the model's Hugging Face page.")
    return 0


def _gpu_setup(args, ask: bool):
    """(total bytes, name, other bytes, description, GPUs available or None, flags for `kvplan check`)."""
    other = int(args.other_gpu_mem_gib * GiB) if args.other_gpu_mem_gib is not None else 0
    if args.gpu_mem_gib is not None:
        return int(args.gpu_mem_gib * GiB), "", other, f"{args.gpu_mem_gib} GiB GPU", None, \
            ["--gpu-mem-gib", str(args.gpu_mem_gib)]
    if args.gpu and args.gpu.lower() != "auto":
        return gpu_total_bytes(args.gpu), args.gpu, other, args.gpu, None, ["--gpu", args.gpu]
    gpus = _detect(args.gpu is None)
    indices = _parse_indices(args.gpu_index)
    if indices is not None:
        gpus = select_gpus(gpus, len(indices), indices)
    total = min(g.cuda_total_mib for g in gpus if g.cuda_total_mib) * 2**20
    if args.other_gpu_mem_gib is None:
        other = max(g.used_mib or 0 for g in gpus) * 2**20
    names = sorted({g.name for g in gpus})
    flags = ["--gpu", "auto"] + (["--gpu-index", args.gpu_index] if args.gpu_index else [])
    return total, names[0] if len(names) == 1 else "", other, " / ".join(names), len(gpus), flags


def _tp_options(args, available: int | None, gpu_desc: str, ask: bool) -> list[int]:
    choice = args.gpus
    if choice is None and ask:
        if available and available > 1:
            choice = _ask(f"This machine has {available} x {gpu_desc}. Use 1 GPU, {available}, or 'any' "
                          "(the fewest that work)", "any")
        elif not available:
            choice = _ask(f"How many {gpu_desc} GPUs will you use", "1")
    choice = (choice or "1").strip().lower()
    limit = available or 8
    if choice == "any":
        return [n for n in (1, 2, 4, 8) if n <= limit]
    if not choice.isdigit() or int(choice) < 1:
        raise ValueError(f"--gpus must be a number of GPUs or 'any', not '{choice}'")
    if available and int(choice) > available:
        raise ValueError(f"--gpus {choice}, but this machine has {available} GPU(s)")
    return [int(choice)]


def _choose(args, attr: str, question: str, options: dict, default: str, ask: bool) -> str:
    value = getattr(args, attr)
    if value is None and ask:
        print(question)
        keys = list(options)
        for i, k in enumerate(keys, 1):
            desc = options[k] if isinstance(options[k], str) else options[k][1]
            print(f"  {i}. {k:<17} {desc}")
        answer = _ask("Choose", default)
        value = keys[int(answer) - 1] if answer.isdigit() and 1 <= int(answer) <= len(keys) else answer
    value = value or default
    if value not in options:
        raise ValueError(f"unknown {attr} '{value}'; choose one of {', '.join(options)}")
    return value


def cmd_recommend(args) -> int:
    ask = _interactive(args)
    total, gpu_name, other, gpu_desc, available, gpu_flags = _gpu_setup(args, ask)
    tp_options = _tp_options(args, available, gpu_desc, ask)
    tasks = {k: v[1] for k, v in TASKS.items()} | {k: "(not served by vLLM)" for k in NOT_VLLM_TASKS}
    task = _choose(args, "task", "What will the model do?", tasks, "chat", ask)
    if task in NOT_VLLM_TASKS:
        print(f"kvplan: {NOT_VLLM_TASKS[task]}", file=sys.stderr)
        return 2
    priority = _choose(args, "priority", "What matters more?", PRIORITIES, "balanced", ask)
    length = args.max_model_len
    if length is None and ask:
        answer = _ask("Context length per request, in tokens", "8192")
        length = int(answer) if answer.isdigit() else None
    length = length or 8192
    token = args.token or find_token()

    def cfg_for_tp(tp):
        return ServeConfig(gpu_total_bytes=total, tp=tp, gpu_memory_utilization=args.gpu_memory_utilization,
                           other_process_bytes=other, gpu_name=gpu_name)

    def progress(i, model_id):
        if sys.stderr.isatty() and not args.json:
            print(f"\r\033[K  checking {i}: {model_id}", end="", file=sys.stderr, flush=True)

    res = recommend(task, priority, cfg_for_tp, tp_options, max_model_len=length,
                    min_concurrency=args.min_concurrency, limit=args.limit, token=token,
                    include_gated=bool(token) and not args.no_gated, progress=progress)
    if sys.stderr.isatty() and not args.json:
        print("\r\033[K", end="", file=sys.stderr)
    if args.json:
        print(json.dumps({"task": task, "priority": priority, "max_model_len": length, "gpu": gpu_desc,
                          "tp_options": tp_options, "checked": res.checked, "considered": res.considered,
                          "left_out": res.skipped,
                          "recommendations": [{"model": r.model.id, "parameters": r.model.total_parameters,
                                               "downloads": r.model.downloads, "tp": r.tp,
                                               "weights_bytes_per_gpu": r.weights_bytes, "kv_tokens": r.kv_tokens,
                                               "max_concurrency": r.max_concurrency,
                                               "max_model_len": r.max_model_len, "warnings": r.notes}
                                              for r in res.recommendations]}, indent=2))
        return 0 if res.recommendations else 1
    recs = res.recommendations
    gpus = "/".join(str(t) for t in tp_options)
    print(f"\n{task} models for {gpus} x {gpu_desc}, {priority} ({PRIORITIES[priority]}), "
          f"{length:,}-token context\n")
    if not recs:
        print("  none of the checked models should start with these settings; try a shorter context, more GPUs "
              "or --priority speed")
    else:
        print(f"  {'#':>2}  {'model':<52} {'size':>6} {'GPUs':>4} {'weights/GPU':>11} {'KV tokens':>11} "
              f"{'requests at once':>16}")
        for i, r in enumerate(recs, 1):
            print(f"  {i:>2}  {r.model.id:<52} {_fmt_count(r.model.total_parameters):>6} {r.tp:>4} "
                  f"{_gib(r.weights_bytes):>11} {r.kv_tokens:>11,} {r.max_concurrency:>15.1f}x"
                  + ("  (gated)" if r.model.gated else ""))
            for n in r.notes:
                print(f"      {n}")
    print(f"\n  Checked {res.checked} of {res.considered} candidates from Hugging Face's most downloaded "
          f"{TASKS[task][0]} models. Ranked by size and downloads, not by benchmark scores.")
    if res.skipped:
        print("  Left out: " + ", ".join(f"{n} {why}" for why, n in res.skipped.items()))
    if not token:
        print("  Gated models (Llama, Gemma, ...) need a token: run `kvplan login`.")
    if not recs:
        return 1

    def check_argv(r):
        return ["check", r.model.id, "--max-model-len", str(r.max_model_len), "-tp", str(r.tp)] + gpu_flags

    if not ask:
        print(f"\n  Details: kvplan {' '.join(check_argv(recs[0]))}")
        print(f"  Download: kvplan download {recs[0].model.id}")
        return 0
    while True:
        answer = _ask("\nNumber for full details, d<number> to download, Enter to finish").lower()
        if not answer:
            return 0
        download = answer.startswith("d")
        num = answer[1:].strip() if download else answer
        if not num.isdigit() or not 1 <= int(num) <= len(recs):
            print(f"  pick 1 to {len(recs)}")
            continue
        r = recs[int(num) - 1]
        print()
        main(["download", r.model.id] if download else check_argv(r))


def _exit_code(p) -> int:
    """0: vLLM should start; 1: it will not; 3: start not confirmed (2 is a kvplan error)."""
    return {True: 0, False: 1, None: 3}[p.starts]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        commands = {"inspect": cmd_inspect_or_download, "download": cmd_inspect_or_download, "check": cmd_check,
                    "search": cmd_search, "login": cmd_login, "recommend": cmd_recommend}
        return commands[args.command](args)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (SourceError, UnsupportedModel, ValueError) as e:
        print(f"kvplan: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
