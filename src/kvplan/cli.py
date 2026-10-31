"""Command line: `kvplan check MODEL --gpu H100-80GB`.

The commands are thin layers over kvplan.api: they read options (from the command line and an
optional config file), call the API, and render the result as text, JSON, JSON Lines, CSV or
Markdown, to the terminal or a file.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

from . import __version__, api, config, hub, output, style
from .gpus import GPUS_MIB
from .recommender import NOT_VLLM_TASKS, PRIORITIES, TASKS
from .sources import SourceError
from .vllm import DEFAULT_UTILIZATION, LARGEST_MEASURED_ENCODER_PEAK, GiB


def _gib(b) -> str:
    return f"{b / GiB:.2f} GiB"


_color = False   # whether the text being rendered goes to a terminal that shows colour


def _c(text: str, *styles: str) -> str:
    return style.paint(text, *styles, on=_color)


def _parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    p = argparse.ArgumentParser(prog="kvplan", description="Plan vLLM deployments without the GPU.")
    p.add_argument("--version", action="version", version=f"kvplan {__version__}")
    p.add_argument("--color", choices=style.MODES, default=None,
                   help="colour the text output: auto (the default: only on a terminal), always or never; "
                        "NO_COLOR also turns it off")
    p.add_argument("--config", help="read default options from this TOML or JSON file (default: $KVPLAN_CONFIG, "
                                    "or config.toml or config.json in ~/.config/kvplan, if present)")
    sub = p.add_subparsers(dest="command", required=True)

    def token_arg(sp):
        sp.add_argument("--token", help="Hugging Face token (default: HF_TOKEN, or the token saved by "
                                        "`kvplan login` or `huggingface-cli login`)")

    def output_args(sp):
        sp.add_argument("--format", choices=output.FORMATS, default=None,
                        help="output format (default: text, or the one the --output file name ends in)")
        sp.add_argument("--json", action="store_true", help="same as --format json")
        sp.add_argument("-o", "--output", metavar="PATH",
                        help="write the result to this file, or into this directory (a file named after the model)")
        sp.add_argument("--append", action="store_true",
                        help="with --output and csv or jsonl: add to the file instead of replacing it")

    def model_args(sp, required=True):
        sp.add_argument("model", nargs=None if required else "?",
                        help="Hugging Face repo id (owner/name), a model name to look up "
                             "(e.g. Qwen2.5-7B-Instruct), or a local directory or file")
        sp.add_argument("--revision", default="main")
        token_arg(sp)
        sp.add_argument("--gguf", nargs="?", const="", default=None, metavar="QUANT",
                        help="read a GGUF file instead of safetensors (optionally pick a quant, e.g. Q4_K_M)")
        sp.add_argument("--no-input", action="store_true", help="never ask questions")

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
    model_args(c, required=False)
    c.add_argument("--models", metavar="FILE",
                   help="check many models: a text file (one per line), a CSV with a 'model' column and optional "
                        "option columns (gpu, tp, max_model_len, ...), or a JSON list; '-' reads standard input")
    gpu_args(c)
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
    c.add_argument("--download", action="store_true",
                   help="download the model when the check says vLLM should start")
    output_args(c)

    i = sub.add_parser("inspect", help="show the model's attention geometry, weights and KV bytes per token")
    model_args(i)
    i.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    i.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])
    output_args(i)

    s = sub.add_parser("search", help="find models on Hugging Face by a loose name, e.g. 'llama 3.1 8b'")
    s.add_argument("query", nargs="+")
    s.add_argument("--limit", type=int, default=10)
    token_arg(s)
    output_args(s)

    d = sub.add_parser("download", help="download a model's weights, configs and tokenizer for vLLM")
    model_args(d)
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
    output_args(r)
    return p, {"check": c, "inspect": i, "search": s, "download": d, "login": lg, "recommend": r}


# --- helpers ---------------------------------------------------------------------------------------

def _fmt(args) -> str:
    return output.choose_format(getattr(args, "format", None), getattr(args, "json", False),
                                getattr(args, "output", None))


def _interactive(args) -> bool:
    return _fmt(args) == "text" and not getattr(args, "output", None) and not getattr(args, "no_input", False) \
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


def _with_suggestions(args, call):
    """Run call(model). When the model is not found, list similar ones, and in a terminal let the
    user pick one and try again."""
    while True:
        try:
            return call(args.model)
        except api.ModelNotFound as e:
            if not e.suggestions:
                raise
            if not _interactive(args):
                raise SourceError(f"{e.problem}. Did you mean:\n{_format_matches(e.suggestions)}") from None
            print(f"{e.problem}. Similar models:")
            print(_format_matches(e.suggestions))
            answer = _ask("Pick a number to use it (Enter to stop)")
            if not (answer.isdigit() and 1 <= int(answer) <= len(e.suggestions)):
                raise SourceError(e.problem) from None
            args.model = e.suggestions[int(answer) - 1].id


def _has_hf_hub() -> bool:
    import importlib.util
    return importlib.util.find_spec("huggingface_hub") is not None


def _emit(args, text_render, data, rows, name: str) -> None:
    """Print or write the result in the chosen format."""
    global _color
    fmt = _fmt(args)
    if fmt == "text":
        _color = not getattr(args, "output", None) and style.enabled(sys.stdout)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                text_render()
        finally:
            _color = False
        body = buf.getvalue()
    else:
        body = output.render(fmt, data, rows)
    if getattr(args, "output", None):
        written = output.write(args.output, body, fmt, name, append=args.append)
        print(f"kvplan: wrote {written}", file=sys.stderr)
    else:
        sys.stdout.write(body if not body or body.endswith("\n") else body + "\n")


# --- check -----------------------------------------------------------------------------------------

_CHECK_OPTIONS = {   # api.check keyword -> args attribute
    "gpu": "gpu", "gpu_mem_gib": "gpu_mem_gib", "gpu_index": "gpu_index", "tp": "tensor_parallel_size",
    "max_model_len": "max_model_len", "max_num_seqs": "max_num_seqs",
    "max_num_batched_tokens": "max_num_batched_tokens", "gpu_memory_utilization": "gpu_memory_utilization",
    "kv_cache_dtype": "kv_cache_dtype", "block_size": "block_size", "enforce_eager": "enforce_eager",
    "other_gpu_mem_gib": "other_gpu_mem_gib", "limit_mm_per_prompt": "limit_mm_per_prompt",
    "mm_processor_kwargs": "mm_processor_kwargs", "skip_mm_profiling": "skip_mm_profiling",
    "overhead_gib": "overhead_gib", "revision": "revision", "token": "token", "gguf": "gguf",
}


def _check_kwargs(args) -> dict:
    kw = {k: getattr(args, a) for k, a in _CHECK_OPTIONS.items()}
    kw["async_scheduling"] = not args.no_async_scheduling
    return kw


def _print_check(r: api.CheckResult, utilization: float) -> None:
    p, w, o, loaded = r.plan, r.weights, r.plan.overheads, r.loaded
    print(f"{r.model} on {p.tp} x {r.gpu.description} "
          f"(max_model_len {p.max_model_len}, max_num_seqs {p.max_num_seqs}, "
          f"gpu_memory_utilization {utilization})")
    print()
    for g in r.gpu.detected:
        used = "unknown" if g.used_mib is None else f"{g.used_mib:,} MiB"
        reserved = f", {g.reserved_mib:,} reserved by the driver" if g.reserved_mib else ""
        print(f"  gpu {g.index:<10} {g.name}: {g.total_mib:,} MiB total{reserved}, {used} in use (nvidia-smi)")
    print(f"  budget         {_gib(p.requested_bytes)} of {_gib(p.gpu_total_bytes)} per GPU")
    print(f"  weights        {_gib(w.per_rank_bytes)} per GPU (exact, from the weight files)")
    if o.user_override:
        print(f"  runtime        {_gib(o.total)} (your --overhead-gib)")
    else:
        kind = loaded.spec.encoder.kind if loaded.spec.encoder else ""
        print(f"  runtime        {_gib(o.total)} estimated: activation peak {_gib(o.activation_bytes)}, "
              f"torch.compile {_gib(o.compile_bytes)}, CUDA graphs {_gib(o.cudagraph_bytes)}, "
              f"other {_gib(o.non_torch_bytes)}"
              + (f", {kind} encoder profiling {_gib(o.encoder_bytes)}"
                 + (" (approximate)" if o.encoder_approximate else "") if o.encoder_bytes else ""))
    print(f"  KV cache       {_gib(max(p.available_kv_bytes, 0))} available "
          f"({p.kv_bytes_per_token:,} bytes per token per GPU, {p.kv_cache_dtype}, block size {p.block_size})")
    if p.kv is not None:
        print(f"  capacity       {p.kv.kv_tokens:,} tokens; max concurrency at {p.max_model_len:,} tokens per "
              f"request: {p.kv.max_concurrency:.2f}x")
    print()
    result = _c(p.verdict, style.verdict_color(p.starts), "bold")
    if p.starts is None:
        result += ": kvplan could not check " + "; ".join(p.unconfirmed)
        if p.start_margin_bytes is not None:
            result += f". Under kvplan's estimates {_gib(max(p.start_margin_bytes, 0))} would be left at the start check"
            if any("encoder" in u for u in p.unconfirmed):
                result += (f"; the largest encoder startup peak measured so far was "
                           f"{_gib(LARGEST_MEASURED_ENCODER_PEAK)}")
    print("  " + _c("RESULT", "bold") + "         " + result)
    for c in p.checks:
        tag = {"error": "error", "warning": "warning", "info": "note"}[c.severity]
        print("  " + _c(f"{tag:<14}", style.SEVERITY_COLOR[c.severity]) + " " + c.message)
    for s in p.suggestions:
        print("  " + _c("try", "cyan") + f"            {s}")
    notes = loaded.spec.notes + w.notes
    if o.encoder_note and not o.user_override:
        notes.append(o.encoder_note)
    if not o.user_override and not o.calibrated:
        notes.append("runtime overheads for this model are a rough estimate: they were calibrated on dense models "
                     "only; pass --overhead-gib to override")
    for n in notes:
        print(_c(f"  note           {n}", "dim"))


def _exit_code(starts) -> int:
    """0: vLLM should start; 1: it will not; 3: start not confirmed (2 is a kvplan error)."""
    return {True: 0, False: 1, None: 3}[starts]


def cmd_check(args) -> int:
    if args.models:
        if args.model:
            raise ValueError("give either a MODEL or --models FILE, not both")
        return _check_many(args)
    if not args.model:
        raise ValueError("check needs a MODEL, or --models FILE to check several")
    r = _with_suggestions(args, lambda m: api.check(m, **_check_kwargs(args)))
    for n in r.notes:
        print(f"kvplan: {n}", file=sys.stderr)
    local = os.path.exists(r.model)

    def text():
        _print_check(r, args.gpu_memory_utilization)
        if r.starts and not local and not args.download:
            print()
            print("  " + _c("download", "cyan") + f"       kvplan download {r.model}   "
                  "(or add --download to this check)")
            print("  " + _c("serve", "cyan") + f"          {r.serve_command}")

    _emit(args, text, r.to_dict(), [r.row()], r.model)
    if args.download and r.starts and not local:
        to_stdout = _fmt(args) == "text" and not args.output
        log = print if to_stdout else (lambda m: print(m, file=sys.stderr))
        if to_stdout:
            print()
        where = hub.download(r.model, r.loaded.files, r.loaded.repo_files, revision=args.revision,
                             token=args.token, log=log)
        log(f"  downloaded     {r.model} to {where}")
        target = r.model if _has_hf_hub() else where
        log(f"  serve          {r.serve_command.replace(r.model, target, 1)}")
    return _exit_code(r.starts)


_SEVERITY = {0: 0, 3: 1, 1: 2, 2: 3}   # exit code -> how bad, for the batch exit code


def _check_many(args) -> int:
    entries = config.read_model_list(args.models)
    base = _check_kwargs(args)
    show_progress = sys.stderr.isatty() and _fmt(args) == "text"
    results, rows, worst = [], [], 0
    for n, entry in enumerate(entries, 1):
        if show_progress:
            print(f"\r\033[K  checking {n}/{len(entries)}: {entry['model']}", end="", file=sys.stderr, flush=True)
        try:
            r = api.check(entry["model"], **(base | {k: v for k, v in entry.items() if k != "model"}))
            results.append(r.to_dict())
            rows.append(r.row())
            code = _exit_code(r.starts)
        except api.KvplanError as e:
            message = str(e).splitlines()[0]
            results.append({"model": entry["model"], "error": str(e)})
            rows.append({"model": entry["model"], "verdict": "error", "problems": message})
            code = 2
        worst = max(worst, _SEVERITY[code])
    if show_progress:
        print("\r\033[K", end="", file=sys.stderr)

    def text():
        verdict_starts = {"vLLM should start": True, "vLLM will not start": False}

        def paint(col, cell, row):
            if col != "verdict":
                return cell
            return _c(cell, "red" if row.get("verdict") == "error" else
                      style.verdict_color(verdict_starts.get(row.get("verdict"))))

        print(output.table(rows, ["model", "gpu", "tp", "max_model_len", "verdict", "weights_gib", "kv_tokens",
                                  "max_concurrency", "problems"], paint=paint))

    _emit(args, text, results, rows, "kvplan-check")
    return {v: k for k, v in _SEVERITY.items()}[worst]


# --- inspect, download, search, login ---------------------------------------------------------------

def cmd_inspect(args) -> int:
    d = _with_suggestions(args, lambda m: api.inspect(m, tp=args.tensor_parallel_size,
                                                      kv_cache_dtype=args.kv_cache_dtype, revision=args.revision,
                                                      token=args.token, gguf=args.gguf))
    for n in d["notes"]:
        print(f"kvplan: {n}", file=sys.stderr)
    s, w, tp = d["spec"], d["weights"], d["tp"]

    def text():
        kinds = {k: s["layer_types"].count(k) for k in dict.fromkeys(s["layer_types"])}
        attn = "MLA (latent KV)" if s["kv_lora_rank"] is not None else (
            f"{s['num_heads']} heads, {s['num_kv_heads']} KV heads, head_dim {s['head_dim']}")
        print(d["model"])
        print(f"  architecture   {s['model_type']} (read from {s['source']})")
        print(f"  layers         {s['num_layers']}: " + ", ".join(f"{n} {k}" for k, n in kinds.items())
              + (f" (window {s['sliding_window']})" if s["sliding_window"] else ""))
        print(f"  attention      {attn}")
        if s["num_experts"]:
            print(f"  experts        {s['num_experts']} (all resident in GPU memory)")
        ctx, default = s["max_position_embeddings"], s["default_max_model_len"]
        if ctx and default and default != ctx:
            print(f"  context        up to {ctx} (vLLM defaults to {default})")
        else:
            print(f"  context        {ctx or 'unknown'}")
        print(f"  KV per token   {d['kv_bytes_per_token']:,} bytes per GPU at tp={tp} ({d['kv_cache_dtype']})")
        print(f"  weights        {_gib(w['per_rank_bytes'])} per GPU at tp={tp} "
              f"(checkpoint {_gib(w['total_checkpoint_bytes'])} in {len(d['files'])} file(s))")
        for n in s["notes"] + w["notes"]:
            print(f"  note           {n}")

    row = {"model": d["model"], "architecture": s["model_type"], "layers": s["num_layers"],
           "heads": s["num_heads"], "kv_heads": s["num_kv_heads"], "head_dim": s["head_dim"],
           "experts": s["num_experts"], "context": s["max_position_embeddings"], "tp": tp,
           "kv_bytes_per_token": d["kv_bytes_per_token"], "weights_gib": round(w["per_rank_bytes"] / GiB, 2)}
    _emit(args, text, d, [row], d["model"])
    return 0


def cmd_download(args) -> int:
    def call(model):
        resolved, note = api.resolve_model(model, args.token)
        if note:
            print(f"kvplan: {note}", file=sys.stderr)
        args.model = resolved
        return api.download(resolved, local_dir=args.local_dir, revision=args.revision, token=args.token,
                            gguf=args.gguf, log=print)

    where = _with_suggestions(args, call)
    print(f"  downloaded     {args.model} to {where}")
    print(f"serve it with: vllm serve {where if args.local_dir or not _has_hf_hub() else args.model}")
    return 0


def cmd_search(args) -> int:
    query = " ".join(args.query)
    found = api.search(query, limit=args.limit, token=args.token)
    if not found:
        print(f"nothing on Hugging Face matches '{query}'; try fewer or shorter words", file=sys.stderr)
        return 1

    def text():
        print(_format_matches([hub.HubModel(id=m["model"], downloads=m["downloads"], gated=m["gated"],
                                            parameters={"BF16": m["parameters"]} if m["parameters"] else {})
                               for m in found]))
        print(f"\nnext: kvplan check {found[0]['model']}")

    _emit(args, text, found, found, "kvplan-search")
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


# --- recommend -------------------------------------------------------------------------------------

def _choose(args, attr: str, question: str, options: dict, default: str, ask: bool) -> str:
    value = getattr(args, attr)
    if value is None and ask:
        print(question)
        keys = list(options)
        for i, k in enumerate(keys, 1):
            print(f"  {i}. {k:<17} {options[k]}")
        answer = _ask("Choose", default)
        value = keys[int(answer) - 1] if answer.isdigit() and 1 <= int(answer) <= len(keys) else answer
    value = value or default
    if value not in options:
        raise ValueError(f"unknown {attr} '{value}'; choose one of {', '.join(options)}")
    return value


def cmd_recommend(args) -> int:
    ask = _interactive(args)
    gpus = args.gpus
    if gpus is None and ask:
        g = api.resolve_gpu(args.gpu, args.gpu_mem_gib, args.gpu_index, None, args.other_gpu_mem_gib)
        if len(g.detected) > 1:
            gpus = _ask(f"This machine has {len(g.detected)} x {g.description}. Use 1 GPU, {len(g.detected)}, "
                        "or 'any' (the fewest that work)", "any")
        elif not g.detected:
            gpus = _ask(f"How many {g.description} GPUs will you use", "1")
    tasks = {k: v[1] for k, v in TASKS.items()} | {k: "(not served by vLLM)" for k in NOT_VLLM_TASKS}
    task = _choose(args, "task", "What will the model do?", tasks, "chat", ask)
    if task in NOT_VLLM_TASKS:
        raise ValueError(NOT_VLLM_TASKS[task])
    priority = _choose(args, "priority", "What matters more?", PRIORITIES, "balanced", ask)
    length = args.max_model_len
    if length is None and ask:
        answer = _ask("Context length per request, in tokens", "8192")
        length = int(answer) if answer.isdigit() else None
    show_progress = sys.stderr.isatty() and _fmt(args) == "text"

    def progress(i, model_id):
        if show_progress:
            print(f"\r\033[K  checking {i}: {model_id}", end="", file=sys.stderr, flush=True)

    res = api.recommend(task=task, priority=priority, gpu=args.gpu, gpu_mem_gib=args.gpu_mem_gib,
                        gpu_index=args.gpu_index, gpus=gpus or "1", max_model_len=length or 8192,
                        min_concurrency=args.min_concurrency, gpu_memory_utilization=args.gpu_memory_utilization,
                        other_gpu_mem_gib=args.other_gpu_mem_gib, limit=args.limit,
                        include_gated=not args.no_gated, token=args.token, progress=progress)
    if show_progress:
        print("\r\033[K", end="", file=sys.stderr)
    recs = res["recommendations"]

    def check_argv(r):
        return ["check", r["model"], "--max-model-len", str(r["max_model_len"]), "-tp", str(r["tp"])] \
            + res["gpu_flags"]

    def text():
        sizes = "/".join(str(t) for t in res["tp_options"])
        print("\n" + _c(f"{task} models for {sizes} x {res['gpu']}, {priority} ({PRIORITIES[priority]}), "
                        f"{res['max_model_len']:,}-token context", "bold") + "\n")
        if not recs:
            print(_c("  none of the checked models should start with these settings; try a shorter context, more "
                     "GPUs or --priority speed", "yellow"))
        else:
            print(f"  {'#':>2}  {'model':<52} {'size':>6} {'GPUs':>4} {'weights/GPU':>11} {'KV tokens':>11} "
                  f"{'requests at once':>16}")
            for n, r in enumerate(recs, 1):
                print(f"  {n:>2}  {r['model']:<52} {_fmt_count(r['parameters']):>6} {r['tp']:>4} "
                      f"{r['weights_gib']:>7.2f} GiB {r['kv_tokens']:>11,} {r['max_concurrency']:>15.1f}x"
                      + _c("  (gated)" if r["gated"] else "", "yellow"))
                for w in r["warnings"]:
                    print(_c(f"      {w}", "yellow"))
        print(f"\n  Checked {res['checked']} of {res['considered']} candidates from Hugging Face's most downloaded "
              f"{res['pipeline_tag']} models. Ranked by size and downloads, not by benchmark scores.")
        if res["left_out"]:
            print("  Left out: " + ", ".join(f"{n} {why}" for why, n in res["left_out"].items()))
        if not res["has_token"]:
            print("  Gated models (Llama, Gemma, ...) need a token: run `kvplan login`.")
        if recs and not ask:
            print(f"\n  Details: kvplan {' '.join(check_argv(recs[0]))}")
            print(f"  Download: kvplan download {recs[0]['model']}")

    rows = [{k: v for k, v in r.items() if k != "warnings"} | {"warnings": "; ".join(r["warnings"])} for r in recs]
    _emit(args, text, res, rows, "kvplan-recommend")
    if not recs:
        return 1
    while ask:
        answer = _ask("\nNumber for full details, d<number> to download, Enter to finish").lower()
        if not answer:
            break
        download = answer.startswith("d")
        num = answer[1:].strip() if download else answer
        if not num.isdigit() or not 1 <= int(num) <= len(recs):
            print(f"  pick 1 to {len(recs)}")
            continue
        r = recs[int(num) - 1]
        print()
        main(["download", r["model"]] if download else check_argv(r))
    return 0


# --- entry point -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser, subparsers = _parser()
    args = parser.parse_args(argv)
    try:
        settings = config.load_config(args.config)
        style.set_mode(args.color or settings.get("color") or "auto")
        if settings:
            config.apply(subparsers[args.command], args.command, settings)
            args = parser.parse_args(argv)
        commands = {"check": cmd_check, "inspect": cmd_inspect, "download": cmd_download, "search": cmd_search,
                    "login": cmd_login, "recommend": cmd_recommend}
        return commands[args.command](args)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except api.KvplanError as e:
        print(style.paint(f"kvplan: {e}", "red", on=style.enabled(sys.stderr)), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
