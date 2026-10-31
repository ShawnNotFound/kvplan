"""Python API: call kvplan from your own code.

    import kvplan
    r = kvplan.check("Qwen/Qwen2.5-7B-Instruct", gpu="A6000", max_model_len=8192)
    r.starts, r.kv_tokens, r.to_dict()

Every function returns plain data (dataclasses with to_dict(), or dicts and lists) and raises
kvplan.KvplanError subclasses on bad input or unreadable models; nothing is printed or asked.
The command line is a thin layer over these functions, and `--format json` prints the same
dictionaries.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field

from . import hub
from .gpus import DetectedGPU, GPUDetectError, gpu_total_bytes, query_nvidia_smi, select_gpus
from .kv import kv_layout, resolve_kv_cache_dtype
from .loader import LoadedModel, load
from .model import UnsupportedModel
from .recommender import NOT_VLLM_TASKS, PRIORITIES, TASKS
from .recommender import recommend as _recommend
from .sources import RepoNotFoundError, SourceError, find_token, open_source
from .vllm import DEFAULT_UTILIZATION, GiB, Plan, ServeConfig, plan
from .weights import WeightBreakdown, weight_bytes_per_rank

KvplanError = (SourceError, UnsupportedModel, ValueError)   # what kvplan raises for bad input or models


class ModelNotFound(SourceError):
    """No model with this name or repo id; `suggestions` holds the closest popular ones."""

    def __init__(self, problem: str, suggestions: list[hub.HubModel]):
        super().__init__(problem if suggestions else
                         f"{problem}, and a search found nothing similar; try `kvplan search` with fewer words")
        self.problem = problem
        self.suggestions = suggestions


def resolve_model(model: str, token: str | None = None) -> tuple[str, str | None]:
    """(repo id or path, note). A name without an owner ('Qwen2.5-7B-Instruct') becomes the most
    downloaded repo with exactly that name; raises ModelNotFound when there is none."""
    if os.path.exists(model) or re.fullmatch(r"[\w.\-]+/[\w.\-]+", model):
        return model, None
    found, matches = hub.resolve_name(model, token)
    if found:
        return found, f"using {found}, the most downloaded repo named '{model}'"
    raise ModelNotFound(f"no Hugging Face repo named '{model}'", matches)


def load_model(model: str, revision: str = "main", token: str | None = None,
               gguf: str | None = None) -> LoadedModel:
    """Read a model's config and weight headers; raises ModelNotFound with suggestions for a missing repo."""
    try:
        return load(open_source(model, revision, token), gguf=gguf)
    except RepoNotFoundError as e:
        raise ModelNotFound(str(e), hub.similar_models(e.repo_id, limit=8, token=token)) from None


@dataclass
class GPUChoice:
    total_bytes: int            # per GPU, as CUDA reports it
    name: str                   # preset or nvidia-smi name ("" when unknown or mixed)
    other_bytes: int            # memory other processes already hold on the busiest GPU
    description: str
    detected: list[DetectedGPU] = field(default_factory=list)   # with gpu="auto"
    flags: list[str] = field(default_factory=list)              # the matching `kvplan check` options


def _parse_indices(indices) -> list[int] | None:
    if indices is None:
        return None
    if isinstance(indices, str):
        try:
            indices = [int(x) for x in indices.split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"--gpu-index must be comma-separated GPU numbers such as 0 or 0,1, "
                             f"not '{indices}'") from None
    indices = list(indices)
    if not indices or any(i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError(f"--gpu-index must list distinct GPU numbers such as 0 or 0,1, not '{indices}'")
    return indices


def detect_gpus(defaulted: bool = False) -> list[DetectedGPU]:
    try:
        return query_nvidia_smi()
    except GPUDetectError as e:
        if defaulted:
            reason = str(e).split("; pass --gpu", 1)[0]
            raise GPUDetectError(f"no --gpu given, and this machine's GPUs could not be read ({reason}). Name the "
                                 "GPU you will run on, e.g. --gpu A6000, or give its memory with --gpu-mem-gib")\
                from None
        raise


def resolve_gpu(gpu: str | None = None, gpu_mem_gib: float | None = None, gpu_index=None, tp: int | None = 1,
                other_gpu_mem_gib: float | None = None) -> GPUChoice:
    """The GPU to plan for: a preset name, an exact memory size, or (the default) this machine's GPUs.
    With tp=None and gpu="auto", every detected GPU (or the ones in gpu_index) is kept."""
    for flag, v in (("--gpu-mem-gib", gpu_mem_gib), ("--other-gpu-mem-gib", other_gpu_mem_gib)):
        if v is not None and not math.isfinite(v):
            raise ValueError(f"{flag} must be a number of GiB, not {v}")
    other = int(other_gpu_mem_gib * GiB) if other_gpu_mem_gib is not None else 0
    if gpu_index is not None and (gpu_mem_gib is not None or (gpu or "auto").lower() != "auto"):
        raise ValueError("--gpu-index only applies to --gpu auto")
    if gpu_mem_gib is not None:
        return GPUChoice(int(gpu_mem_gib * GiB), "", other, f"{gpu_mem_gib} GiB GPU",
                         flags=["--gpu-mem-gib", str(gpu_mem_gib)])
    if gpu and gpu.lower() != "auto":
        return GPUChoice(gpu_total_bytes(gpu), gpu, other, gpu, flags=["--gpu", gpu])
    found = detect_gpus(defaulted=gpu is None)
    indices = _parse_indices(gpu_index)
    chosen = select_gpus(found, tp or len(indices or found), indices)
    # vLLM sizes every rank like the smallest GPU; the busiest GPU decides whether it starts.
    total = min(g.cuda_total_mib for g in chosen) * 2**20
    if other_gpu_mem_gib is None:
        other = max(g.used_mib or 0 for g in chosen) * 2**20
    names = sorted({g.name for g in chosen})
    flags = ["--gpu", "auto"] + (["--gpu-index", ",".join(str(i) for i in indices)] if indices else [])
    return GPUChoice(total, names[0] if len(names) == 1 else "", other, " / ".join(names), chosen, flags)


def _mm_limits(value) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('--limit-mm-per-prompt must be a JSON object such as \'{"image": 1, "video": 0}\'')
    # vLLM also accepts {"image": {"count": 1, ...}}.
    return {k: int(v["count"] if isinstance(v, dict) else v) for k, v in value.items()}


def serve_command(target: str, tp: int = 1, max_model_len: int | None = None, max_num_seqs: int | None = None,
                  gpu_memory_utilization: float = DEFAULT_UTILIZATION, kv_cache_dtype: str = "auto",
                  enforce_eager: bool = False) -> str:
    parts = ["vllm serve", target]
    if tp != 1:
        parts.append(f"--tensor-parallel-size {tp}")
    if max_model_len:
        parts.append(f"--max-model-len {max_model_len}")
    if max_num_seqs:
        parts.append(f"--max-num-seqs {max_num_seqs}")
    if gpu_memory_utilization != DEFAULT_UTILIZATION:
        parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")
    if kv_cache_dtype != "auto":
        parts.append(f"--kv-cache-dtype {kv_cache_dtype}")
    if enforce_eager:
        parts.append("--enforce-eager")
    return " ".join(parts)


@dataclass
class CheckResult:
    model: str
    gpu: GPUChoice
    plan: Plan
    weights: WeightBreakdown
    loaded: LoadedModel
    serve_command: str
    notes: list[str] = field(default_factory=list)   # e.g. which repo a bare name resolved to

    @property
    def starts(self) -> bool | None:
        """True: vLLM should start; False: it will not; None: start not confirmed."""
        return self.plan.starts

    @property
    def kv_tokens(self) -> int:
        return self.plan.kv.kv_tokens if self.plan.kv else 0

    def to_dict(self) -> dict:
        d = {"model": self.model, "gpu": self.gpu.description} | self.plan.to_dict()
        d |= {"spec_notes": self.loaded.spec.notes, "other_process_bytes": self.gpu.other_bytes,
              "serve_command": self.serve_command if self.starts else None, "notes": self.notes}
        if self.gpu.detected:
            d["detected_gpus"] = [g.__dict__ | {"cuda_total_mib": g.cuda_total_mib} for g in self.gpu.detected]
        return d

    def row(self) -> dict:
        """One flat row for CSV or a table."""
        p, kv = self.plan, self.plan.kv
        return {"model": self.model, "gpu": self.gpu.description, "tp": p.tp, "max_model_len": p.max_model_len,
                "verdict": p.verdict, "starts": p.starts,
                "weights_gib": round(self.weights.per_rank_bytes / GiB, 2),
                "kv_cache_gib": round(max(p.available_kv_bytes, 0) / GiB, 2),
                "kv_tokens": kv.kv_tokens if kv else 0,
                "max_concurrency": round(kv.max_concurrency, 2) if kv else 0,
                "kv_cache_dtype": p.kv_cache_dtype,
                "problems": "; ".join(c.message for c in p.checks if c.severity != "info"),
                "serve_command": self.serve_command if p.starts else ""}


def check(model: str, *, gpu: str | None = None, gpu_mem_gib: float | None = None, gpu_index=None,
          tp: int = 1, max_model_len: int | None = None, max_num_seqs: int | None = None,
          max_num_batched_tokens: int | None = None, gpu_memory_utilization: float = DEFAULT_UTILIZATION,
          kv_cache_dtype: str = "auto", block_size: int | None = None, enforce_eager: bool = False,
          async_scheduling: bool = True, other_gpu_mem_gib: float | None = None, limit_mm_per_prompt=None,
          mm_processor_kwargs: dict | None = None, skip_mm_profiling: bool = False,
          overhead_gib: float | None = None, revision: str = "main", token: str | None = None,
          gguf: str | None = None) -> CheckResult:
    """Will vLLM start with this model and configuration, and how much KV cache will it get?

    Without gpu or gpu_mem_gib, kvplan plans for the GPUs on this machine (gpu="auto").
    """
    if overhead_gib is not None and not math.isfinite(overhead_gib):
        raise ValueError(f"--overhead-gib must be a number of GiB, not {overhead_gib}")
    # Resolve the GPU first: a typo or a missing driver should fail before any network access.
    g = resolve_gpu(gpu, gpu_mem_gib, gpu_index, tp, other_gpu_mem_gib)
    model, note = resolve_model(model, token)
    loaded = load_model(model, revision, token, gguf)
    cfg = ServeConfig(
        gpu_total_bytes=g.total_bytes, tp=tp, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens, gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype, block_size=block_size, enforce_eager=enforce_eager,
        async_scheduling=async_scheduling, other_process_bytes=g.other_bytes,
        overhead_bytes=int(overhead_gib * GiB) if overhead_gib is not None else None, gpu_name=g.name,
        mm_limits=_mm_limits(limit_mm_per_prompt),
        mm_max_pixels=(mm_processor_kwargs or {}).get("max_pixels"),
        mm_min_pixels=(mm_processor_kwargs or {}).get("min_pixels"),
        skip_mm_profiling=skip_mm_profiling,
    )
    cfg.validate()
    w = weight_bytes_per_rank(loaded.spec, loaded.tensors, tp)
    p = plan(loaded.spec, w, cfg, model)
    cmd = serve_command(model, tp, max_model_len, max_num_seqs, gpu_memory_utilization, kv_cache_dtype,
                        enforce_eager)
    return CheckResult(model, g, p, w, loaded, cmd, [note] if note else [])


def inspect(model: str, *, tp: int = 1, kv_cache_dtype: str = "auto", revision: str = "main",
            token: str | None = None, gguf: str | None = None) -> dict:
    """The model's geometry, KV bytes per token and weight sizes, without any GPU."""
    model, note = resolve_model(model, token)
    loaded = load_model(model, revision, token, gguf)
    w = weight_bytes_per_rank(loaded.spec, loaded.tensors, tp)
    layout = kv_layout(loaded.spec, tp, kv_cache_dtype)
    return {"model": model, "tp": tp, "kv_cache_dtype": resolve_kv_cache_dtype(loaded.spec, kv_cache_dtype),
            "kv_bytes_per_token": sum(layer.page_bytes for layer in layout.layers) // layout.block_size,
            "spec": asdict(loaded.spec), "weights": asdict(w), "files": loaded.files,
            "notes": [note] if note else []}


def search(query: str, limit: int = 10, token: str | None = None) -> list[dict]:
    """Hugging Face repos matching a loose name ('llama 3.1 8b instruct'), most downloaded first."""
    return [{"model": m.id, "parameters": m.total_parameters, "downloads": m.downloads, "gated": m.gated,
             "architecture": m.architecture, "pipeline_tag": m.pipeline_tag}
            for m in hub.search_models(query, limit=limit, token=token)]


def download(model: str, *, local_dir: str | None = None, revision: str = "main", token: str | None = None,
             gguf: str | None = None, log=lambda message: None) -> str:
    """Download the files vLLM needs; returns the Hugging Face cache snapshot or the directory."""
    model, _ = resolve_model(model, token)
    if os.path.exists(model):
        raise ValueError(f"{model} is already on this machine")
    loaded = load_model(model, revision, token, gguf)
    return hub.download(model, loaded.files, loaded.repo_files, revision=revision, token=token,
                        local_dir=local_dir, log=log)


def tp_options(gpus, available: int | None) -> list[int]:
    """Tensor-parallel sizes to try: a number of GPUs, or 'any' for 1, 2, 4, 8 up to what is available."""
    choice = str(gpus if gpus is not None else 1).strip().lower()
    if choice == "any":
        return [n for n in (1, 2, 4, 8) if n <= (available or 8)]
    if not choice.isdigit() or int(choice) < 1:
        raise ValueError(f"--gpus must be a number of GPUs or 'any', not '{choice}'")
    if available and int(choice) > available:
        raise ValueError(f"--gpus {choice}, but this machine has {available} GPU(s)")
    return [int(choice)]


def recommend(*, task: str = "chat", priority: str = "balanced", gpu: str | None = None,
              gpu_mem_gib: float | None = None, gpu_index=None, gpus="1", max_model_len: int = 8192,
              min_concurrency: float = 1.0, gpu_memory_utilization: float = DEFAULT_UTILIZATION,
              other_gpu_mem_gib: float | None = None, limit: int = 10, include_gated: bool = True,
              token: str | None = None, progress=None) -> dict:
    """Popular Hugging Face models that vLLM should start on these GPUs, each fully checked.

    gpus is a number of GPUs or 'any' (the fewest of 1, 2, 4, 8 that work). include_gated only
    has an effect with a token.
    """
    if task in NOT_VLLM_TASKS:
        raise ValueError(NOT_VLLM_TASKS[task])
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}'; choose one of {', '.join(list(TASKS) + list(NOT_VLLM_TASKS))}")
    if priority not in PRIORITIES:
        raise ValueError(f"unknown priority '{priority}'; choose one of {', '.join(PRIORITIES)}")
    g = resolve_gpu(gpu, gpu_mem_gib, gpu_index, None, other_gpu_mem_gib)
    options = tp_options(gpus, len(g.detected) or None)
    token = token or find_token()

    def cfg_for_tp(tp):
        return ServeConfig(gpu_total_bytes=g.total_bytes, tp=tp, gpu_memory_utilization=gpu_memory_utilization,
                           other_process_bytes=g.other_bytes, gpu_name=g.name)

    res = _recommend(task, priority, cfg_for_tp, options, max_model_len=max_model_len,
                     min_concurrency=min_concurrency, limit=limit, token=token,
                     include_gated=bool(token) and include_gated, progress=progress)
    return {"task": task, "priority": priority, "max_model_len": max_model_len, "gpu": g.description,
            "gpu_flags": g.flags, "tp_options": options, "pipeline_tag": TASKS[task][0], "checked": res.checked,
            "considered": res.considered, "left_out": res.skipped, "has_token": bool(token),
            "recommendations": [
                {"model": r.model.id, "parameters": r.model.total_parameters, "downloads": r.model.downloads,
                 "gated": r.model.gated, "tp": r.tp, "weights_gib": round(r.weights_bytes / GiB, 2),
                 "kv_tokens": r.kv_tokens, "max_concurrency": round(r.max_concurrency, 2),
                 "max_model_len": r.max_model_len, "warnings": r.notes,
                 "serve_command": serve_command(r.model.id, r.tp, r.max_model_len,
                                                gpu_memory_utilization=gpu_memory_utilization)}
                for r in res.recommendations]}
