"""`kvplan recommend`: popular Hugging Face models that vLLM should start on your GPUs.

Candidates come from the Hub's model list for the task, most downloaded first. kvplan drops the
ones vLLM has no native implementation of or whose weights cannot fit, orders the rest by the
priority (quality: larger models first; speed: smaller ones; balanced: in between), and then runs
the full check on each until it has enough that should start. Model size is only a proxy for
quality and speed; the list is a starting point, not a benchmark.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .hub import HubModel, list_models
from .loader import load
from .model import UnsupportedModel
from .sources import SourceError, open_source
from .vllm import plan
from .vllm_architectures import VLLM_ARCHITECTURES
from .weights import weight_bytes_per_rank

# task -> (Hub pipeline tag, description, extra filter on the repo id)
TASKS = {
    "chat": ("text-generation", "chat and general text generation", None),
    "code": ("text-generation", "code generation", re.compile(r"cod(e|er|ing)", re.I)),
    "vision": ("image-text-to-text", "images and text in, text out", None),
    "audio": ("audio-text-to-text", "audio and text in, text out", None),
    "omni": ("any-to-any", "several input types (text, image, audio, video)", None),
}
NOT_VLLM_TASKS = {
    "image-generation": "image generation models (Stable Diffusion, FLUX, Qwen-Image) are not served by vLLM; "
                        "they run in diffusers or ComfyUI, which kvplan does not plan for",
    "embedding": "kvplan does not plan embedding models yet",
}
PRIORITIES = {
    "quality": "larger models first: usually more capable, fewer requests at once",
    "balanced": "mid-sized models that leave room for many requests",
    "speed": "smaller models first: faster replies and more requests at once",
}
# Repo names that are not chat or instruction models, or are formats vLLM does not load directly.
_SKIP = re.compile(r"(gguf|mlx|onnx|-base$|-base-|exl2|awq-gemv|draft|eagle|-mtp$|reward|guard)", re.I)


@dataclass
class Recommendation:
    model: HubModel
    tp: int
    starts: bool | None
    verdict: str
    weights_bytes: int
    kv_tokens: int
    max_concurrency: float
    max_model_len: int
    notes: list[str] = field(default_factory=list)


@dataclass
class RecommendResult:
    recommendations: list[Recommendation]
    considered: int
    checked: int
    skipped: dict[str, int]


def _size_score(m: HubModel) -> float:
    return math.log2(max(m.checkpoint_bytes, 1))


def order_candidates(models: list[HubModel], priority: str, capacity_bytes: int) -> list[HubModel]:
    """Order by size for the priority, most downloaded first within a size class (a factor of 2)."""
    if priority == "quality":
        key = lambda m: (-round(_size_score(m)), -m.downloads)
    elif priority == "speed":
        key = lambda m: (round(_size_score(m)), -m.downloads)
    else:
        target = math.log2(max(capacity_bytes * 0.35, 1))   # about a third of the memory for weights
        key = lambda m: (round(abs(_size_score(m) - target)), -m.downloads)
    return sorted(models, key=key)


def candidates(task: str, *, token: str | None, include_gated: bool, budget_bytes: int, max_tp: int,
               pool: int = 150, lister=list_models) -> tuple[list[HubModel], dict[str, int]]:
    tag, _, name_filter = TASKS[task]
    if name_filter is not None:
        seen, models = set(), []
        for q in ("coder", "code"):
            for m in lister(search=q, pipeline_tag=tag, limit=pool, token=token):
                if m.id not in seen:
                    seen.add(m.id)
                    models.append(m)
    else:
        models = lister(pipeline_tag=tag, limit=pool, token=token)
    skipped = {"not a native vLLM architecture": 0, "no safetensors weights": 0, "gated (no token)": 0,
               "weights too large for the GPUs": 0, "GGUF, MLX, base or draft variants": 0, "too small (< 0.5B)": 0}
    out = []
    for m in models:
        if name_filter is not None and not name_filter.search(m.id):
            skipped["GGUF, MLX, base or draft variants"] += 1
        elif _SKIP.search(m.id):
            skipped["GGUF, MLX, base or draft variants"] += 1
        elif not m.parameters:
            skipped["no safetensors weights"] += 1
        elif m.architecture not in VLLM_ARCHITECTURES:
            skipped["not a native vLLM architecture"] += 1
        elif m.gated and not include_gated:
            skipped["gated (no token)"] += 1
        elif m.checkpoint_bytes > budget_bytes * max_tp:
            skipped["weights too large for the GPUs"] += 1
        elif m.total_parameters < 5e8:
            skipped["too small (< 0.5B)"] += 1
        else:
            out.append(m)
    return out, {k: v for k, v in skipped.items() if v}


def evaluate(m: HubModel, cfg_for_tp, tp_options: list[int], max_model_len: int, token: str | None,
             loader=None) -> Recommendation | None:
    """Run the full check, at the smallest tensor-parallel size that starts. None if it cannot be read."""
    try:
        loaded = (loader or (lambda mid: load(open_source(mid, token=token))))(m.id)
    except (SourceError, UnsupportedModel, ValueError):
        return None
    spec = loaded.spec
    length = min(max_model_len, spec.default_max_model_len or max_model_len)
    best = None
    for tp in tp_options:
        if spec.num_heads % tp:
            continue
        cfg = cfg_for_tp(tp)
        cfg.max_model_len = length
        try:
            w = weight_bytes_per_rank(spec, loaded.tensors, tp)
            p = plan(spec, w, cfg, m.id)
        except (ValueError, UnsupportedModel):
            continue
        rec = Recommendation(m, tp, p.starts, p.verdict, w.per_rank_bytes, p.kv.kv_tokens if p.kv else 0,
                             p.kv.max_concurrency if p.kv else 0.0, length,
                             [c.message for c in p.checks if c.severity != "info"])
        if p.starts:
            return rec
        best = best or rec
    return best


def recommend(task: str, priority: str, cfg_for_tp, tp_options: list[int], *, max_model_len: int = 8192,
              min_concurrency: float = 1.0, limit: int = 10, token: str | None = None,
              include_gated: bool = False, max_checks: int = 40, lister=list_models, loader=None,
              progress=None) -> RecommendResult:
    probe = cfg_for_tp(1)
    budget = int(probe.gpu_total_bytes * probe.gpu_memory_utilization) - probe.other_process_bytes
    pool, skipped = candidates(task, token=token, include_gated=include_gated, budget_bytes=budget,
                               max_tp=max(tp_options), lister=lister)
    ordered = order_candidates(pool, priority, budget * max(tp_options))
    recs, checked = [], 0
    per_owner: dict[str, int] = {}
    for m in ordered:
        if len(recs) >= limit or checked >= max_checks:
            break
        owner = m.id.split("/", 1)[0]
        if per_owner.get(owner, 0) >= 3:          # keep the list varied
            continue
        checked += 1
        if progress:
            progress(checked, m.id)
        r = evaluate(m, cfg_for_tp, tp_options, max_model_len, token, loader)
        if r is None:
            skipped["could not be read or sized"] = skipped.get("could not be read or sized", 0) + 1
            continue
        if r.starts is not True:
            skipped["vLLM would not start (or not confirmed)"] = \
                skipped.get("vLLM would not start (or not confirmed)", 0) + 1
            continue
        if r.max_concurrency < min_concurrency:
            skipped["too few requests at once"] = skipped.get("too few requests at once", 0) + 1
            continue
        per_owner[owner] = per_owner.get(owner, 0) + 1
        recs.append(r)
    if priority != "balanced":
        recs.sort(key=lambda r: r.weights_bytes * r.tp, reverse=priority == "quality")
    return RecommendResult(recs, len(pool), checked, skipped)
