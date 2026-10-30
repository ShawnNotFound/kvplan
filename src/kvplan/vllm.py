"""Reproduce vLLM's startup memory accounting for one GPU (one tensor-parallel rank).

vLLM (Worker.determine_available_memory, vllm/v1/worker/gpu_worker.py):

    requested = ceil(total * gpu_memory_utilization)
    available_kv = requested - consumed - profiled_peak

`consumed` is the weights plus non-torch memory; `profiled_peak` is the peak of the
profiling forward pass and sampler warm-up, which with CUDA graphs on also holds the
torch.compile buffers and the graph pool (vLLM 0.32 logs: available KV equals
requested - consumed - peak exactly).

Memory used before vLLM's snapshot (CUDA context, NCCL and cuBLAS buffers, other
processes) is not charged to `requested`; it must fit in `total - requested`, or vLLM
refuses to start.

Weights and KV sizes are computed exactly from the model files. The activation peak,
torch.compile buffers, CUDA-graph memory and non-torch memory are fitted to vLLM 0.32
startup logs on RTX A6000 and RTX A5000 GPUs for dense models (docs/validation.md); for
MoE models and models whose MLP width is unknown they fall back to a rougher estimate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import mm
from .kv import KVCapacity, kv_capacity, kv_layout, resolve_kv_cache_dtype
from .model import ModelSpec
from .vllm_architectures import VLLM_ARCHITECTURES
from .weights import WeightBreakdown

GiB = 2**30
DEFAULT_UTILIZATION = 0.92
# Fitted to vLLM 0.32 startup logs (docs/validation.md). The CUDA context measured 0.2-0.46 GiB;
# 0.5 keeps the start check on the safe side.
CUDA_CONTEXT_BYTES = int(0.5 * GiB)
NCCL_BYTES_PER_EXTRA_RANK = int(0.25 * GiB)
# With CUDA graphs on, vLLM's profiled peak (which already contains the graph pool, measured
# 0.20-0.24 GiB) grows by a fixed part plus a part that scales with batched tokens x hidden.
CUDAGRAPH_BYTES = int(0.25 * GiB)
COMPILE_ACTIVATION_FACTOR = 7
NON_TORCH_BYTES = int(0.25 * GiB)       # measured 0.16-0.39 GiB beyond the weight files
# Multimodal models leave more non-torch memory after profiling: 0.39-0.57 GiB (median 0.45) on the
# A6000 TP=1 fit runs in docs/validation.md. The cause was not traced to a vLLM allocation, so this
# is a fitted constant, applied to every model with a vision or audio encoder.
MM_NON_TORCH_EXTRA_BYTES = int(0.20 * GiB)
MLP_PEAK_FACTOR = 2                     # eager peak = 2 x the gate/up projection output


@dataclass
class ServeConfig:
    gpu_total_bytes: int
    tp: int = 1
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float = DEFAULT_UTILIZATION
    kv_cache_dtype: str = "auto"
    block_size: int | None = None       # None: vLLM's backend default (16, or 64/128 for MLA)
    enforce_eager: bool = False
    async_scheduling: bool = True       # vLLM's default; keeps two batches in flight
    other_process_bytes: int = 0
    overhead_bytes: int | None = None   # replaces the activation/graph/non-torch estimate
    gpu_name: str = ""
    mm_limits: dict[str, int] = field(default_factory=dict)   # --limit-mm-per-prompt
    mm_max_pixels: int | None = None    # --mm-processor-kwargs max_pixels
    mm_min_pixels: int | None = None
    skip_mm_profiling: bool = False

    def validate(self) -> None:
        if self.gpu_total_bytes <= 0:
            raise ValueError("GPU memory must be positive (check --gpu-mem-gib)")
        if self.other_process_bytes < 0:
            raise ValueError("--other-gpu-mem-gib cannot be negative")
        if self.other_process_bytes >= self.gpu_total_bytes:
            raise ValueError("--other-gpu-mem-gib is at least the GPU's total memory; check the value")
        if self.overhead_bytes is not None and self.overhead_bytes < 0:
            raise ValueError("--overhead-gib cannot be negative")
        if self.tp < 1:
            raise ValueError("--tensor-parallel-size must be at least 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("--gpu-memory-utilization must be in (0, 1]")
        for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "block_size"):
            v = getattr(self, name)
            if v is not None and v < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be at least 1")


def default_block_size(spec: ModelSpec, gpu_name: str) -> int:
    """vLLM's attention backends choose the block size: 16 for most, 64 for FlashMLA (Hopper),
    128 for CUTLASS MLA (Blackwell) (get_preferred_block_size in vllm/v1/attention/backends)."""
    if not spec.is_mla:
        return 16
    return 128 if any(t in gpu_name.upper() for t in ("B200", "B300", "GB200", "GB300")) else 64


def batch_defaults(total_bytes: int, gpu_name: str = "") -> tuple[int, int]:
    """vLLM's OpenAI-server defaults (max_num_batched_tokens, max_num_seqs) by GPU
    (EngineArgs.get_batch_defaults): A100 and GB10 keep the small defaults despite their memory."""
    small_class = any(tag in gpu_name.upper() for tag in ("A100", "GB10"))
    if total_bytes >= 160 * GiB:
        return 16384, 1024
    if total_bytes >= 70 * GiB and not small_class:
        return 8192, 1024
    return 2048, 256


@dataclass
class Overheads:
    activation_bytes: int     # peak of one profiling forward pass plus the sampler warm-up
    compile_bytes: int        # extra torch.compile buffers (0 with enforce_eager)
    cudagraph_bytes: int
    non_torch_bytes: int
    user_override: bool = False
    calibrated: bool = True   # False when the model falls outside what was measured (MoE, unknown MLP)
    encoder_bytes: int = 0    # extra peak from profiling the multimodal encoder (mm.py)
    encoder_note: str = ""
    encoder_approximate: bool = False

    @property
    def total(self) -> int:
        return (self.activation_bytes + self.compile_bytes + self.cudagraph_bytes + self.non_torch_bytes
                + self.encoder_bytes)


def encoder_profile_bytes(spec: ModelSpec, cfg: ServeConfig, batched_tokens: int, num_seqs: int,
                          max_model_len: int, text_activation: int) -> tuple[int, str, bool]:
    """Extra profiled peak from vLLM's multimodal startup profiling (mm.py), a note saying how it was
    derived, and whether it is approximate.

    The encoder pass runs first and its outputs stay in the encoder cache during the text pass, so
    the profiled peak is max(encoder peak, encoder outputs + text peak).
    """
    enc = spec.encoder
    if enc is None or cfg.skip_mm_profiling:
        return 0, "", False
    prof = mm.profile(enc, spec.hidden_size or 0, cfg.tp, batched_tokens, num_seqs, max_model_len,
                      cfg.mm_limits, cfg.mm_max_pixels, cfg.mm_min_pixels)
    if prof is None:
        return 0, "", False
    peak = max(prof.encoder_peak_bytes, prof.output_bytes + text_activation)
    extra = peak - text_activation
    what = (f"{prof.items} {prof.modality} item(s) of {prof.item_outputs:,} tokens"
            + (f" (budgeted as {prof.item_tokens:,})" if prof.item_outputs != prof.item_tokens else ""))
    note = (f"includes {extra / GiB:.2f} GiB for vLLM's startup profiling of the {enc.kind} encoder on {what}"
            + ("; " + "; ".join(prof.notes) if prof.notes else "")
            + ". --limit-mm-per-prompt, --mm-processor-kwargs and --skip-mm-profiling change this")
    if prof.approximate:
        note = ("APPROXIMATE: " + note + f". kvplan's encoder memory rule for this model ({enc.family}) is "
                "approximate (docs/accounting.md, section 8), so the KV figure may be off by a few percent")
    return extra, note, prof.approximate


def estimate_overheads(spec: ModelSpec, cfg: ServeConfig, batched_tokens: int, num_seqs: int,
                       max_model_len: int = 0) -> Overheads:
    if cfg.overhead_bytes is not None:
        return Overheads(cfg.overhead_bytes, 0, 0, 0, user_override=True)
    vocab = spec.vocab_size or 32000
    hidden = spec.hidden_size or spec.num_heads * spec.head_dim
    dtype = spec.model_dtype_bytes
    num_reqs = min(num_seqs, batched_tokens)
    # The sampler warm-up materialises fp32 logits for every request row.
    logits = num_reqs * vocab * 4
    if spec.intermediate_size:
        # The profiling pass peaks in the MLP (gate and up projections over every batched token),
        # unless the sampler's logits are larger (big vocabularies, small batches).
        mlp = batched_tokens * spec.intermediate_size * 2 * dtype * MLP_PEAK_FACTOR // cfg.tp
        activation, calibrated = max(mlp, logits), True
    else:
        # MoE or unknown MLP width: the uncalibrated v0.1 estimate.
        activation, calibrated = batched_tokens * hidden * dtype * 12 // cfg.tp + 2 * logits, False
    compiled = 0 if cfg.enforce_eager else batched_tokens * hidden * dtype * COMPILE_ACTIVATION_FACTOR // cfg.tp
    graphs = 0 if cfg.enforce_eager else CUDAGRAPH_BYTES
    encoder, note, approx = encoder_profile_bytes(spec, cfg, batched_tokens, num_seqs, max_model_len, activation)
    non_torch = NON_TORCH_BYTES + (MM_NON_TORCH_EXTRA_BYTES if spec.encoder is not None else 0)
    return Overheads(activation, compiled, graphs, non_torch, calibrated=calibrated,
                     encoder_bytes=encoder, encoder_note=note, encoder_approximate=approx)


VERDICT_STARTS = "vLLM should start"
VERDICT_FAILS = "vLLM will not start"
VERDICT_UNCONFIRMED = "start not confirmed"
LARGEST_MEASURED_ENCODER_PEAK = int(5.95 * GiB)   # MiniCPM-o 2.6, docs/validation.md


@dataclass
class Check:
    ok: bool
    severity: str        # "error" | "warning" | "info"
    message: str


@dataclass
class Plan:
    model: str
    tp: int
    gpu_total_bytes: int
    requested_bytes: int
    weights: WeightBreakdown
    overheads: Overheads
    available_kv_bytes: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_bytes_per_token: int
    kv_cache_dtype: str
    block_size: int
    kv: KVCapacity | None
    starts: bool | None       # None: no error found, but an unsized term means the start is not confirmed
    checks: list[Check] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    verdict: str = VERDICT_STARTS
    unconfirmed: list[str] = field(default_factory=list)   # the terms kvplan could not size
    start_margin_bytes: int | None = None   # memory to spare at the start check under kvplan's estimates

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overheads"]["total"] = self.overheads.total
        return d


def _gib(b: int | float) -> str:
    return f"{b / GiB:.2f} GiB"


def plan(spec: ModelSpec, weights: WeightBreakdown, cfg: ServeConfig, model_name: str = "") -> Plan:
    cfg.validate()
    checks: list[Check] = []
    suggestions: list[str] = []
    tp = cfg.tp

    if spec.source != "tensor shapes":
        if spec.num_heads % tp:
            checks.append(Check(False, "error", f"tensor_parallel_size={tp} does not divide the "
                                                f"{spec.num_heads} attention heads; vLLM will refuse to start"))
        if not spec.is_mla and spec.num_kv_heads % tp and tp % spec.num_kv_heads:
            checks.append(Check(False, "error", f"{spec.num_kv_heads} KV heads cannot be split or replicated "
                                                f"evenly over {tp} GPUs"))
    unconfirmed: list[str] = []
    measure_once = ("measure once: start vLLM, then pass its logged peak activation plus consumed memory minus "
                    "weights as --overhead-gib")
    if spec.source == "tensor shapes":
        unconfirmed.append("the attention geometry (derived from weight shapes)")
        checks.append(Check(True, "warning", "UNCERTAIN: the attention geometry was derived from weight shapes, "
                                             "not from config.json, so the KV figures may be wrong"))
        suggestions.append("add the missing attention fields to config.json, or " + measure_once)
    if spec.architecture and spec.architecture not in VLLM_ARCHITECTURES:
        unconfirmed.append(f"the {spec.architecture} architecture (no native vLLM implementation)")
        checks.append(Check(True, "warning", f"UNCERTAIN: vLLM 0.32 has no native implementation of "
                                             f"{spec.architecture}; it may refuse the model or run it through "
                                             "the Transformers backend, whose memory use kvplan has not measured"))
        suggestions.append(measure_once)
    encoder_unsized = (spec.encoder is not None and spec.encoder.family == "generic" and not cfg.skip_mm_profiling
                       and cfg.overhead_bytes is None)
    if encoder_unsized:
        unconfirmed.append(f"the {spec.encoder.kind} encoder's startup profiling")
        checks.append(Check(True, "warning", f"UNCERTAIN: kvplan cannot size vLLM's startup profiling of this "
                                             f"model's {spec.encoder.kind} encoder; in measurements it ranged from "
                                             "0.1 to 5.9 GiB (MiniCPM-o 2.6), so the KV figure and the start "
                                             "check may be wrong"))
        suggestions += ["--skip-mm-profiling, if you run vLLM with it: no encoder memory is reserved at startup, "
                        "so large inputs can run out of memory later",
                        "--limit-mm-per-prompt (e.g. '{\"video\": 0}') or --mm-processor-kwargs "
                        "'{\"max_pixels\": ...}' shrink what vLLM profiles",
                        measure_once]
    if spec.num_kv_heads < tp and not spec.is_mla and spec.source != "tensor shapes":
        checks.append(Check(True, "info", f"only {spec.num_kv_heads} KV heads for {tp} GPUs: each GPU keeps a "
                                          "replicated head, so KV memory does not shrink further with more GPUs"))

    native_len = spec.max_position_embeddings
    max_model_len = cfg.max_model_len or spec.default_max_model_len or native_len
    if not max_model_len:
        raise ValueError("the model does not state its context length; pass --max-model-len")
    if native_len and max_model_len > native_len:
        checks.append(Check(False, "warning", f"max_model_len {max_model_len} is above the {native_len} tokens "
                                              "vLLM derives from the model config; vLLM rejects this unless "
                                              "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is set"))

    d_tokens, d_seqs = batch_defaults(cfg.gpu_total_bytes, cfg.gpu_name)
    batched = cfg.max_num_batched_tokens or d_tokens
    seqs = cfg.max_num_seqs or d_seqs
    requested = math.ceil(cfg.gpu_total_bytes * cfg.gpu_memory_utilization)

    outside = CUDA_CONTEXT_BYTES + NCCL_BYTES_PER_EXTRA_RANK * (tp > 1) + cfg.other_process_bytes
    free_at_start = cfg.gpu_total_bytes - outside
    if free_at_start < requested:
        checks.append(Check(False, "error",
                            f"gpu_memory_utilization={cfg.gpu_memory_utilization} asks for {_gib(requested)}, but "
                            f"only about {_gib(free_at_start)} is free after the CUDA context"
                            + (", NCCL" if tp > 1 else "")
                            + (" and other processes" if cfg.other_process_bytes else "")
                            + ". vLLM stops with 'Free memory on device ... is less than desired GPU memory "
                              "utilization'"))
        suggestions.append(f"lower --gpu-memory-utilization to "
                           f"{math.floor(free_at_start / cfg.gpu_total_bytes * 100) / 100:.2f}")

    over = estimate_overheads(spec, cfg, batched, seqs, max_model_len)
    memory_short = False
    if weights.per_rank_bytes > requested:
        memory_short = True
        checks.append(Check(False, "error", f"weights need {_gib(weights.per_rank_bytes)} per GPU, more than the "
                                            f"{_gib(requested)} vLLM may use (gpu_memory_utilization="
                                            f"{cfg.gpu_memory_utilization}); loading fails with out-of-memory"))
    available = requested - weights.per_rank_bytes - over.total

    block_size = cfg.block_size or default_block_size(spec, cfg.gpu_name)
    layout = kv_layout(spec, tp, cfg.kv_cache_dtype, block_size)
    per_token = sum(layer.page_bytes for layer in layout.layers) // block_size
    # vllm/config/vllm.py max_in_flight_tokens: async scheduling (default) keeps two batches in flight.
    in_flight = batched * (2 if cfg.async_scheduling else 1)
    cap = None
    if available <= 0:
        memory_short = True
        checks.append(Check(False, "error", f"no memory left for the KV cache: {_gib(requested)} budget - "
                                            f"{_gib(weights.per_rank_bytes)} weights - {_gib(over.total)} "
                                            f"runtime = {_gib(available)}. vLLM: 'No available memory for "
                                            "the cache blocks'"))
    else:
        cap = kv_capacity(layout, available, max_model_len, in_flight)
        if not cap.fits_one_request:
            memory_short = True
            checks.append(Check(False, "error",
                                f"one request at max_model_len={max_model_len} needs "
                                f"{_gib(cap.bytes_needed_one_request)} of KV cache, but only {_gib(available)} is "
                                f"available; vLLM's estimate of the longest length that fits is "
                                f"{cap.max_model_len_that_fits}"))
            if cap.max_model_len_that_fits > 0:
                suggestions.append(f"--max-model-len {cap.max_model_len_that_fits} (or lower)")

    no_errors = not any(not c.ok and c.severity == "error" for c in checks)
    margin = None
    if cap is not None:
        # The start check needs one full-length request plus vLLM's null block.
        margin = available - layout.bytes_per_block - cap.bytes_needed_one_request
    if not no_errors:
        verdict, starts = VERDICT_FAILS, False
    elif unconfirmed:
        verdict, starts = VERDICT_UNCONFIRMED, None
    else:
        verdict, starts = VERDICT_STARTS, True
    if margin is not None and encoder_unsized:
        assumed = over.activation_bytes + over.encoder_bytes
        if assumed + margin < LARGEST_MEASURED_ENCODER_PEAK:
            checks.append(Check(True, "warning",
                                f"vLLM starts only if its startup profiling peak (encoder pass or text pass) stays "
                                f"below {_gib(assumed + margin)}; kvplan assumed {_gib(assumed)}, and the largest "
                                f"measured with an encoder kvplan cannot size was "
                                f"{_gib(LARGEST_MEASURED_ENCODER_PEAK)} (MiniCPM-o 2.6)"))
    if memory_short:
        if resolve_kv_cache_dtype(spec, cfg.kv_cache_dtype) == "auto" and weights.per_rank_bytes < requested:
            suggestions.append("--kv-cache-dtype fp8 halves the KV cache")
        if spec.source == "tensor shapes" or spec.num_heads % (tp * 2) == 0:
            suggestions.append(f"--tensor-parallel-size {tp * 2} on {tp * 2} GPUs splits weights and KV further")
    elif no_errors and cap is not None and cap.max_concurrency < 1.5:
        suggestions.append("fewer than 2 full-length requests fit at once; consider a lower --max-model-len "
                           "or --kv-cache-dtype fp8 for more concurrency")

    return Plan(
        model=model_name, tp=tp, gpu_total_bytes=cfg.gpu_total_bytes, requested_bytes=requested, weights=weights,
        overheads=over, available_kv_bytes=available, max_model_len=max_model_len, max_num_seqs=seqs,
        max_num_batched_tokens=batched, kv_bytes_per_token=per_token,
        kv_cache_dtype=resolve_kv_cache_dtype(spec, cfg.kv_cache_dtype), block_size=block_size, kv=cap, starts=starts, checks=checks,
        suggestions=list(dict.fromkeys(suggestions)), verdict=verdict, unconfirmed=unconfirmed,
        start_margin_bytes=margin,
    )
