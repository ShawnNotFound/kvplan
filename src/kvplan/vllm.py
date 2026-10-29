"""Reproduce vLLM's startup memory accounting for one GPU (one tensor-parallel rank).

vLLM (Worker.determine_available_memory, vllm/v1/worker/gpu_worker.py):

    requested = ceil(total * gpu_memory_utilization)
    available_kv = requested - weights - other_non_kv - cudagraph_estimate

Memory used before vLLM's snapshot (CUDA context, NCCL and cuBLAS buffers, other
processes) is not charged to `requested`; it must fit in `total - requested`, or vLLM
refuses to start.

Weights and KV sizes are computed exactly from the model files. Activation peak,
sampler logits, CUDA-graph memory and the CUDA context are estimates (marked as such in
the output) until calibrated against real vLLM logs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .kv import KVCapacity, kv_capacity, kv_layout, resolve_kv_cache_dtype
from .model import ModelSpec
from .weights import WeightBreakdown

GiB = 2**30
DEFAULT_UTILIZATION = 0.92
# Estimates, to be calibrated against vLLM startup logs (see docs/accounting.md).
CUDA_CONTEXT_BYTES = int(0.5 * GiB)
NCCL_BYTES_PER_EXTRA_RANK = int(0.25 * GiB)
CUDAGRAPH_BYTES = int(0.5 * GiB)
NON_TORCH_BYTES = int(0.25 * GiB)
LOGITS_COPIES = 2


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

    def validate(self) -> None:
        if self.gpu_total_bytes <= 0:
            raise ValueError("GPU memory must be positive")
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
    activation_bytes: int
    logits_bytes: int
    cudagraph_bytes: int
    non_torch_bytes: int
    user_override: bool = False

    @property
    def total(self) -> int:
        return self.activation_bytes + self.logits_bytes + self.cudagraph_bytes + self.non_torch_bytes


def estimate_overheads(spec: ModelSpec, cfg: ServeConfig, batched_tokens: int, num_seqs: int) -> Overheads:
    if cfg.overhead_bytes is not None:
        return Overheads(cfg.overhead_bytes, 0, 0, 0, user_override=True)
    vocab = spec.vocab_size or 32000
    hidden = spec.hidden_size or spec.num_heads * spec.head_dim
    num_reqs = min(num_seqs, batched_tokens)
    # Sampler warm-up materialises fp32 logits for every request row.
    logits = num_reqs * vocab * 4 * LOGITS_COPIES
    # Rough transient activation peak for one forward pass over the batched tokens.
    activation = batched_tokens * hidden * spec.model_dtype_bytes * 12 // cfg.tp
    graphs = 0 if cfg.enforce_eager else CUDAGRAPH_BYTES
    return Overheads(activation, logits, graphs, NON_TORCH_BYTES)


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
    starts: bool
    checks: list[Check] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

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
    if spec.num_kv_heads < tp and not spec.is_mla:
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

    over = estimate_overheads(spec, cfg, batched, seqs)
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

    starts = not any(not c.ok and c.severity == "error" for c in checks)
    if memory_short:
        if resolve_kv_cache_dtype(spec, cfg.kv_cache_dtype) == "auto" and weights.per_rank_bytes < requested:
            suggestions.append("--kv-cache-dtype fp8 halves the KV cache")
        if spec.source == "tensor shapes" or spec.num_heads % (tp * 2) == 0:
            suggestions.append(f"--tensor-parallel-size {tp * 2} on {tp * 2} GPUs splits weights and KV further")
    elif starts and cap is not None and cap.max_concurrency < 1.5:
        suggestions.append("fewer than 2 full-length requests fit at once; consider a lower --max-model-len "
                           "or --kv-cache-dtype fp8 for more concurrency")

    return Plan(
        model=model_name, tp=tp, gpu_total_bytes=cfg.gpu_total_bytes, requested_bytes=requested, weights=weights,
        overheads=over, available_kv_bytes=available, max_model_len=max_model_len, max_num_seqs=seqs,
        max_num_batched_tokens=batched, kv_bytes_per_token=per_token,
        kv_cache_dtype=resolve_kv_cache_dtype(spec, cfg.kv_cache_dtype), block_size=block_size, kv=cap, starts=starts, checks=checks,
        suggestions=list(dict.fromkeys(suggestions)),
    )
