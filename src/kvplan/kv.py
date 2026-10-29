"""KV-cache sizes and block planning, following vLLM's v1 KV cache manager.

References (vLLM main @ 434eadc):
- per-layer page size: vllm/v1/kv_cache_interface.py `AttentionSpec.page_size_bytes`
- per-request usage: `FullAttentionSpec.max_memory_usage_bytes`,
  `SlidingWindowSpec.max_admission_blocks_per_request`
- grouping of hybrid models: vllm/v1/core/kv_cache_planning.py
  `_get_kv_cache_groups_uniform_page_size`
- block count and capacity: `get_kv_cache_config_from_groups`,
  vllm/v1/core/kv_cache_utils.py `get_max_concurrency_for_kv_cache_config`
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import MLA, SLIDING, ModelSpec

KV_DTYPE_BYTES = {"fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def resolve_kv_cache_dtype(spec: ModelSpec, kv_cache_dtype: str) -> str:
    """`auto` follows the checkpoint's quantization config when it sets one, as vLLM does."""
    if kv_cache_dtype == "auto" and spec.default_kv_cache_dtype:
        return spec.default_kv_cache_dtype
    return kv_cache_dtype


def kv_dtype_bytes(spec: ModelSpec, kv_cache_dtype: str) -> int:
    kv_cache_dtype = resolve_kv_cache_dtype(spec, kv_cache_dtype)
    if kv_cache_dtype == "auto":
        return spec.model_dtype_bytes
    if kv_cache_dtype in KV_DTYPE_BYTES:
        return KV_DTYPE_BYTES[kv_cache_dtype]
    raise ValueError(f"kv_cache_dtype '{kv_cache_dtype}' is not supported (use auto, fp8, fp8_e4m3 or fp8_e5m2)")


def kv_heads_per_rank(spec: ModelSpec, tp: int) -> int:
    """vLLM `ModelConfig.get_num_kv_heads`: heads are replicated when there are fewer than TP ranks."""
    if spec.is_mla:
        return 1
    return max(1, spec.num_kv_heads // tp)


def bytes_per_token(spec: ModelSpec, layer_type: str, tp: int, dtype_bytes: int) -> int:
    """KV bytes one token occupies in one layer on one TP rank."""
    if layer_type == MLA:
        # The latent cache (kv_lora_rank + rope dims) is not split across TP ranks.
        return (spec.kv_lora_rank + spec.qk_rope_head_dim) * dtype_bytes
    return kv_heads_per_rank(spec, tp) * (spec.head_dim + spec.v_head_dim) * dtype_bytes


@dataclass(frozen=True)
class LayerSpec:
    kind: str
    page_bytes: int
    window: int | None

    def blocks_per_request(self, max_model_len: int, block_size: int, in_flight_tokens: int) -> int:
        if self.kind == SLIDING:
            tokens = min(self.window - 1 + in_flight_tokens, max_model_len)
            return cdiv(tokens, block_size) + 1
        return cdiv(max_model_len, block_size)


@dataclass
class KVLayout:
    """How vLLM lays out the KV cache for one TP rank."""

    layers: list[LayerSpec]
    groups: list[list[int]]          # layer indices per KV cache group
    bytes_per_block: int
    block_size: int
    padded_layers: int = 0

    def bytes_needed_for_one_request(self, max_model_len: int, in_flight_tokens: int) -> int:
        """Memory one full-length request needs, as vLLM's start check counts it: whole pool
        blocks per group, which includes padding layers in hybrid models
        (kv_cache_planning.py `_max_memory_usage_bytes_from_groups`)."""
        return self.bytes_per_block * self.blocks_per_request(max_model_len, in_flight_tokens)

    def blocks_per_request(self, max_model_len: int, in_flight_tokens: int) -> int:
        """Blocks one full-length request draws from the shared pool (one per group per block step)."""
        return sum(self.layers[g[0]].blocks_per_request(max_model_len, self.block_size, in_flight_tokens)
                   for g in self.groups)


def kv_layout(spec: ModelSpec, tp: int, kv_cache_dtype: str = "auto", block_size: int = 16) -> KVLayout:
    b = kv_dtype_bytes(spec, kv_cache_dtype)
    layers = []
    for kind in spec.layer_types:
        per_token = bytes_per_token(spec, kind, tp, b)
        layers.append(LayerSpec(kind, block_size * per_token, spec.sliding_window if kind == SLIDING else None))

    buckets: dict[tuple, list[int]] = {}
    for i, layer in enumerate(layers):
        buckets.setdefault((layer.kind, layer.window, layer.page_bytes), []).append(i)

    if len(buckets) == 1:
        # Uniform model: one group, every block holds a page for every layer.
        return KVLayout(layers, [list(range(len(layers)))], sum(layer.page_bytes for layer in layers), block_size)

    page_sizes = {layer.page_bytes for layer in layers}
    if len(page_sizes) != 1:
        raise ValueError("hybrid model whose layer types have different page sizes; vLLM pads or uses another "
                         "planner here, which kvplan does not model yet")
    # vLLM `_get_kv_cache_groups_uniform_page_size`.
    sizes = [len(v) for v in buckets.values()]
    group_size = min(sizes)
    if max(sizes) < min(sizes) * 1.5:
        group_size = max(sizes)
    groups, padded = [], 0
    for idxs in buckets.values():
        n_groups = cdiv(len(idxs), group_size)
        padded += n_groups * group_size - len(idxs)
        for g in range(n_groups):
            groups.append(idxs[g::n_groups])
    return KVLayout(layers, groups, group_size * page_sizes.pop(), block_size, padded)


@dataclass
class KVCapacity:
    num_blocks: int
    kv_tokens: int               # vLLM's "GPU KV cache size: N tokens"
    max_concurrency: float       # vLLM's "Maximum concurrency for L tokens per request"
    bytes_needed_one_request: int
    fits_one_request: bool
    max_model_len_that_fits: int


def kv_capacity(layout: KVLayout, available_bytes: int, max_model_len: int, in_flight_tokens: int) -> KVCapacity:
    num_blocks = max(0, available_bytes // layout.bytes_per_block)
    per_req = layout.blocks_per_request(max_model_len, in_flight_tokens)
    concurrency = num_blocks / per_req if per_req else 0.0
    needed = layout.bytes_needed_for_one_request(max_model_len, in_flight_tokens)
    # vLLM checks against available memory minus the null block BlockPool keeps back.
    check_memory = available_bytes - layout.bytes_per_block
    fits = available_bytes > 0 and needed <= check_memory
    return KVCapacity(
        num_blocks=num_blocks,
        kv_tokens=int(concurrency * max_model_len),
        max_concurrency=concurrency,
        bytes_needed_one_request=needed,
        fits_one_request=fits,
        max_model_len_that_fits=max_len_that_fits(layout, check_memory, max_model_len, in_flight_tokens),
    )


def max_len_that_fits(layout: KVLayout, memory: int, upper: int, in_flight_tokens: int) -> int:
    """vLLM `estimate_max_model_len`: binary search for the longest request that fits."""
    if memory <= 0 or layout.bytes_needed_for_one_request(1, in_flight_tokens) > memory:
        return 0
    lo, hi = 1, upper
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if layout.bytes_needed_for_one_request(mid, in_flight_tokens) <= memory:
            lo = mid
        else:
            hi = mid - 1
    return lo
