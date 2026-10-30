"""Weight memory per GPU under vLLM tensor parallelism.

Rules (vLLM main, see docs/accounting.md):
- Linear layers and expert weights are split across TP ranks (1/tp each).
- Embedding and lm_head are vocab-parallel: the vocab is padded to a multiple of 64,
  then split. A tied lm_head reuses the embedding.
- K and V projections are replicated when the model has fewer KV heads than TP ranks:
  each rank then holds one whole KV head.
- Norms, biases of 1-D shape, MoE routers and MLA's low-rank down-projections
  (q_a_proj, kv_a_proj_with_mqa) are replicated on every rank.
- Unquantized floating-point weights are held in 16-bit; quantized weights and their
  scales keep their checkpoint size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import ModelSpec, language_model_layers
from .weights_meta import Tensor

VOCAB_PADDING = 64
_FLOAT_DTYPES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}
_QUANT_AUX = re.compile(r"(scale|zero|qzeros|g_idx|absmax|quant_map|quant_state|_blocks|_scales)", re.I)
_EMBED = re.compile(r"(embed_tokens|wte|word_embeddings|tok_embeddings|token_embd)\.weight$")
_LM_HEAD = re.compile(r"(^|\.)(lm_head|output_layer)\.weight$|^output\.weight$")
_KV_PROJ = re.compile(r"\.(k_proj|v_proj|attn_k|attn_v)\.")
_REPLICATED = re.compile(r"(norm|\.router\.|e_score_correction_bias|q_a_proj|kv_a_proj_with_mqa|"
                         r"q_a_layernorm|kv_a_layernorm|\.gate\.weight$|ffn_gate_inp)")
_LAYER_INDEX = re.compile(r"(?:^|\.)(?:layers|blk|h)\.(\d+)\.")
_NOT_LOADED = re.compile(r"rotary_emb\.inv_freq$")

# Checkpoint parts vLLM drops at load time, per architecture: the prefixes its model class maps to
# None in hf_to_vllm_mapper (vLLM 0.32). Omni models ship a speech generator vLLM does not run.
NOT_LOADED_PREFIXES = {
    # vllm/model_executor/models/qwen2_5_omni_thinker.py Qwen2_5OmniThinkerForConditionalGeneration
    "Qwen2_5OmniModel": ("talker.", "token2wav."),
    "Qwen2_5OmniForConditionalGeneration": ("talker.", "token2wav."),
    # vllm/model_executor/models/qwen3_omni_moe_thinker.py Qwen3OmniMoeThinkerForConditionalGeneration
    "Qwen3OmniMoeForConditionalGeneration": ("talker.", "code2wav."),
    # vllm/model_executor/models/qwen3_asr.py Qwen3ASRForConditionalGeneration
    "Qwen3ASRForConditionalGeneration": ("talker.", "code2wav."),
    # vllm/model_executor/models/minicpmo.py MiniCPMO
    "MiniCPMO": ("tts",),
}
# Encoder towers vLLM builds from the plain transformers module (nn.Linear, not its tensor-parallel
# layers), so every GPU holds a full copy.
REPLICATED_PREFIXES = {
    # vllm/model_executor/models/qwen2_5_omni_thinker.py: audio_tower = transformers' Qwen2_5OmniAudioEncoder
    "Qwen2_5OmniModel": ("thinker.audio_tower.",),
    "Qwen2_5OmniForConditionalGeneration": ("thinker.audio_tower.",),
    # vllm/model_executor/models/qwen2_audio.py: audio_tower = transformers' Qwen2AudioEncoder
    "Qwen2AudioForConditionalGeneration": ("audio_tower.",),
}
# Top-level parts that look like a speech or audio generator; for architectures not listed above,
# kvplan counts them but warns that vLLM may not load them.
_GENERATOR_PART = re.compile(r"^(talker|token2wav|code2wav|tts|vocoder|audio_tokenizer|audio_invert_tower)[._]")


@dataclass
class WeightBreakdown:
    per_rank_bytes: int
    total_checkpoint_bytes: int
    embedding_bytes: int
    replicated_bytes: int
    skipped_bytes: int
    tied: bool
    notes: list[str]
    buffer_bytes: int = 0     # non-weight buffers vLLM allocates at load time (RoPE caches)


def _resident_bytes(t: Tensor, dtype_bytes: int) -> int:
    """Bytes vLLM keeps for this tensor before sharding."""
    if t.dtype in _FLOAT_DTYPES and not _QUANT_AUX.search(t.name):
        return t.numel * dtype_bytes
    return t.nbytes


def _is_extra_layer(name: str, num_layers: int) -> bool:
    """Layers past num_hidden_layers are multi-token-prediction (MTP) heads, e.g. DeepSeek-V3's
    layer 61. vLLM loads them only for speculative decoding."""
    m = _LAYER_INDEX.search(name)
    return bool(m) and m.start() == 0 and int(m.group(1)) >= num_layers


def weight_bytes_per_rank(spec: ModelSpec, tensors: list[Tensor], tp: int = 1) -> WeightBreakdown:
    """Weight bytes on the busiest tensor-parallel rank."""
    if tp < 1:
        raise ValueError("--tensor-parallel-size must be at least 1")
    dtype_bytes = spec.model_dtype_bytes
    names = {t.name for t in tensors}
    tied = not any(_LM_HEAD.search(n) for n in names)
    notes: list[str] = []

    per_rank = 0.0
    embedding = replicated = skipped = mtp = dropped = 0
    kv_replicas = spec.num_kv_heads < tp and spec.source != "tensor shapes"
    drop = NOT_LOADED_PREFIXES.get(spec.architecture or "", ())
    whole = REPLICATED_PREFIXES.get(spec.architecture or "", ())
    lm = language_model_layers(tensors)
    lm_prefix = lm[0] if lm else ""
    unknown_parts: dict[str, int] = {}
    for t in tensors:
        if _NOT_LOADED.search(t.name):
            skipped += t.nbytes
            continue
        if drop and t.name.startswith(drop):
            skipped += t.nbytes
            dropped += t.nbytes
            continue
        if not drop and (m := _GENERATOR_PART.match(t.name)):
            unknown_parts[m.group(1)] = unknown_parts.get(m.group(1), 0) + t.nbytes
        if t.name.startswith(lm_prefix) and _is_extra_layer(t.name[len(lm_prefix):] if lm_prefix else t.name,
                                                             spec.num_layers):
            skipped += t.nbytes
            mtp += t.nbytes
            continue
        size = _resident_bytes(t, dtype_bytes)
        if whole and t.name.startswith(whole):
            per_rank += size
            replicated += size
        elif _EMBED.search(t.name) or _LM_HEAD.search(t.name):
            rows = t.shape[0] if t.shape else 0
            if rows and t.dtype in _FLOAT_DTYPES:
                padded = -(-rows // VOCAB_PADDING) * VOCAB_PADDING
                part = -(-padded // tp)
                b = part * (t.numel // rows) * dtype_bytes
            else:
                b = -(-size // tp)
            per_rank += b
            embedding += b
            if tied and _EMBED.search(t.name):
                notes.append("lm_head is tied to the embedding (stored once)")
        elif len(t.shape) <= 1 or _REPLICATED.search(t.name):
            per_rank += size
            replicated += size
        elif _KV_PROJ.search(t.name) and kv_replicas:
            # Each rank holds one whole KV head out of num_kv_heads.
            per_rank += size / spec.num_kv_heads
        else:
            per_rank += size / tp
            if spec.is_mla and t.name.endswith("kv_b_proj.weight") and t.dtype not in _FLOAT_DTYPES:
                # vLLM keeps kv_b_proj and adds a dequantized 16-bit copy split into W_UK/W_UV
                # (vllm/model_executor/layers/attention/mla_attention.py split_kv_b_proj).
                per_rank += t.numel * dtype_bytes / tp
                notes.append("includes the 16-bit W_UK/W_UV copies vLLM makes of quantized MLA kv_b_proj")
    if dropped:
        notes.append(f"skipped {dropped / 2**30:.2f} GiB under {', '.join(drop)}, which vLLM's "
                     f"{spec.architecture} does not load")
    if unknown_parts:
        parts = ", ".join(f"{k} ({v / 2**30:.2f} GiB)" for k, v in sorted(unknown_parts.items()))
        notes.append(f"WARNING: counted {parts}, which look like a speech generator; kvplan does not know "
                     f"whether vLLM loads them for {spec.architecture or 'this architecture'}, so the weight "
                     "figure may be too high")
    if mtp:
        notes.append(f"skipped {mtp / 2**30:.2f} GiB of multi-token-prediction layers, which vLLM loads only "
                     "for speculative decoding")
    buffers = spec.rotary_cache_bytes
    if buffers:
        notes.append(f"includes {buffers / 2**30:.2f} GiB of RoPE cos/sin cache that vLLM allocates when it loads "
                     "the model")
    return WeightBreakdown(
        per_rank_bytes=int(round(per_rank)) + buffers,
        total_checkpoint_bytes=sum(t.nbytes for t in tensors),
        embedding_bytes=embedding,
        replicated_bytes=replicated,
        skipped_bytes=skipped,
        tied=tied,
        notes=sorted(set(notes)),
        buffer_bytes=buffers,
    )
