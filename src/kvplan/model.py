"""A model's attention geometry, read from config.json, GGUF metadata or tensor shapes.

Per-layer attention types follow what vLLM does: it uses `config.layer_types` when present
(transformers fills it with model-specific defaults for some model types), and otherwise
applies `sliding_window` to every layer. Both cases are reproduced in `_default_layer_types`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FULL, SLIDING, MLA = "full", "sliding", "mla"

# Sub-configs transformers' PretrainedConfig.get_text_config() looks for (any one may be present).
_TEXT_CONFIG_NAMES = ("text_encoder", "decoder", "generator", "text_config")
# Other names vLLM maps onto text_config: llm_config (InternVL, Bagel; _CONFIG_ATTRS_MAPPING in
# vllm/transformers_utils/config.py) and language_config (DeepSeek-VL2's config class).
_TEXT_CONFIG_ALIASES = ("llm_config", "language_config")
# Defaults of the transformers config classes for fields a nested text config may leave out
# (Llava's text_config often names only model_type llama; transformers fills in LlamaConfig's defaults).
_CONFIG_DEFAULTS = {
    "llama": {"num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096, "intermediate_size": 11008,
              "max_position_embeddings": 2048, "vocab_size": 32000},
}
# Sub-configs that describe multimodal encoders vLLM runs at startup to profile memory.
_ENCODER_CONFIG_KEYS = ("vision_config", "audio_config")

# Config keys that mean the model has Mamba / linear-attention layers (not supported in v0.1).
_HYBRID_KEYS = ("layers_block_type", "hybrid_override_pattern", "attn_type_list", "full_attention_interval",
                "linear_num_value_heads", "ssm_cfg", "mamba_d_state", "mamba_expand", "linear_attn_config")
_HYBRID_TYPES = ("mamba", "jamba", "zamba", "falcon_h1", "nemotron_h", "qwen3_next", "granitemoehybrid",
                 "lfm2", "plamo2", "bamba", "rwkv", "recurrent_gemma", "qwen3_5")

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float": 4, "half": 2}


class UnsupportedModel(Exception):
    """kvplan cannot compute this model's memory exactly."""


class MissingConfigFields(UnsupportedModel):
    """config.json lacks attention fields; the weight shapes may still describe the model."""


@dataclass
class EncoderSpec:
    """A multimodal encoder (vision or audio tower) that vLLM runs during its startup profiling."""
    kind: str                       # "vision" or "audio"
    hidden_size: int | None
    intermediate_size: int | None
    spatial_merge_size: int = 1     # encoder positions per output token = merge^2
    max_item_tokens: int | None = None  # output tokens of the largest item, when it can be derived


@dataclass
class ModelSpec:
    model_type: str
    num_layers: int
    hidden_size: int | None
    num_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    vocab_size: int | None
    max_position_embeddings: int | None     # the largest max_model_len vLLM accepts
    layer_types: list[str]
    sliding_window: int | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"
    quantization: dict | None = None
    num_experts: int | None = None
    default_max_model_len: int | None = None  # what vLLM uses when --max-model-len is not given
    default_kv_cache_dtype: str | None = None  # what --kv-cache-dtype auto resolves to, if not the model dtype
    source: str = "config"
    notes: list[str] = field(default_factory=list)
    intermediate_size: int | None = None   # dense MLP width; None for MoE or when unknown
    architecture: str | None = None        # config.json "architectures"[0]
    encoder: EncoderSpec | None = None      # multimodal encoder profiled at startup

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def model_dtype_bytes(self) -> int:
        """Bytes per element vLLM uses for unquantized weights and an `auto` KV cache.

        vLLM's dtype="auto" keeps fp16/bf16 and runs fp32 checkpoints in 16-bit.
        """
        return 2


def text_config(top: dict) -> tuple[dict, dict, str]:
    """(language-model config, the config that holds it, its path), resolved as vLLM 0.32 does.

    vLLM's get_hf_text_config (vllm/transformers_utils/config.py) calls transformers'
    PretrainedConfig.get_text_config(): it returns the one sub-config named text_encoder, decoder,
    generator or text_config, the config itself when there is none, and fails when there are
    several. Qwen2.5-Omni and Qwen3-Omni override it to return thinker_config's text config.
    """
    container, path = top, ""
    thinker = top.get("thinker_config")
    if isinstance(thinker, dict):
        container, path = thinker, "thinker_config."
    found = [k for k in _TEXT_CONFIG_NAMES if isinstance(container.get(k), dict)]
    if not found:
        found = [k for k in _TEXT_CONFIG_ALIASES if isinstance(container.get(k), dict)][:1]
    if len(found) > 1:
        raise UnsupportedModel(f"config.json has several language-model sub-configs ({', '.join(found)}); "
                               "vLLM cannot pick one either")
    if not found:
        return container, container, path.rstrip(".") or "config.json"
    sub = container[found[0]]
    defaults = _CONFIG_DEFAULTS.get(sub.get("model_type"), {})
    return {**defaults, **sub}, container, path + found[0]


def _int_or_none(v) -> int | None:
    """A positive integer, or None for missing, per-layer (list) or nonsense values."""
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


def _first(cfg: dict, *keys, default=None):
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def _default_layer_types(model_type: str, cfg: dict, n: int) -> tuple[list[str], int | None]:
    """Per-layer attention types when config.json has no `layer_types`, and the window used.

    For model types whose transformers config class fills in `layer_types`, that default is
    reproduced. For the others (Mistral, Mixtral, Phi-3, Starcoder2, ...), vLLM applies the
    config's `sliding_window` to every layer (vllm/engine/arg_utils.py sets
    CacheConfig.sliding_window when `layer_types` is absent; Attention falls back to it).
    """
    sw = cfg.get("sliding_window")
    full = [FULL] * n
    if model_type in ("qwen2", "qwen3", "qwen3_moe"):
        if not cfg.get("use_sliding_window", False) or sw is None:
            return full, None
        if model_type == "qwen3_moe":
            return [SLIDING] * n, sw
        start = cfg.get("max_window_layers", 28)
        types = [SLIDING if i >= start else FULL for i in range(n)]
        if FULL in types and SLIDING in types:
            raise UnsupportedModel(f"{model_type}: vLLM refuses to start this model with sliding window on only "
                                   f"some layers (max_window_layers={start} < {n} layers)")
        return types, (sw if SLIDING in types else None)
    if model_type == "qwen2_moe":
        # vLLM's qwen2_moe implementation runs every layer with full attention.
        return full, None
    if model_type in ("gemma2", "gpt_oss"):
        return [SLIDING if (i + 1) % 2 else FULL for i in range(n)], sw
    if model_type in ("gemma3", "gemma3_text"):
        p = cfg.get("sliding_window_pattern", 6)
        return [SLIDING if (i + 1) % p else FULL for i in range(n)], sw
    if model_type == "cohere2":
        p = cfg.get("sliding_window_pattern", 4)
        return [SLIDING if (i + 1) % p else FULL for i in range(n)], sw
    if model_type == "olmo3":
        return [SLIDING if (i + 1) % 4 else FULL for i in range(n)], sw
    if model_type == "smollm3":
        if cfg.get("use_sliding_window") and sw is not None:
            raise UnsupportedModel("smollm3: sliding window together with NoPE layers is not supported yet")
        return full, None
    return ([SLIDING] * n, sw) if sw is not None else (full, None)


# Model types vLLM runs with MLA (vllm/transformers_utils/model_arch_config_convertor.py is_deepseek_mla).
_MLA_TYPES = {"deepseek_v2", "deepseek_v3", "kimi_k2", "kimi_vl", "glm4_moe_lite", "longcat_flash",
              "pangu_ultra_moe", "dots3_note", "axk1"}

# Context-length keys vLLM considers; it takes the smallest present
# (model_arch_config_convertor.py derive_max_model_len_and_key).
_LENGTH_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "model_max_length",
                "max_target_positions", "max_sequence_length", "max_seq_length", "seq_len")
# RoPE types whose max_position_embeddings is already scaled (vllm/config/model.py _get_and_verify_max_len).
_ROPE_ALREADY_SCALED = {"su", "longrope", "llama3", "yarn", "deepseek_yarn", "deepseek_llama_scaling"}


def _rope_parameter_sets(cfg: dict) -> list[dict]:
    rp = cfg.get("rope_parameters") or cfg.get("rope_scaling")
    if not isinstance(rp, dict) or not rp:
        return []
    if all(isinstance(v, dict) for v in rp.values()):
        return [v for v in rp.values() if v]
    return [rp]


def context_lengths(cfg: dict, model_type: str, tokenizer_config: dict | None = None) -> tuple[int | None, int | None]:
    """(maximum vLLM allows, default vLLM uses) for max_model_len, following _get_and_verify_max_len."""
    found = [cfg[k] for k in _LENGTH_KEYS if isinstance(cfg.get(k), (int, float))]
    if not found:
        return None, None
    derived = min(found)
    tok_max = (tokenizer_config or {}).get("model_max_length")
    if isinstance(tok_max, (int, float)) and tok_max > 0:
        derived = min(derived, tok_max)
    rope = _rope_parameter_sets(cfg)
    if rope and "gemma3" not in model_type:
        factor = 1.0
        for rp in rope:
            rtype = rp.get("rope_type") or rp.get("type") or "default"
            if rtype not in _ROPE_ALREADY_SCALED:
                factor = rp.get("factor", factor)
        derived *= factor if factor is not None else 1.0
    derived = int(derived)
    default = derived
    if any((rp.get("rope_type") or rp.get("type")) == "longrope" for rp in rope):
        default = int(cfg.get("original_max_position_embeddings", derived))
    return derived, default


def _check_supported(model_type: str, cfg: dict, top: dict) -> None:
    if top.get("is_encoder_decoder") or cfg.get("is_encoder_decoder"):
        raise UnsupportedModel(f"{model_type}: encoder-decoder models are not supported")
    if any(t in model_type for t in _HYBRID_TYPES) or any(k in cfg for k in _HYBRID_KEYS):
        raise UnsupportedModel(
            f"{model_type}: models with Mamba or linear-attention layers are not supported yet "
            "(their per-request state is planned for a later version)")
    if cfg.get("index_topk") is not None:
        raise UnsupportedModel(f"{model_type}: sparse-attention MLA (DeepSeek V3.2 style) is not supported yet")
    if cfg.get("global_head_dim") or cfg.get("num_global_key_value_heads"):
        raise UnsupportedModel(f"{model_type}: full-attention layers with their own head size or KV-head count "
                               "(Gemma 4 style) are not supported yet")
    if cfg.get("attention_chunk_size"):
        raise UnsupportedModel(f"{model_type}: chunked local attention (Llama 4 style) is not supported yet")
    if cfg.get("kv_lora_rank") is not None and model_type not in _MLA_TYPES:
        raise UnsupportedModel(f"{model_type}: has kv_lora_rank but vLLM does not run it as MLA; "
                               "its attention layout is not supported yet")


def kv_cache_dtype_from_quant(quant: dict | None) -> str | None:
    """vLLM resolves --kv-cache-dtype auto from a modelopt quantization config
    (vllm/utils/torch_utils.py get_kv_cache_quant_algo_string)."""
    if not isinstance(quant, dict) or not str(quant.get("quant_method", "")).startswith("modelopt"):
        return None
    inner = quant.get("quantization", quant)
    algo = (inner.get("kv_cache_scheme") or quant.get("kv_cache_scheme")
            or inner.get("kv_cache_quant_algo") or quant.get("kv_cache_quant_algo"))
    if isinstance(algo, dict):
        if algo.get("dynamic") is False and algo.get("num_bits") == 8 and algo.get("type") == "float":
            return "fp8"
        if algo.get("num_bits") == 4 and algo.get("type") == "float":
            return "nvfp4"
        return None
    if isinstance(algo, str):
        return {"fp8": "fp8", "nvfp4": "nvfp4"}.get(algo.lower())
    return None


def encoder_spec(container: dict, preprocessor: dict | None) -> EncoderSpec | None:
    """The multimodal encoder vLLM profiles at startup, from the config that holds the text config."""
    for key in _ENCODER_CONFIG_KEYS:
        enc = container.get(key)
        if not isinstance(enc, dict):
            continue
        hidden = _int_or_none(_first(enc, "hidden_size", "d_model", "embed_dim", "width"))
        inter = _int_or_none(_first(enc, "intermediate_size", "encoder_ffn_dim", "mlp_dim", "ffn_dim"))
        merge = _int_or_none(enc.get("spatial_merge_size")) or 1
        patch = _int_or_none(enc.get("patch_size"))
        max_pixels = None
        if preprocessor:
            size = preprocessor.get("size") if isinstance(preprocessor.get("size"), dict) else {}
            max_pixels = _int_or_none(preprocessor.get("max_pixels")) or _int_or_none(size.get("longest_edge"))
        # Qwen2-VL-style processors: the largest item has max_pixels / (patch * merge)^2 tokens
        # (Qwen2VLProcessingInfo.get_image_size_with_most_features in vllm/model_executor/models/qwen2_vl.py).
        item_tokens = max_pixels // (patch * merge) ** 2 if (max_pixels and patch and merge > 1) else None
        return EncoderSpec(kind=key.split("_")[0], hidden_size=hidden, intermediate_size=inter,
                           spatial_merge_size=merge, max_item_tokens=item_tokens)
    return None


def spec_from_config(top: dict, tokenizer_config: dict | None = None, preprocessor: dict | None = None) -> ModelSpec:
    cfg, container, path = text_config(top)
    model_type = cfg.get("model_type") or top.get("model_type") or "unknown"
    _check_supported(model_type, cfg, top)

    n = _first(cfg, "num_hidden_layers", "n_layer", "num_layers")
    heads = _first(cfg, "num_attention_heads", "n_head", "num_heads")
    hidden = _first(cfg, "hidden_size", "n_embd", "d_model")
    if not n or not heads:
        raise MissingConfigFields(f"{model_type}: config.json has no layer or attention-head count")

    if cfg.get("multi_query") is True:
        kv_heads = 1
    else:
        # vllm/transformers_utils/config.py CONFIG_FIELD_ALIASES["total_num_kv_heads"]
        kv_heads = _first(cfg, "n_head_kv", "num_kv_heads", "num_key_value_heads", "multi_query_group_num",
                          "num_attention_groups", default=heads)
    head_dim = cfg.get("head_dim")
    if not head_dim:
        if not hidden:
            raise MissingConfigFields(f"{model_type}: cannot derive head_dim (no head_dim or hidden_size)")
        if hidden % heads:
            raise UnsupportedModel(f"{model_type}: hidden_size {hidden} is not divisible by {heads} heads")
        head_dim = hidden // heads

    notes = []
    if path != "config.json":
        notes.append(f"language model read from {path}")
    kv_lora_rank = cfg.get("kv_lora_rank")
    qk_rope = cfg.get("qk_rope_head_dim")
    if kv_lora_rank is not None:
        if qk_rope is None:
            raise UnsupportedModel(f"{model_type}: MLA config without qk_rope_head_dim")
        layer_types, window = [MLA] * n, None
    else:
        explicit = cfg.get("layer_types")
        if explicit:
            if len(explicit) != n:
                raise UnsupportedModel(f"{model_type}: layer_types has {len(explicit)} entries for {n} layers")
            mapping = {"full_attention": FULL, "sliding_attention": SLIDING}
            bad = sorted({t for t in explicit if t not in mapping})
            if bad:
                raise UnsupportedModel(f"{model_type}: unsupported layer types {bad}")
            layer_types = [mapping[t] for t in explicit]
            window = cfg.get("sliding_window") if SLIDING in layer_types else None
        else:
            layer_types, window = _default_layer_types(model_type, cfg, n)
        if SLIDING in layer_types and not window:
            raise UnsupportedModel(f"{model_type}: sliding-window layers without a sliding_window size")
        if window and model_type == "cohere2":
            window += 1   # vllm/model_executor/models/commandr.py passes sliding_window + 1

    dtype = str(_first(cfg, "torch_dtype", "dtype", default=_first(top, "torch_dtype", "dtype", default="bfloat16")))
    if dtype not in _DTYPE_BYTES:
        notes.append(f"unrecognised dtype '{dtype}', assuming 16-bit")
    experts = _first(cfg, "num_local_experts", "num_experts", "n_routed_experts")
    quant = _first(top, "quantization_config", default=cfg.get("quantization_config"))
    max_len, default_len = context_lengths(cfg, model_type, tokenizer_config)
    kv_default = kv_cache_dtype_from_quant(quant)
    if kv_default:
        notes.append(f"the checkpoint's quantization config sets the KV cache to {kv_default}")
    return ModelSpec(
        model_type=model_type,
        num_layers=int(n),
        hidden_size=int(hidden) if hidden else None,
        num_heads=int(heads),
        num_kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        v_head_dim=int(cfg.get("v_head_dim") or head_dim) if kv_lora_rank is None else 0,
        vocab_size=_first(cfg, "vocab_size", "padded_vocab_size", default=top.get("vocab_size")),
        max_position_embeddings=max_len,
        default_max_model_len=default_len,
        layer_types=layer_types,
        sliding_window=int(window) if window else None,
        kv_lora_rank=int(kv_lora_rank) if kv_lora_rank is not None else None,
        qk_rope_head_dim=int(qk_rope) if qk_rope is not None else None,
        tie_word_embeddings=bool(_first(top, "tie_word_embeddings", default=cfg.get("tie_word_embeddings", False))),
        dtype=dtype,
        quantization=quant,
        default_kv_cache_dtype=kv_default,
        num_experts=int(experts) if experts else None,
        source="config",
        notes=notes,
        intermediate_size=None if experts else _int_or_none(_first(cfg, "intermediate_size", "ffn_hidden_size",
                                                                   "n_inner")),
        architecture=_architecture(top),
        encoder=encoder_spec(container, preprocessor),
    )


def _architecture(top: dict) -> str | None:
    archs = top.get("architectures")
    return archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else None


# ------------------------------------------------------------------- GGUF

# llama.cpp architecture names whose defaults differ from a plain full-attention model.
_GGUF_ARCH_TO_MODEL_TYPE = {"gemma2": "gemma2", "gemma3": "gemma3_text", "cohere2": "cohere2", "gpt-oss": "gpt_oss",
                            "olmo3": "olmo3"}


def spec_from_gguf(meta: dict) -> ModelSpec:
    arch = meta.get("general.architecture")
    if not arch:
        raise UnsupportedModel("GGUF file has no general.architecture")

    def g(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    n = g("block_count")
    hidden = g("embedding_length")
    heads = g("attention.head_count")
    if isinstance(heads, list):
        heads = max(heads)
    if not n or not heads:
        raise UnsupportedModel(f"GGUF '{arch}': missing block_count or attention.head_count")
    if g("ssm.state_size") is not None or g("ssm.inner_size") is not None:
        raise UnsupportedModel(f"GGUF '{arch}': models with Mamba/SSM layers are not supported yet")
    kv = g("attention.head_count_kv", heads)
    if isinstance(kv, list):
        if len(set(kv)) > 1:
            raise UnsupportedModel(f"GGUF '{arch}': per-layer KV head counts are not supported yet")
        kv = kv[0]
    head_dim = g("attention.key_length") or (hidden // heads if hidden else None)
    if not head_dim:
        raise UnsupportedModel(f"GGUF '{arch}': cannot derive head size")
    v_head_dim = g("attention.value_length") or head_dim
    tokens = meta.get("tokenizer.ggml.tokens")
    vocab = g("vocab_size") or (int(tokens) if isinstance(tokens, int) else (len(tokens) if tokens else None))

    kv_lora = g("attention.kv_lora_rank")
    model_type = _GGUF_ARCH_TO_MODEL_TYPE.get(arch, arch)
    if kv_lora is not None:
        rope = g("rope.dimension_count")
        layer_types, window = [MLA] * n, None
    else:
        rope = None
        cfg = {"sliding_window": g("attention.sliding_window")}
        if g("attention.sliding_window_pattern") is not None:
            cfg["sliding_window_pattern"] = g("attention.sliding_window_pattern")
        layer_types, window = _default_layer_types(model_type, cfg, n)
        if window and model_type == "cohere2":
            window += 1
    return ModelSpec(
        model_type=model_type, num_layers=int(n), hidden_size=hidden, num_heads=int(heads), num_kv_heads=int(kv),
        head_dim=int(head_dim), v_head_dim=int(v_head_dim) if kv_lora is None else 0, vocab_size=vocab,
        max_position_embeddings=g("context_length"), default_max_model_len=g("context_length"),
        layer_types=layer_types,
        sliding_window=int(window) if window else None,
        kv_lora_rank=int(kv_lora) if kv_lora is not None else None, qk_rope_head_dim=int(rope) if rope else None,
        tie_word_embeddings=False, dtype="bfloat16", num_experts=g("expert_count"), source="gguf",
        intermediate_size=None if g("expert_count") else _int_or_none(g("feed_forward_length")),
    )


# --------------------------------------------------------------- tensor shapes

_LAYER_RE = re.compile(r"^((?:[^.]*\.)*?)(?:layers|h|blocks|layer)\.(\d+)\.")
_K_PROJ_RE = re.compile(r"\.(k_proj|key)\.weight$")
# Prefixes of multimodal encoders and speech-output parts; their layers are not the language model's.
_NON_LM_PREFIX = re.compile(r"vision|visual|vit\b|siglip|clip|image|audio|whisper|encoder|mm_projector|"
                            r"talker|token2wav|code2wav|tts|vocoder", re.I)
# Checkpoint dtypes whose shape is the logical shape (packed quantized formats are not).
_PLAIN_DTYPES = {"F32", "F16", "BF16", "F64", "F8_E4M3", "F8_E5M2"}


def language_model_layers(tensors) -> tuple[str, int] | None:
    """(name prefix, layer count) of the language model's transformer layers in the weights.

    Layers under encoder or speech-output prefixes are ignored. With several candidates, the
    one with the most layers wins (a language model's prefix also holds its MTP layers).
    """
    groups: dict[str, set[int]] = {}
    for t in tensors:
        m = _LAYER_RE.search(t.name)
        if m and not _NON_LM_PREFIX.search(m.group(1)):
            groups.setdefault(m.group(1), set()).add(int(m.group(2)))
    if not groups:
        return None
    prefix = max(groups, key=lambda p: (len(groups[p]), -len(p)))
    return prefix, max(groups[prefix]) + 1


def kv_widths_from_tensors(tensors, prefix: str) -> dict[int, int]:
    """Output rows of each layer's K projection (num_kv_heads * head_dim) under the language model prefix."""
    out = {}
    for t in tensors:
        if not t.name.startswith(prefix) or not _K_PROJ_RE.search(t.name) or len(t.shape) != 2:
            continue
        if t.dtype not in _PLAIN_DTYPES:
            continue
        m = _LAYER_RE.search(t.name)
        if m and m.group(1) == prefix:
            out[int(m.group(2))] = t.shape[0]
    return out


def check_against_tensors(spec: ModelSpec, tensors) -> None:
    """Refuse when the config's attention geometry contradicts the language model's weights."""
    if spec.is_mla or spec.source != "config":
        return
    lm = language_model_layers(tensors)
    if lm is None:
        return
    prefix, n = lm
    widths = kv_widths_from_tensors(tensors, prefix)
    expected = spec.num_kv_heads * spec.head_dim
    bad = sorted({w for w in widths.values() if w != expected})
    if bad:
        raise UnsupportedModel(
            f"{spec.model_type}: the config gives {spec.num_kv_heads} KV heads x head_dim {spec.head_dim} = "
            f"{expected} K channels per layer, but the weights under '{prefix}' have {bad}. kvplan will not "
            "guess which is right; check config.json (or the sub-config it was read from) against the "
            "checkpoint, or report the model at https://github.com/ShawnNotFound/kvplan/issues")
    if n < spec.num_layers:
        raise UnsupportedModel(
            f"{spec.model_type}: the config gives {spec.num_layers} layers, but the weights under '{prefix}' "
            f"have only {n}. kvplan will not guess which is right; check config.json against the checkpoint")


def spec_from_tensors(tensors, model_type: str = "unknown", reason: str = "config.json has no attention fields",
                      architecture: str | None = None) -> ModelSpec:
    """Fallback when config.json lacks the attention fields: derive them from weight shapes.

    Layer count comes from the layer indices of the language model's tensors. The KV width per
    layer (num_kv_heads * head_dim) comes from the K projection's output rows, which is all
    the KV-cache size needs. Head counts themselves cannot be recovered exactly, so
    tensor-parallel checks are skipped for such models.
    """
    lm = language_model_layers(tensors)
    if lm is None:
        raise UnsupportedModel(f"{model_type}: cannot find transformer layers in the weight names")
    prefix, n = lm
    widths = kv_widths_from_tensors(tensors, prefix)
    if not widths:
        raise UnsupportedModel(f"{model_type}: no separate K projection in the language model's weights; "
                               "cannot derive the KV width")
    if len(set(widths.values())) > 1:
        raise UnsupportedModel(f"{model_type}: K projections of different widths across layers "
                               f"({sorted(set(widths.values()))}); per-layer KV sizes are not supported yet")
    kv_width = next(iter(widths.values()))
    emb = [t for t in tensors if t.name.startswith(prefix)
           and re.search(r"(embed_tokens|wte|word_embeddings|tok_embeddings)\.weight$", t.name)]
    vocab, hidden = (emb[0].shape[0], emb[0].shape[1]) if emb and len(emb[0].shape) == 2 else (None, None)
    return ModelSpec(
        model_type=model_type, num_layers=n, hidden_size=hidden, num_heads=1, num_kv_heads=1,
        head_dim=kv_width, v_head_dim=kv_width, vocab_size=vocab, max_position_embeddings=None,
        layer_types=[FULL] * n, source="tensor shapes", architecture=architecture,
        notes=[f"attention geometry derived from the weight shapes under '{prefix}' because {reason}; "
               "head counts are unknown, so the KV size assumes no tensor-parallel head replication"],
    )
