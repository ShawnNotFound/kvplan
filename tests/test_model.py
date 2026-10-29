import pytest

from kvplan.model import FULL, MLA, SLIDING, MissingConfigFields, UnsupportedModel, spec_from_config, spec_from_gguf

LLAMA_8B = {
    "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
    "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 131072, "torch_dtype": "bfloat16",
}


def test_llama_gqa_fields():
    s = spec_from_config(LLAMA_8B)
    assert (s.num_layers, s.num_heads, s.num_kv_heads, s.head_dim, s.v_head_dim) == (32, 32, 8, 128, 128)
    assert s.layer_types == [FULL] * 32 and s.sliding_window is None


def test_kv_heads_default_to_attention_heads():
    cfg = {"model_type": "gpt2", "n_layer": 12, "n_embd": 768, "n_head": 12, "vocab_size": 50257}
    s = spec_from_config(cfg)
    assert (s.num_kv_heads, s.head_dim) == (12, 64)


def test_multi_query():
    cfg = {"model_type": "gpt_bigcode", "n_layer": 2, "n_embd": 512, "n_head": 8, "multi_query": True}
    assert spec_from_config(cfg).num_kv_heads == 1


def test_explicit_head_dim_wins():
    cfg = dict(LLAMA_8B, head_dim=96)
    assert spec_from_config(cfg).head_dim == 96


def test_gemma2_alternates_sliding_from_layer_0():
    cfg = {"model_type": "gemma2", "num_hidden_layers": 4, "hidden_size": 3584, "num_attention_heads": 16,
           "num_key_value_heads": 8, "head_dim": 256, "sliding_window": 4096}
    s = spec_from_config(cfg)
    assert s.layer_types == [SLIDING, FULL, SLIDING, FULL]
    assert s.sliding_window == 4096


def test_gemma3_every_sixth_layer_is_full():
    cfg = {"model_type": "gemma3", "text_config": {
        "model_type": "gemma3_text", "num_hidden_layers": 12, "hidden_size": 2560, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 256, "sliding_window": 1024}}
    s = spec_from_config(cfg)
    assert [i for i, t in enumerate(s.layer_types) if t == FULL] == [5, 11]


def test_explicit_layer_types_override_defaults():
    cfg = {"model_type": "gpt_oss", "num_hidden_layers": 3, "hidden_size": 2880, "num_attention_heads": 64,
           "num_key_value_heads": 8, "head_dim": 64, "sliding_window": 128,
           "layer_types": ["full_attention", "full_attention", "sliding_attention"]}
    assert spec_from_config(cfg).layer_types == [FULL, FULL, SLIDING]


def test_qwen2_sliding_window_is_off_unless_enabled():
    cfg = {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
           "num_key_value_heads": 4, "sliding_window": 131072, "use_sliding_window": False, "max_window_layers": 28}
    s = spec_from_config(cfg)
    assert s.layer_types == [FULL] * 28 and s.sliding_window is None


def test_mistral_sliding_window_applies_to_every_layer():
    # No layer_types: vLLM sets CacheConfig.sliding_window, which every layer then uses.
    s = spec_from_config(dict(LLAMA_8B, model_type="mistral", sliding_window=4096))
    assert s.layer_types == [SLIDING] * 32 and s.sliding_window == 4096
    assert spec_from_config(dict(LLAMA_8B, model_type="mistral", sliding_window=None)).layer_types == [FULL] * 32


def test_qwen2_mixed_sliding_pattern_is_refused_like_vllm():
    cfg = {"model_type": "qwen2", "num_hidden_layers": 4, "hidden_size": 256, "num_attention_heads": 4,
           "sliding_window": 512, "use_sliding_window": True, "max_window_layers": 2}
    with pytest.raises(UnsupportedModel, match="refuses"):
        spec_from_config(cfg)
    assert spec_from_config(dict(cfg, max_window_layers=0)).layer_types == [SLIDING] * 4


def test_qwen2_moe_runs_full_attention():
    cfg = {"model_type": "qwen2_moe", "num_hidden_layers": 4, "hidden_size": 256, "num_attention_heads": 4,
           "sliding_window": 512, "use_sliding_window": True, "max_window_layers": 4}
    assert spec_from_config(cfg).layer_types == [FULL] * 4


def test_cohere2_window_is_one_larger():
    cfg = {"model_type": "cohere2", "num_hidden_layers": 4, "hidden_size": 256, "num_attention_heads": 4,
           "sliding_window": 4096}
    s = spec_from_config(cfg)
    assert s.layer_types == [SLIDING, SLIDING, SLIDING, FULL] and s.sliding_window == 4097


def test_chatglm_kv_heads_alias():
    cfg = {"model_type": "chatglm", "num_layers": 40, "hidden_size": 4096, "num_attention_heads": 32,
           "multi_query_group_num": 2, "padded_vocab_size": 151552, "seq_length": 131072}
    s = spec_from_config(cfg)
    assert (s.num_kv_heads, s.head_dim, s.vocab_size, s.max_position_embeddings) == (2, 128, 151552, 131072)


def test_kv_lora_rank_outside_vllm_mla_list_is_rejected():
    cfg = {"model_type": "minicpm3", "num_hidden_layers": 62, "hidden_size": 2560, "num_attention_heads": 40,
           "kv_lora_rank": 256, "qk_rope_head_dim": 32}
    with pytest.raises(UnsupportedModel, match="does not run it as MLA"):
        spec_from_config(cfg)


def test_llama4_chunked_attention_rejected():
    cfg = dict(LLAMA_8B, model_type="llama4_text", attention_chunk_size=8192)
    with pytest.raises(UnsupportedModel, match="chunked local attention"):
        spec_from_config(cfg)


def test_context_length_takes_smallest_key_and_tokenizer_limit():
    s = spec_from_config(dict(LLAMA_8B, max_position_embeddings=8192, seq_length=4096))
    assert s.max_position_embeddings == 4096
    s = spec_from_config(LLAMA_8B, tokenizer_config={"model_max_length": 2048})
    assert s.max_position_embeddings == 2048
    s = spec_from_config(LLAMA_8B, tokenizer_config={"model_max_length": 1e30})
    assert s.max_position_embeddings == 131072


def test_linear_rope_scaling_multiplies_context():
    cfg = dict(LLAMA_8B, max_position_embeddings=4096, rope_scaling={"type": "linear", "factor": 8.0})
    assert spec_from_config(cfg).max_position_embeddings == 32768
    # llama3 and yarn configs already state the scaled length.
    cfg = dict(LLAMA_8B, rope_scaling={"rope_type": "llama3", "factor": 8.0})
    assert spec_from_config(cfg).max_position_embeddings == 131072


def test_longrope_defaults_to_original_length():
    cfg = dict(LLAMA_8B, model_type="phi3", max_position_embeddings=131072, original_max_position_embeddings=4096,
               rope_scaling={"type": "longrope", "short_factor": [1.0], "long_factor": [2.0]})
    s = spec_from_config(cfg)
    assert (s.max_position_embeddings, s.default_max_model_len) == (131072, 4096)


def test_modelopt_fp8_kv_cache():
    quant = {"quant_method": "modelopt", "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"}}
    assert spec_from_config(dict(LLAMA_8B, quantization_config=quant)).default_kv_cache_dtype == "fp8"
    assert spec_from_config(dict(LLAMA_8B, quantization_config={"quant_method": "gptq"})).default_kv_cache_dtype \
        is None


def test_mla():
    cfg = {"model_type": "deepseek_v3", "num_hidden_layers": 61, "hidden_size": 7168, "num_attention_heads": 128,
           "num_key_value_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
           "v_head_dim": 128, "n_routed_experts": 256}
    s = spec_from_config(cfg)
    assert s.is_mla and s.layer_types == [MLA] * 61 and s.num_experts == 256


def test_hybrid_models_are_rejected():
    with pytest.raises(UnsupportedModel, match="Mamba or linear-attention"):
        spec_from_config({"model_type": "qwen3_next", "num_hidden_layers": 4, "hidden_size": 64,
                          "num_attention_heads": 4})
    with pytest.raises(UnsupportedModel, match="Mamba or linear-attention"):
        spec_from_config(dict(LLAMA_8B, hybrid_override_pattern="M-M-"))


def test_sparse_mla_rejected():
    with pytest.raises(UnsupportedModel, match="sparse-attention"):
        spec_from_config({"model_type": "deepseek_v32", "num_hidden_layers": 2, "hidden_size": 64,
                          "num_attention_heads": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "index_topk": 2048})


def test_encoder_decoder_rejected():
    with pytest.raises(UnsupportedModel, match="encoder-decoder"):
        spec_from_config({"model_type": "t5", "num_layers": 6, "d_model": 512, "num_heads": 8,
                          "is_encoder_decoder": True})


def test_missing_fields_are_marked_for_fallback():
    with pytest.raises(MissingConfigFields):
        spec_from_config({"model_type": "weird", "hidden_size": 64})


def test_unknown_layer_type_is_not_silently_full():
    cfg = dict(LLAMA_8B, num_hidden_layers=2, layer_types=["full_attention", "chunked_attention"])
    with pytest.raises(UnsupportedModel, match="unsupported layer types"):
        spec_from_config(cfg)


def test_gguf_metadata():
    meta = {"general.architecture": "llama", "llama.block_count": 32, "llama.embedding_length": 4096,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8, "llama.context_length": 8192,
            "tokenizer.ggml.tokens": 128256}
    s = spec_from_gguf(meta)
    assert (s.num_layers, s.num_kv_heads, s.head_dim, s.vocab_size) == (32, 8, 128, 128256)


def test_gguf_gemma2_uses_alternating_layers():
    meta = {"general.architecture": "gemma2", "gemma2.block_count": 4, "gemma2.embedding_length": 3584,
            "gemma2.attention.head_count": 16, "gemma2.attention.head_count_kv": 8,
            "gemma2.attention.key_length": 256, "gemma2.attention.sliding_window": 4096}
    assert spec_from_gguf(meta).layer_types == [SLIDING, FULL, SLIDING, FULL]
