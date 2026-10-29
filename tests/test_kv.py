"""KV sizes and block planning, checked against hand calculations and vLLM's documented rules."""

import pytest

from kvplan.kv import bytes_per_token, kv_capacity, kv_layout, max_len_that_fits
from kvplan.model import FULL, MLA, SLIDING, spec_from_config

LLAMA_8B = spec_from_config({
    "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
    "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 131072})
LLAMA_70B = spec_from_config({
    "model_type": "llama", "num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64,
    "num_key_value_heads": 8, "vocab_size": 128256})
DEEPSEEK_V3 = spec_from_config({
    "model_type": "deepseek_v3", "num_hidden_layers": 61, "hidden_size": 7168, "num_attention_heads": 128,
    "num_key_value_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64})


def total_per_token(spec, tp=1, dtype="auto"):
    layout = kv_layout(spec, tp, dtype)
    return sum(layer.page_bytes for layer in layout.layers) // layout.block_size


def test_llama_8b_kv_per_token():
    # 2 (K,V) * 8 heads * 128 dim * 2 bytes * 32 layers
    assert total_per_token(LLAMA_8B) == 131072


def test_tp_splits_kv_heads():
    assert total_per_token(LLAMA_70B, tp=1) == 2 * 8 * 128 * 2 * 80
    assert total_per_token(LLAMA_70B, tp=8) == 2 * 1 * 128 * 2 * 80


def test_kv_heads_replicated_beyond_kv_head_count():
    # 8 KV heads on 16 GPUs: each GPU still stores one full head.
    assert total_per_token(LLAMA_70B, tp=16) == total_per_token(LLAMA_70B, tp=8)


def test_fp8_halves_kv():
    assert total_per_token(LLAMA_8B, dtype="fp8") * 2 == total_per_token(LLAMA_8B)


def test_unknown_kv_dtype_rejected():
    with pytest.raises(ValueError):
        kv_layout(LLAMA_8B, 1, "int3")


def test_mla_latent_is_not_split_by_tp():
    assert bytes_per_token(DEEPSEEK_V3, MLA, 8, 2) == (512 + 64) * 2
    assert total_per_token(DEEPSEEK_V3, tp=8) == 576 * 2 * 61


def test_uniform_model_one_group():
    layout = kv_layout(LLAMA_8B, 1)
    assert len(layout.groups) == 1
    assert layout.bytes_per_block == 16 * 131072


def hybrid(n_full, n_sw, window=4096):
    types = ["full_attention"] * n_full + ["sliding_attention"] * n_sw
    return spec_from_config({"model_type": "gemma2", "num_hidden_layers": n_full + n_sw, "hidden_size": 2048,
                             "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
                             "sliding_window": window, "layer_types": types})


def test_vllm_docstring_example_10_full_20_sw_is_3_groups_of_10():
    layout = kv_layout(hybrid(10, 20), 1)
    assert sorted(len(g) for g in layout.groups) == [10, 10, 10]
    page = 16 * 2 * 4 * 256 * 2
    assert layout.bytes_per_block == 10 * page


def test_close_bucket_sizes_use_the_larger_group_and_pad():
    # 5 full + 7 sliding: 7 < 5 * 1.5, so group_size = 7 and the full bucket is padded by 2.
    layout = kv_layout(hybrid(5, 7), 1)
    assert sorted(len(g) for g in layout.groups) == [5, 7]
    assert layout.padded_layers == 2
    assert layout.bytes_per_block == 7 * 16 * 2 * 4 * 256 * 2


def test_sliding_window_blocks_are_capped():
    layout = kv_layout(hybrid(1, 1, window=1024), 1)
    sw = [layer for layer in layout.layers if layer.kind == SLIDING][0]
    full = [layer for layer in layout.layers if layer.kind == FULL][0]
    # vLLM: cdiv(min(window - 1 + in_flight, max_len), block) + 1
    assert sw.blocks_per_request(131072, 16, 2048) == -(-(1024 - 1 + 2048) // 16) + 1
    assert full.blocks_per_request(131072, 16, 2048) == 131072 // 16
    # Short requests never need more than max_model_len tokens' worth.
    assert sw.blocks_per_request(512, 16, 2048) == 512 // 16 + 1


def test_padding_layers_count_in_the_start_check():
    # 5 full + 7 sliding pads the full bucket to 7: one request needs whole pool blocks for 7 layers.
    layout = kv_layout(hybrid(5, 7, window=1024), 1)
    need = layout.bytes_needed_for_one_request(65536, 2048)
    assert need == layout.bytes_per_block * layout.blocks_per_request(65536, 2048)
    real_layers_only = sum(layer.blocks_per_request(65536, 16, 2048) * layer.page_bytes for layer in layout.layers)
    assert need > real_layers_only


def test_modelopt_fp8_checkpoint_makes_auto_kv_fp8():
    spec = spec_from_config({"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
                             "num_attention_heads": 32, "num_key_value_heads": 8,
                             "quantization_config": {"quant_method": "modelopt", "kv_cache_quant_algo": "FP8"}})
    assert total_per_token(spec) == 131072 // 2


def test_capacity_concurrency_and_tokens():
    layout = kv_layout(LLAMA_8B, 1)
    avail = 10 * 2**30
    cap = kv_capacity(layout, avail, 8192, 2048)
    blocks = avail // (16 * 131072)
    assert cap.num_blocks == blocks
    assert cap.max_concurrency == pytest.approx(blocks / (8192 // 16))
    assert cap.kv_tokens == int(cap.max_concurrency * 8192)
    assert cap.fits_one_request


def test_does_not_fit_and_estimates_max_len():
    layout = kv_layout(LLAMA_8B, 1)
    avail = 4 * 2**30   # 4 GiB holds 32768 tokens at 128 KiB/token
    cap = kv_capacity(layout, avail, 131072, 2048)
    assert not cap.fits_one_request
    # vLLM reserves one block before checking: (4 GiB - 1 block) / 128 KiB, rounded down to a block multiple.
    usable = avail - layout.bytes_per_block
    assert cap.max_model_len_that_fits == usable // 131072 // 16 * 16
    assert cap.max_model_len_that_fits == max_len_that_fits(layout, usable, 131072, 2048)


def test_no_memory():
    cap = kv_capacity(kv_layout(LLAMA_8B, 1), 0, 8192, 2048)
    assert not cap.fits_one_request and cap.num_blocks == 0 and cap.max_model_len_that_fits == 0
