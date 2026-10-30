"""Header readers, the loader's fallbacks, and per-GPU weight sharding."""

import pytest

from helpers import llama_like_tensors, write_gguf, write_model_dir, write_safetensors
from kvplan.loader import load
from kvplan.model import spec_from_config
from kvplan.sources import LocalSource
from kvplan.weights import weight_bytes_per_rank
from kvplan.weights_meta import gguf_files, read_gguf, read_safetensors_header, safetensors_files

CFG = {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 8,
       "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 1000, "max_position_embeddings": 2048,
       "torch_dtype": "bfloat16"}


def test_safetensors_header_sizes(tmp_path):
    write_safetensors(tmp_path / "m.safetensors", {"a": ("BF16", (3, 4)), "b": ("F32", (5,))})
    t = {x.name: x for x in read_safetensors_header(LocalSource(tmp_path), "m.safetensors")}
    assert t["a"].nbytes == 24 and t["a"].shape == (3, 4)
    assert t["b"].nbytes == 20 and t["b"].numel == 5


def test_safetensors_file_selection():
    files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
             "consolidated.safetensors", "sub/x.safetensors"]
    assert safetensors_files(files, None) == files[:2]
    assert safetensors_files(["consolidated.safetensors"], None) == ["consolidated.safetensors"]
    assert safetensors_files(files, {"weight_map": {"a": "model-00002-of-00002.safetensors"}}) == [
        "model-00002-of-00002.safetensors"]


def test_gguf_reader(tmp_path):
    meta = {"general.architecture": "llama", "general.alignment": 32, "llama.block_count": 2,
            "llama.attention.head_count": 8, "llama.embedding_length": 64, "tokenizer.ggml.tokens": ["a", "b"]}
    write_gguf(tmp_path / "m-Q4_K_M.gguf", meta, [("token_embd.weight", (64, 2), 0, 512),
                                                  ("blk.0.attn_k.weight", (64, 16), 12, 100)])
    g = read_gguf(LocalSource(tmp_path), "m-Q4_K_M.gguf")
    assert g.metadata["llama.block_count"] == 2 and g.metadata["tokenizer.ggml.tokens"] == ["a", "b"]
    sizes = {t.name: t for t in g.tensors}
    assert sizes["token_embd.weight"].nbytes == 512 and sizes["token_embd.weight"].shape == (2, 64)
    assert sizes["blk.0.attn_k.weight"].nbytes == 128   # 100 bytes padded to the 32-byte alignment


def test_gguf_split_and_quant_selection():
    files = ["a-Q8_0.gguf", "a-Q4_K_M-00001-of-00002.gguf", "a-Q4_K_M-00002-of-00002.gguf", "mmproj-f16.gguf"]
    assert gguf_files(files) == ["a-Q4_K_M-00001-of-00002.gguf", "a-Q4_K_M-00002-of-00002.gguf"]
    assert gguf_files(files, "q8_0") == ["a-Q8_0.gguf"]
    with pytest.raises(Exception, match="no GGUF file matching"):
        gguf_files(files, "IQ2")


def tensors_for(tmp_path, **kw):
    t = llama_like_tensors(layers=2, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000, **kw)
    model = load(LocalSource(write_model_dir(tmp_path / "m", CFG, t)))
    return model


def test_weights_tp1_equal_bf16_checkpoint(tmp_path):
    m = tensors_for(tmp_path)
    w = weight_bytes_per_rank(m.spec, m.tensors, tp=1)
    # Vocab 1000 is padded to 1024 rows for embed and lm_head.
    padding = 2 * 24 * 64 * 2
    assert w.per_rank_bytes == w.total_checkpoint_bytes + padding
    assert not w.tied


def test_weights_tied_embedding_detected(tmp_path):
    m = tensors_for(tmp_path, tied=True)
    assert weight_bytes_per_rank(m.spec, m.tensors).tied


def test_fp32_checkpoint_loaded_in_16_bit(tmp_path):
    m = tensors_for(tmp_path, dtype="F32")
    w = weight_bytes_per_rank(m.spec, m.tensors)
    assert w.per_rank_bytes == pytest.approx(w.total_checkpoint_bytes / 2, rel=0.1)


def test_tp_split_and_kv_replication(tmp_path):
    m = tensors_for(tmp_path)
    hd, hidden, inter, layers = 8, 64, 128, 2
    w2 = weight_bytes_per_rank(m.spec, m.tensors, tp=2)
    w4 = weight_bytes_per_rank(m.spec, m.tensors, tp=4)
    norms = (2 * layers + 1) * hidden * 2
    emb = 2 * (1024 // 2) * hidden * 2
    q_o = 2 * (8 * hd * hidden) * 2 * layers
    mlp = 3 * inter * hidden * 2 * layers
    kv = 2 * (2 * hd * hidden) * 2 * layers
    assert w2.per_rank_bytes == norms + emb + (q_o + mlp + kv) // 2
    # 2 KV heads on 4 GPUs: K/V projections hold one head per GPU, i.e. half, not a quarter.
    emb4 = 2 * (1024 // 4) * hidden * 2
    assert w4.per_rank_bytes == norms + emb4 + (q_o + mlp) // 4 + kv // 2


def test_tiny_rope_cache_is_counted_without_a_note(tmp_path):
    cfg = dict(CFG, rope_theta=10000.0)
    t = llama_like_tensors(layers=2, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000)
    m = load(LocalSource(write_model_dir(tmp_path / "m", cfg, t)))
    w = weight_bytes_per_rank(m.spec, m.tensors)
    # 2048 positions x 8 dims in bf16: counted, but a "0.00 GiB" note would only be noise.
    assert w.buffer_bytes == 2048 * 8 * 2
    assert not any("RoPE" in n for n in w.notes)


def test_mtp_layers_are_skipped(tmp_path):
    t = llama_like_tensors(layers=3, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000)
    m = load(LocalSource(write_model_dir(tmp_path / "m", CFG, t)))   # config says 2 layers
    w = weight_bytes_per_rank(m.spec, m.tensors)
    assert w.skipped_bytes > 0 and any("multi-token" in n for n in w.notes)


def test_loader_falls_back_to_tensor_shapes(tmp_path):
    t = llama_like_tensors(layers=2, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000)
    m = load(LocalSource(write_model_dir(tmp_path / "m", {"model_type": "custom"}, t)))
    assert m.spec.source == "tensor shapes"
    assert m.spec.num_layers == 2
    # KV width = k_proj rows = 2 heads * 8 dims.
    assert m.spec.num_kv_heads * m.spec.head_dim == 16


def test_loader_reads_gguf_when_no_safetensors(tmp_path):
    meta = {"general.architecture": "llama", "llama.block_count": 2, "llama.attention.head_count": 8,
            "llama.attention.head_count_kv": 2, "llama.embedding_length": 64, "llama.context_length": 4096}
    write_gguf(tmp_path / "m-Q4_K_M.gguf", meta, [("token_embd.weight", (64, 1000), 12, 36000)])
    m = load(LocalSource(tmp_path))
    assert m.spec.source == "gguf" and m.spec.num_kv_heads == 2 and m.tensors[0].nbytes >= 36000


def test_loader_rejects_empty_dir(tmp_path):
    with pytest.raises(Exception, match="no config.json"):
        load(LocalSource(tmp_path))


def test_spec_has_hidden_size():
    assert spec_from_config(CFG).hidden_size == 64
