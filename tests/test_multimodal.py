"""Multimodal and multi-component models: nested text configs, parts vLLM does not load,
shape-derived fallbacks and the encoder profiling term. Configs under fixtures/configs are the
real config.json files of the named Hugging Face repos."""

import json
from pathlib import Path

import pytest

from helpers import llama_like_tensors, write_model_dir
from kvplan.loader import load
from kvplan.model import UnsupportedModel, spec_from_config, text_config
from kvplan.sources import LocalSource
from kvplan.vllm import GiB, ServeConfig, encoder_profile_bytes, plan
from kvplan.weights import weight_bytes_per_rank

FIXTURES = Path(__file__).parent / "fixtures" / "configs"


def fixture(repo: str, kind: str = "config") -> dict:
    return json.loads((FIXTURES / f"{repo.replace('/', '__')}.{kind}.json").read_text())


@pytest.mark.parametrize("repo, path, geometry", [
    # (layers, heads, kv heads, head_dim, hidden)
    ("llava-hf/llava-1.5-7b-hf", "text_config", (32, 32, 32, 128, 4096)),
    ("unsloth/gemma-3-4b-it", "text_config", (34, 8, 4, 256, 2560)),
    ("Qwen/Qwen3-VL-8B-Instruct", "text_config", (36, 32, 8, 128, 4096)),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "config.json", (28, 28, 4, 128, 3584)),
    ("Qwen/Qwen2.5-Omni-7B", "thinker_config.text_config", (28, 28, 4, 128, 3584)),
    ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "thinker_config.text_config", (48, 32, 4, 128, 2048)),
    ("OpenGVLab/InternVL3-8B", "llm_config", (28, 28, 4, 128, 3584)),
    ("deepseek-ai/deepseek-vl2-tiny", "language_config", (12, 10, 10, 128, 1280)),
])
def test_text_config_resolution_on_real_configs(repo, path, geometry):
    cfg = fixture(repo)
    assert text_config(cfg)[2] == path
    s = spec_from_config(cfg)
    assert (s.num_layers, s.num_heads, s.num_kv_heads, s.head_dim, s.hidden_size) == geometry
    assert s.source == "config"


def test_llava_text_config_takes_llama_defaults():
    s = spec_from_config(fixture("llava-hf/llava-1.5-7b-hf"))
    assert s.max_position_embeddings == 4096          # given in text_config
    assert s.intermediate_size == 11008               # LlamaConfig default
    assert s.vocab_size == 32064


def test_several_text_configs_are_ambiguous():
    with pytest.raises(UnsupportedModel, match="several language-model sub-configs"):
        text_config({"text_config": {"num_hidden_layers": 2}, "decoder": {"num_hidden_layers": 2}})


def test_omni_kv_bytes_per_token_match_vllm():
    # vLLM 0.32 on 2 x RTX A6000 reported 32.54 GiB for 1,218,704 tokens: 28,672 bytes per token per GPU.
    s = spec_from_config(fixture("Qwen/Qwen2.5-Omni-7B"))
    p = plan(s, weight_bytes_per_rank(s, [], 2), ServeConfig(gpu_total_bytes=48 * GiB, tp=2, max_model_len=4096))
    assert p.kv_bytes_per_token == 28_672


# -------------------------------------------------------------- weights and config cross-checks

SMALL = {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 8,
         "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 1000, "max_position_embeddings": 2048}


def prefixed(tensors: dict, prefix: str) -> dict:
    return {prefix + k: v for k, v in tensors.items()}


def omni_dir(tmp_path, arch="Qwen2_5OmniModel"):
    lm = prefixed(llama_like_tensors(layers=2, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000), "thinker.")
    audio = prefixed(llama_like_tensors(layers=4, hidden=32, heads=4, kv_heads=4, inter=64, vocab=10, tied=True),
                     "thinker.audio_tower.")
    talker = prefixed(llama_like_tensors(layers=3, hidden=32, heads=4, kv_heads=1, inter=64, vocab=10), "talker.")
    vocoder = {"token2wav.code2wav_dit_model.proj.weight": ("BF16", (256, 256))}
    cfg = {"architectures": [arch], "model_type": "qwen2_5_omni",
           "thinker_config": {"model_type": "qwen2_5_omni_thinker", "text_config": SMALL},
           "talker_config": {"model_type": "qwen2_5_omni_talker"}}
    return write_model_dir(tmp_path / arch, cfg, lm | audio | talker | vocoder), lm, audio


def test_omni_talker_and_token2wav_are_not_loaded(tmp_path):
    path, lm, audio = omni_dir(tmp_path)
    m = load(LocalSource(path))
    assert m.spec.num_layers == 2 and m.spec.num_kv_heads == 2
    w1 = weight_bytes_per_rank(m.spec, m.tensors, 1)
    thinker_only = [t for t in m.tensors if t.name.startswith("thinker.")]
    assert w1.per_rank_bytes == weight_bytes_per_rank(m.spec, thinker_only, 1).per_rank_bytes
    assert w1.skipped_bytes == sum(t.nbytes for t in m.tensors if not t.name.startswith("thinker."))
    assert any("which vLLM's Qwen2_5OmniModel does not load" in n for n in w1.notes)
    # The audio tower is replicated on every GPU; only the language model is split.
    w2 = weight_bytes_per_rank(m.spec, m.tensors, 2)
    audio_bytes = sum(2 * _numel(s) for _, s in audio.values())
    assert w2.replicated_bytes == w1.replicated_bytes and w2.replicated_bytes >= audio_bytes


def test_unknown_multi_component_model_warns_about_generator_weights(tmp_path):
    path, _, _ = omni_dir(tmp_path, arch="SomeNewOmniModel")
    m = load(LocalSource(path))
    w = weight_bytes_per_rank(m.spec, m.tensors, 1)
    assert any(n.startswith("WARNING: counted talker") for n in w.notes)
    p = plan(m.spec, w, ServeConfig(gpu_total_bytes=24 * GiB))
    assert any("no native implementation of SomeNewOmniModel" in c.message for c in p.checks)


def test_audio_layers_beyond_the_text_layer_count_are_not_mtp(tmp_path):
    path, _, audio = omni_dir(tmp_path)
    w = weight_bytes_per_rank(load(LocalSource(path)).spec, load(LocalSource(path)).tensors, 1)
    assert not any("multi-token-prediction" in n for n in w.notes)


def test_config_contradicting_weights_is_refused(tmp_path):
    cfg = dict(SMALL, num_key_value_heads=4)    # the weights have 2 KV heads
    t = llama_like_tensors(layers=2, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000)
    with pytest.raises(UnsupportedModel, match="kvplan will not guess"):
        load(LocalSource(write_model_dir(tmp_path / "m", cfg, t)))


def test_per_layer_head_sizes_are_refused():
    cfg = dict(SMALL, model_type="gemma4_text", global_head_dim=16, num_global_key_value_heads=1)
    with pytest.raises(UnsupportedModel, match="Gemma 4 style"):
        spec_from_config(cfg)


def test_config_with_more_layers_than_weights_is_refused(tmp_path):
    cfg = dict(SMALL, num_hidden_layers=4)
    t = llama_like_tensors(layers=2, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000)
    with pytest.raises(UnsupportedModel, match="have only 2"):
        load(LocalSource(write_model_dir(tmp_path / "m", cfg, t)))


def test_shape_fallback_uses_the_language_model_and_warns(tmp_path):
    lm = prefixed(llama_like_tensors(layers=3, hidden=64, heads=8, kv_heads=2, inter=128, vocab=1000),
                  "language_model.")
    vision = prefixed(llama_like_tensors(layers=6, hidden=32, heads=4, kv_heads=4, inter=64, vocab=10),
                      "vision_tower.")
    m = load(LocalSource(write_model_dir(tmp_path / "m", {"model_type": "mystery"}, lm | vision)))
    assert m.spec.source == "tensor shapes"
    assert (m.spec.num_layers, m.spec.head_dim) == (3, 16)    # 2 KV heads x head_dim 8
    p = plan(m.spec, weight_bytes_per_rank(m.spec, m.tensors, 1),
             ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=1024))
    assert any(c.message.startswith("UNCERTAIN: the attention geometry was derived from weight shapes")
               for c in p.checks)


# ---------------------------------------------------------------- encoder profiling

def test_qwen_vl_largest_item_from_preprocessor_config():
    s = spec_from_config(fixture("Qwen/Qwen2.5-VL-7B-Instruct"), None,
                         fixture("Qwen/Qwen2.5-VL-7B-Instruct", "preprocessor_config"))
    # max_pixels 12845056 / (patch 14 x merge 2)^2: the 16,384-token item vLLM 0.32 logged.
    assert s.encoder.max_item_tokens == 16384 and s.encoder.spatial_merge_size == 2


def test_encoder_term_follows_vllm_budget_rules():
    s = spec_from_config(fixture("Qwen/Qwen2.5-VL-7B-Instruct"), None,
                         fixture("Qwen/Qwen2.5-VL-7B-Instruct", "preprocessor_config"))
    extra, note = encoder_profile_bytes(s, tp=2, batched_tokens=8192, num_seqs=256, max_model_len=4096,
                                        text_activation=0)
    out = 16384 * 3584 * 2                                    # encoder outputs kept in the encoder cache
    peak = 2 * 16384 * 4 * 3420 * 2 * 2 // 2                  # 2 x gate/up over 65,536 positions, split over 2 GPUs
    assert extra == out + peak
    assert note.startswith("APPROXIMATE") and "1 item(s) of 16,384 tokens" in note
    # Without the preprocessor config the item size is unknown and the budget is assumed.
    s.encoder.max_item_tokens = None
    _, note = encoder_profile_bytes(s, 2, 8192, 256, 4096, 0)
    assert "could not be derived" in note


def test_text_only_models_get_no_encoder_term():
    s = spec_from_config(SMALL)
    assert encoder_profile_bytes(s, 1, 8192, 256, 2048, 0) == (0, "")


def test_encoder_term_is_part_of_the_plan():
    s = spec_from_config(fixture("Qwen/Qwen2.5-VL-7B-Instruct"), None,
                         fixture("Qwen/Qwen2.5-VL-7B-Instruct", "preprocessor_config"))
    cfg = ServeConfig(gpu_total_bytes=48 * GiB, tp=2, max_model_len=4096, max_num_batched_tokens=8192,
                      enforce_eager=True)
    p = plan(s, weight_bytes_per_rank(s, [], 2), cfg)
    assert p.overheads.encoder_bytes > 0
    assert p.overheads.total == (p.overheads.activation_bytes + p.overheads.non_torch_bytes
                                 + p.overheads.encoder_bytes)


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n
