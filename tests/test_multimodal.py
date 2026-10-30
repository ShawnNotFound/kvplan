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
from kvplan import mm
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
    # text_config names only model_type qwen2: Qwen2Config's defaults apply (32 KV heads, as vLLM uses).
    ("Qwen/Qwen2-Audio-7B-Instruct", "text_config", (32, 32, 32, 128, 4096)),
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
    assert m.spec.num_layers == 3
    assert m.spec.num_kv_heads * m.spec.head_dim == 16           # 2 KV heads x head_dim 8
    # Unknown head counts: the KV width splits across GPUs instead of being copied to each.
    from kvplan.kv import kv_layout
    per_gpu = [sum(l.page_bytes for l in kv_layout(m.spec, tp).layers) for tp in (1, 2)]
    assert per_gpu[1] * 2 == per_gpu[0]
    p = plan(m.spec, weight_bytes_per_rank(m.spec, m.tensors, 1),
             ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=1024))
    assert any(c.message.startswith("UNCERTAIN: the attention geometry was derived from weight shapes")
               for c in p.checks)


# ---------------------------------------------------------------- encoder profiling

def vl_spec(repo="Qwen/Qwen2.5-VL-7B-Instruct"):
    return spec_from_config(fixture(repo), None, fixture(repo, "preprocessor_config"))


# (max_pixels override, limits, max_model_len) -> (modality, items, tokens budgeted, tokens produced,
# encoder positions per item), as vLLM 0.32 logged them on RTX A6000 / A5000 (docs/validation.md).
@pytest.mark.parametrize("repo, max_pixels, limits, max_len, expected", [
    ("Qwen/Qwen2.5-VL-7B-Instruct", None, {}, 4096, ("video", 1, 16384, 16384, 65536)),
    ("Qwen/Qwen2.5-VL-7B-Instruct", 1003520, {}, 4096, ("video", 2, 3840, 3840, 15360)),
    ("Qwen/Qwen2.5-VL-7B-Instruct", 3211264, {}, 4096, ("video", 2, 4096, 4096, 16384)),
    ("Qwen/Qwen2.5-VL-7B-Instruct", 1003520, {}, 8192, ("video", 1, 7680, 7680, 30720)),
    ("Qwen/Qwen2.5-VL-7B-Instruct", None, {"image": 1, "video": 0}, 4096, ("image", 1, 16384, 16384, 65536)),
    # The Omni HF processor resizes video to 768 x 28 x 28 pixels per frame pair: 729 tokens, not 16,384.
    ("Qwen/Qwen2.5-Omni-7B", None, {}, 4096, ("video", 1, 16384, 729, 2916)),
    ("Qwen/Qwen2.5-Omni-7B", None, {"video": 0}, 4096, ("image", 1, 16384, 16384, 65536)),
    ("Qwen/Qwen2.5-Omni-7B", 1003520, {}, 4096, ("video", 2, 3840, 3840, 15360)),
])
def test_qwen_vl_dummy_items_match_vllm(repo, max_pixels, limits, max_len, expected):
    s = vl_spec(repo)
    p = mm.profile(s.encoder, s.hidden_size, 1, 8192, 256, max_len, limits, max_pixels)
    assert (p.modality, p.items, p.item_tokens, p.item_outputs, p.item_positions) == expected
    assert not p.approximate


def test_qwen_vit_bytes_per_position():
    # Qwen2.5-VL ViT: h 1280, i 3420. 3h + 3i (+ h at tp > 1) in bf16 plus the uint8 pixels.
    s = vl_spec()
    assert mm.bytes_per_position(s.encoder, 1) == (3 * 1280 + 3 * 3420) * 2 + 1176
    assert mm.bytes_per_position(s.encoder, 2) == (4 * 1280 + 3 * 3420 // 2) * 2 + 1176


def test_profiled_peak_is_encoder_or_outputs_plus_text_pass():
    s = vl_spec()
    cfg = ServeConfig(gpu_total_bytes=48 * GiB, max_model_len=4096, max_num_batched_tokens=8192, enforce_eager=True)
    # Large item: the encoder pass sets the peak.
    extra, note, approx = encoder_profile_bytes(s, cfg, 8192, 256, 4096, text_activation=GiB)
    assert extra == 65536 * mm.bytes_per_position(s.encoder, 1) - GiB and not approx
    assert "1 video item(s) of 16,384 tokens" in note
    # Small items: the text pass with the encoder outputs held sets it.
    cfg.mm_max_pixels = 1003520
    extra, _, _ = encoder_profile_bytes(s, cfg, 8192, 256, 4096, text_activation=GiB)
    assert extra == 2 * 3840 * 3584 * 2
    cfg.skip_mm_profiling = True
    assert encoder_profile_bytes(s, cfg, 8192, 256, 4096, GiB) == (0, "", False)


def test_whisper_audio_items():
    s = spec_from_config(fixture("Qwen/Qwen2-Audio-7B-Instruct"))
    assert s.encoder.family == "whisper"
    p = mm.profile(s.encoder, s.hidden_size, 1, 8192, 256, 4096)
    # vLLM 0.32 logged 10 audio items of 750 tokens (budget 8192).
    assert (p.modality, p.items, p.item_outputs, p.item_positions) == ("audio", 10, 750, 1500)
    assert p.approximate


def test_rotary_cache_counted_with_the_weights():
    s = vl_spec()
    # M-RoPE: 4 x 128,000 positions x 128 dims in bf16, as vLLM 0.32 allocated (0.122 GiB).
    assert s.rotary_cache_bytes == 512_000 * 128 * 2
    w = weight_bytes_per_rank(s, [], 1)
    assert w.per_rank_bytes == w.buffer_bytes == s.rotary_cache_bytes


def test_generic_encoders_are_flagged_uncertain():
    s = spec_from_config(fixture("unsloth/gemma-3-4b-it"))
    p = plan(s, weight_bytes_per_rank(s, [], 1), ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=4096))
    assert any(c.message.startswith("UNCERTAIN: kvplan cannot size vLLM's startup profiling") for c in p.checks)


def test_unknown_encoders_are_approximate():
    s = spec_from_config(fixture("unsloth/gemma-3-4b-it"))
    cfg = ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=4096)
    extra, note, approx = encoder_profile_bytes(s, cfg, 2048, 256, 4096, text_activation=0)
    assert approx and note.startswith("APPROXIMATE") and "could not be derived" in note and extra > 0


def test_text_only_models_get_no_encoder_term():
    s = spec_from_config(SMALL)
    cfg = ServeConfig(gpu_total_bytes=24 * GiB)
    assert encoder_profile_bytes(s, cfg, 8192, 256, 2048, 0) == (0, "", False)


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
