import json

import pytest

from helpers import llama_like_tensors, write_model_dir
from kvplan.cli import main
from kvplan.model import spec_from_config
from kvplan.vllm import GiB, ServeConfig, batch_defaults, plan
from kvplan.weights import WeightBreakdown

LLAMA_8B = spec_from_config({
    "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
    "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 131072})


def weights(gib):
    return WeightBreakdown(int(gib * GiB), int(gib * GiB), 0, 0, 0, False, [])


def test_budget_is_fraction_of_total_and_formula_holds():
    cfg = ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=8192, overhead_bytes=GiB)
    p = plan(LLAMA_8B, weights(15), cfg)
    assert p.requested_bytes == int(-(-24 * GiB * 0.92 // 1))
    assert p.available_kv_bytes == p.requested_bytes - 15 * GiB - GiB
    assert p.starts and p.kv.fits_one_request


def test_default_context_too_long_fails_with_estimate():
    cfg = ServeConfig(gpu_total_bytes=24 * GiB, overhead_bytes=GiB)
    p = plan(LLAMA_8B, weights(15), cfg)
    assert not p.starts
    assert any("max_model_len=131072" in c.message for c in p.checks)
    assert any(s.startswith("--max-model-len") for s in p.suggestions)


def test_weights_larger_than_budget():
    p = plan(LLAMA_8B, weights(30), ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=4096))
    assert not p.starts
    assert any("weights need" in c.message for c in p.checks)
    assert not any("fp8" in s for s in p.suggestions)   # fp8 KV cannot fix a weights overflow


def test_tp_must_divide_heads():
    p = plan(LLAMA_8B, weights(4), ServeConfig(gpu_total_bytes=80 * GiB, tp=3, max_model_len=4096))
    assert not p.starts
    assert any("does not divide" in c.message for c in p.checks)
    assert not any("tensor-parallel-size 6" in s for s in p.suggestions)


def test_utilization_above_free_memory():
    cfg = ServeConfig(gpu_total_bytes=24 * GiB, gpu_memory_utilization=0.99, max_model_len=4096)
    p = plan(LLAMA_8B, weights(10), cfg)
    assert not p.starts
    assert any("Free memory on device" in c.message for c in p.checks)


def test_other_processes_count_against_free_memory():
    cfg = ServeConfig(gpu_total_bytes=24 * GiB, other_process_bytes=2 * GiB, max_model_len=4096)
    assert not plan(LLAMA_8B, weights(10), cfg).starts


def test_context_above_model_limit_warns():
    p = plan(LLAMA_8B, weights(15), ServeConfig(gpu_total_bytes=80 * GiB, max_model_len=262144,
                                                overhead_bytes=GiB))
    assert any(c.severity == "warning" for c in p.checks)


def test_batch_defaults():
    assert batch_defaults(24 * GiB) == (2048, 256)
    assert batch_defaults(79 * GiB, "H100-80GB") == (8192, 1024)
    assert batch_defaults(79 * GiB, "A100-80GB") == (2048, 256)
    assert batch_defaults(178 * GiB, "B200") == (16384, 1024)


def test_enforce_eager_removes_graph_estimate():
    cfg = ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=4096)
    eager = ServeConfig(gpu_total_bytes=24 * GiB, max_model_len=4096, enforce_eager=True)
    a, b = plan(LLAMA_8B, weights(10), cfg), plan(LLAMA_8B, weights(10), eager)
    assert b.available_kv_bytes > a.available_kv_bytes and b.overheads.cudagraph_bytes == 0


def gpt_oss_like():
    return spec_from_config({
        "model_type": "gpt_oss", "num_hidden_layers": 24, "hidden_size": 2880, "num_attention_heads": 64,
        "num_key_value_heads": 8, "head_dim": 64, "sliding_window": 128, "vocab_size": 201088,
        "max_position_embeddings": 131072})


def test_async_scheduling_doubles_in_flight_tokens_for_sliding_layers():
    spec = gpt_oss_like()
    base = dict(gpu_total_bytes=80 * GiB, max_model_len=131072, max_num_batched_tokens=8192, overhead_bytes=GiB)
    on = plan(spec, weights(13), ServeConfig(**base))
    off = plan(spec, weights(13), ServeConfig(**base, async_scheduling=False))
    # sliding blocks per request: cdiv(127 + 2*8192, 16) + 1 = 1033 vs cdiv(127 + 8192, 16) + 1 = 521
    full = 131072 // 16
    assert on.kv.max_concurrency == pytest.approx(on.kv.num_blocks / (full + 1033))
    assert off.kv.max_concurrency == pytest.approx(off.kv.num_blocks / (full + 521))


def test_mla_uses_backend_block_size():
    spec = spec_from_config({"model_type": "deepseek_v3", "num_hidden_layers": 61, "hidden_size": 7168,
                             "num_attention_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                             "vocab_size": 129280, "max_position_embeddings": 163840})
    cfg = dict(gpu_total_bytes=140 * GiB, tp=8, max_model_len=65536, overhead_bytes=GiB)
    assert plan(spec, weights(80), ServeConfig(**cfg, gpu_name="H200")).block_size == 64
    assert plan(spec, weights(80), ServeConfig(**cfg, gpu_name="B200")).block_size == 128
    assert plan(LLAMA_8B, weights(15), ServeConfig(gpu_total_bytes=80 * GiB, max_model_len=4096)).block_size == 16


def test_default_max_model_len_follows_longrope():
    spec = spec_from_config({"model_type": "phi3", "num_hidden_layers": 32, "hidden_size": 3072,
                             "num_attention_heads": 32, "max_position_embeddings": 131072,
                             "original_max_position_embeddings": 4096, "vocab_size": 32064,
                             "rope_scaling": {"type": "longrope", "short_factor": [1], "long_factor": [2]}})
    assert plan(spec, weights(7), ServeConfig(gpu_total_bytes=24 * GiB, overhead_bytes=GiB)).max_model_len == 4096


@pytest.mark.parametrize("bad", [{"tp": 0}, {"block_size": 0}, {"gpu_memory_utilization": 0},
                                 {"gpu_memory_utilization": 1.5}, {"max_model_len": 0}])
def test_invalid_inputs_raise_value_error(bad):
    with pytest.raises(ValueError):
        plan(LLAMA_8B, weights(10), ServeConfig(**({"gpu_total_bytes": 24 * GiB} | bad)))


def test_cli_rejects_zero_tp(tmp_path, capsys):
    d = write_model_dir(tmp_path / "m", {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64,
                                         "num_attention_heads": 8}, llama_like_tensors(2, 64, 8, 8, 128, 1000))
    assert main(["check", str(d), "--gpu", "L4", "-tp", "0"]) == 2
    assert "tensor-parallel-size" in capsys.readouterr().err


def test_cli_check_json(tmp_path, capsys):
    cfg = {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 8,
           "num_key_value_heads": 2, "vocab_size": 1000, "max_position_embeddings": 2048}
    d = write_model_dir(tmp_path / "m", cfg, llama_like_tensors(2, 64, 8, 2, 128, 1000))
    rc = main(["check", str(d), "--gpu", "RTX4090", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["starts"] and out["kv"]["max_concurrency"] > 1
    assert out["kv_bytes_per_token"] == 2 * 2 * 8 * 2 * 2


def test_cli_reports_unsupported_cleanly(tmp_path, capsys):
    cfg = {"model_type": "qwen3_next", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 8}
    d = write_model_dir(tmp_path / "m", cfg, llama_like_tensors(2, 64, 8, 2, 128, 1000))
    assert main(["check", str(d), "--gpu", "L4"]) == 2
    assert "not supported" in capsys.readouterr().err


def test_cli_unknown_gpu(tmp_path, capsys):
    d = write_model_dir(tmp_path / "m", {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64,
                                         "num_attention_heads": 8}, llama_like_tensors(2, 64, 8, 8, 128, 1000))
    assert main(["check", str(d), "--gpu", "TPUv9"]) == 2
    assert "unknown GPU" in capsys.readouterr().err


@pytest.mark.network
def test_real_model_matches_hand_calculation():
    from kvplan.loader import load
    from kvplan.sources import open_source
    from kvplan.weights import weight_bytes_per_rank
    m = load(open_source("Qwen/Qwen2.5-0.5B-Instruct"))
    assert (m.spec.num_layers, m.spec.num_kv_heads, m.spec.head_dim) == (24, 2, 64)
    w = weight_bytes_per_rank(m.spec, m.tensors)
    assert 0.9 * GiB < w.per_rank_bytes < 1.0 * GiB
