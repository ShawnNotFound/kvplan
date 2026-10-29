"""Runtime overheads, checked against vLLM 0.32 startup logs (docs/validation.md)."""

import pytest

from helpers import llama_like_tensors, write_model_dir
from kvplan.cli import main
from kvplan.model import spec_from_config
from kvplan.vllm import GiB, ServeConfig, plan
from kvplan.weights import WeightBreakdown

LLAMA31_8B = {"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
              "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
              "max_position_embeddings": 131072}
A6000 = 48541 * 2**20      # CUDA-visible totals
A5000 = 24114 * 2**20


def weights(n):
    return WeightBreakdown(n, n, 0, 0, 0, False, [])


@pytest.mark.parametrize("total, tp, w, eager, measured_tokens", [
    # NousResearch/Meta-Llama-3.1-8B-Instruct on vLLM 0.32.0 with LLM() defaults (8192 batched tokens).
    (A6000, 1, 16060522496, True, 217808),
    (A6000, 1, 16060522496, False, 212240),
    (A5000, 1, 16060522496, True, 42016),
    (A5000, 1, 16060522496, False, 36432),
    (A5000, 2, 8030527488, True, 212672),
    (A5000, 2, 8030527488, False, 206464),
])
def test_llama31_8b_matches_measured_kv_tokens(total, tp, w, eager, measured_tokens):
    cfg = ServeConfig(gpu_total_bytes=total, tp=tp, max_model_len=4096, gpu_memory_utilization=0.9,
                      max_num_batched_tokens=8192, enforce_eager=eager)
    p = plan(spec_from_config(LLAMA31_8B), weights(w), cfg)
    assert p.kv.kv_tokens == pytest.approx(measured_tokens, rel=0.01)


def test_eager_peak_is_mlp_or_sampler_whichever_is_larger():
    spec = spec_from_config(LLAMA31_8B)
    base = dict(gpu_total_bytes=A6000, max_model_len=4096, enforce_eager=True)
    big = plan(spec, weights(15 * GiB), ServeConfig(**base, max_num_batched_tokens=8192)).overheads
    assert big.activation_bytes == 8192 * 14336 * 2 * 2 * 2       # gate/up output x 2
    small = plan(spec, weights(15 * GiB), ServeConfig(**base, max_num_batched_tokens=1024)).overheads
    assert small.activation_bytes == 256 * 128256 * 4             # fp32 sampler logits
    assert big.compile_bytes == big.cudagraph_bytes == 0


def test_graph_mode_adds_compile_and_graph_memory():
    spec = spec_from_config(LLAMA31_8B)
    base = dict(gpu_total_bytes=A6000, max_model_len=4096, max_num_batched_tokens=8192)
    eager = plan(spec, weights(15 * GiB), ServeConfig(**base, enforce_eager=True)).overheads
    graphs = plan(spec, weights(15 * GiB), ServeConfig(**base)).overheads
    assert graphs.activation_bytes == eager.activation_bytes
    assert graphs.compile_bytes == 8192 * 4096 * 2 * 7 and graphs.cudagraph_bytes == GiB // 4
    tp2 = plan(spec, weights(8 * GiB), ServeConfig(**base, tp=2)).overheads
    assert tp2.compile_bytes == graphs.compile_bytes // 2


def test_moe_and_unknown_mlp_fall_back_and_say_so(tmp_path, capsys):
    moe = spec_from_config(LLAMA31_8B | {"model_type": "mixtral", "num_local_experts": 8})
    assert moe.intermediate_size is None
    cfg = ServeConfig(gpu_total_bytes=A6000, max_model_len=4096)
    assert not plan(moe, weights(15 * GiB), cfg).overheads.calibrated
    small = {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 8,
             "num_key_value_heads": 2, "vocab_size": 1000, "max_position_embeddings": 2048}
    tensors = llama_like_tensors(2, 64, 8, 2, 128, 1000)
    main(["check", str(write_model_dir(tmp_path / "a", small, tensors)), "--gpu", "A6000"])
    assert "rough estimate" in capsys.readouterr().out
    main(["check", str(write_model_dir(tmp_path / "b", small | {"intermediate_size": 128}, tensors)),
          "--gpu", "A6000"])
    assert "rough estimate" not in capsys.readouterr().out


@pytest.mark.parametrize("value", [None, 0, -5, [11008, 11008], "big", True])
def test_odd_intermediate_size_is_ignored(value):
    assert spec_from_config(LLAMA31_8B | {"intermediate_size": value}).intermediate_size is None
