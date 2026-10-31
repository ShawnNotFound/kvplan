"""Search, name suggestions, login, download selection and recommendations, without the network."""

import io
import urllib.error

import pytest

from kvplan import cli, hub, recommender, sources
from kvplan.hub import HubModel
from kvplan.model import spec_from_config
from kvplan.vllm import GiB, ServeConfig
from kvplan.weights_meta import Tensor

LLAMA_CFG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "num_hidden_layers": 32,
             "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
             "max_position_embeddings": 131072, "intermediate_size": 14336}


def m(mid, downloads=1000, params=8e9, arch="LlamaForCausalLM", gated=False, dtype="BF16"):
    return HubModel(id=mid, downloads=downloads, gated=gated, architecture=arch,
                    parameters={dtype: int(params)} if params else {})


CATALOG = [
    m("meta-llama/Llama-3.1-8B-Instruct", 4_600_000, gated=True),
    m("unsloth/Llama-3.1-8B-Instruct", 212_000),
    m("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", 370_000, params=0),
    m("Qwen/Qwen2.5-7B-Instruct", 5_400_000, 7.6e9, "Qwen2ForCausalLM"),
    m("Qwen/Qwen2.5-0.5B-Instruct", 5_100_000, 0.49e9, "Qwen2ForCausalLM"),
    m("Qwen/Qwen2.5-72B-Instruct", 900_000, 72.7e9, "Qwen2ForCausalLM"),
    m("Qwen/Qwen2.5-Coder-7B-Instruct", 800_000, 7.6e9, "Qwen2ForCausalLM"),
    m("someone/Exotic-7B", 50_000, 7e9, "ExoticForCausalLM"),
]


def fake_list(search=None, pipeline_tag=None, limit=50, token=None, **_):
    hits = [x for x in CATALOG if search is None or search.lower() in x.id.lower()]
    return sorted(hits, key=lambda x: -x.downloads)[:limit]


@pytest.fixture
def offline_hub(monkeypatch):
    monkeypatch.setattr(hub, "list_models", fake_list)


def test_search_matches_loose_names(offline_hub):
    ids = [x.id for x in hub.search_models("llama 3.1 8b instruct")]
    assert ids[0] == "meta-llama/Llama-3.1-8B-Instruct" and "unsloth/Llama-3.1-8B-Instruct" in ids
    assert all("qwen" not in i.lower() for i in ids)


def test_bare_name_resolves_to_most_downloaded_exact_match(offline_hub):
    found, _ = hub.resolve_name("Llama-3.1-8B-Instruct")
    assert found == "meta-llama/Llama-3.1-8B-Instruct"
    found, matches = hub.resolve_name("llama-3.1-8b")
    assert found is None and matches


def test_misspelled_repo_gets_close_suggestions(offline_hub):
    ids = [x.id for x in hub.similar_models("Qwen/Qwen2.5-7B-Instrct")]
    assert ids[0] == "Qwen/Qwen2.5-7B-Instruct"
    assert "someone/Exotic-7B" not in ids


def test_missing_repo_is_not_reported_as_gated(monkeypatch):
    def fail(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO())
    monkeypatch.setattr(sources._opener, "open", fail)
    with pytest.raises(sources.RepoNotFoundError, match="not found on Hugging Face"):
        sources.HubSource("Qwen/Nope").list_files()


def test_unknown_name_prints_suggestions(offline_hub, capsys, monkeypatch):
    monkeypatch.setattr(hub, "similar_models", lambda *a, **k: CATALOG[:2])
    monkeypatch.setattr(hub, "search_models", lambda *a, **k: [])
    assert cli.main(["inspect", "no-such-model-name"]) == 2
    err = capsys.readouterr().err
    assert "no Hugging Face repo named 'no-such-model-name'" in err and "unsloth/Llama-3.1-8B-Instruct" in err


def test_save_token_is_private_and_found(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "hf" / "token"))
    path = hub.save_token("hf_abc\n")
    assert path.read_text() == "hf_abc" and (path.stat().st_mode & 0o777) == 0o600
    assert sources.find_token() == "hf_abc"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert sources.find_token() == "hf_env"


def test_download_picks_weights_and_support_files(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def no_hf_hub(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_hf_hub)
    fetched = []
    monkeypatch.setattr(hub, "_fetch", lambda url, dest, *a: fetched.append(dest.name))
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md",
             "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.bin",
             "original/consolidated.pth", "model-Q4_K_M.gguf"]
    where = hub.download("o/m", files[5:7], files, local_dir=str(tmp_path), log=lambda s: None)
    assert where == str(tmp_path)
    assert sorted(fetched) == sorted(["config.json", "tokenizer.json", "tokenizer_config.json",
                                      "chat_template.jinja", "model-00001-of-00002.safetensors",
                                      "model-00002-of-00002.safetensors"])


def test_no_gpu_given_and_none_found_explains(monkeypatch, capsys):
    def no_smi(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr("kvplan.gpus.subprocess.run", no_smi)
    assert cli.main(["check", "o/m"]) == 2
    err = capsys.readouterr().err
    assert "no --gpu given" in err and "--gpu A6000" in err


# recommendations

def cfg_for(total_gib):
    return lambda tp: ServeConfig(gpu_total_bytes=int(total_gib * GiB), tp=tp)


def fake_loader(mid):
    """A Llama-shaped model whose weights match the catalog's parameter count."""
    hm = next(x for x in CATALOG if x.id == mid)
    spec = spec_from_config(dict(LLAMA_CFG, architectures=[hm.architecture]))
    nbytes = hm.checkpoint_bytes
    tensors = [Tensor("model.layers.0.mlp.up_proj.weight", "BF16", (nbytes // 4, 2), nbytes)]
    return type("Loaded", (), {"spec": spec, "tensors": tensors, "files": [], "repo_files": []})()


def test_candidates_filter_unsupported_gated_and_oversized():
    pool, skipped = recommender.candidates("chat", token=None, include_gated=False, budget_bytes=40 * GiB,
                                         max_tp=1, lister=fake_list)
    ids = {x.id for x in pool}
    assert "Qwen/Qwen2.5-72B-Instruct" not in ids          # 145 GB of weights on one 40 GiB budget
    assert "meta-llama/Llama-3.1-8B-Instruct" not in ids   # gated, no token
    assert "someone/Exotic-7B" not in ids                  # no native vLLM implementation
    assert "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" not in ids
    assert "Qwen/Qwen2.5-0.5B-Instruct" not in ids         # under 0.5B parameters
    assert skipped["gated (no token)"] == 1 and skipped["not a native vLLM architecture"] == 1


def test_code_task_keeps_code_models():
    pool, _ = recommender.candidates("code", token=None, include_gated=False, budget_bytes=40 * GiB, max_tp=1,
                                   lister=fake_list)
    assert [x.id for x in pool] == ["Qwen/Qwen2.5-Coder-7B-Instruct"]


def test_priority_orders_by_size():
    pool = [m("a/small", 10, 1e9), m("b/big", 10, 30e9), m("c/mid", 10, 7e9)]
    assert [x.id for x in recommender.order_candidates(pool, "quality", 80 * GiB)] == ["b/big", "c/mid", "a/small"]
    assert [x.id for x in recommender.order_candidates(pool, "speed", 80 * GiB)] == ["a/small", "c/mid", "b/big"]


def test_recommend_uses_more_gpus_only_when_needed():
    res = recommender.recommend("chat", "quality", cfg_for(48), [1, 2, 4], max_model_len=8192, token=None,
                              lister=fake_list, loader=fake_loader)
    by_id = {r.model.id: r for r in res.recommendations}
    assert by_id["Qwen/Qwen2.5-7B-Instruct"].tp == 1
    assert by_id["Qwen/Qwen2.5-72B-Instruct"].tp == 4       # 145 GB of weights need four 48 GiB GPUs
    assert all(r.starts for r in res.recommendations)
    sizes = [r.weights_bytes * r.tp for r in res.recommendations]
    assert sizes == sorted(sizes, reverse=True)


def test_recommend_drops_models_that_do_not_start():
    res = recommender.recommend("chat", "speed", cfg_for(20), [1], max_model_len=8192, token=None,
                              lister=fake_list, loader=fake_loader)
    assert {r.model.id for r in res.recommendations} == {"Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct",
                                                         "unsloth/Llama-3.1-8B-Instruct"}


def test_image_generation_is_explained(capsys):
    assert cli.main(["recommend", "--gpu", "A6000", "--task", "image-generation", "--no-input"]) == 2
    assert "not served by vLLM" in capsys.readouterr().err
