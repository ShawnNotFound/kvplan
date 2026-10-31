"""Output formats and files, config files, model lists and the Python API, on local models."""

import csv
import json
import sys

import pytest

import kvplan
from helpers import llama_like_tensors, write_model_dir
from kvplan import cli, config, output

CFG = {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 8,
       "num_key_value_heads": 2, "vocab_size": 1000, "max_position_embeddings": 2048}


@pytest.fixture
def models(tmp_path):
    t = llama_like_tensors(2, 64, 8, 2, 128, 1000)
    return [str(write_model_dir(tmp_path / name, CFG, t)) for name in ("a", "b")]


def test_python_api_returns_plain_data(models):
    r = kvplan.check(models[0], gpu="A5000", max_model_len=1024)
    assert r.starts is True and r.kv_tokens > 0
    d = r.to_dict()
    json.dumps(d)                                   # everything is JSON-serialisable
    assert d["model"] == models[0] and d["starts"] is True
    row = r.row()
    assert row["verdict"] == "vLLM should start" and row["tp"] == 1 and row["max_model_len"] == 1024
    info = kvplan.inspect(models[0])
    assert info["kv_bytes_per_token"] > 0 and info["spec"]["num_layers"] == 2


def test_api_errors_are_kvplan_errors(models):
    with pytest.raises(kvplan.KvplanError):
        kvplan.check(models[0], gpu="NoSuchGPU")
    with pytest.raises(kvplan.KvplanError):
        kvplan.check(models[0], gpu_mem_gib=24, tp=0)


@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv", "markdown"])
def test_formats(models, capsys, fmt):
    assert cli.main(["check", models[0], "--gpu", "A5000", "--format", fmt]) == 0
    out = capsys.readouterr().out
    if fmt == "json":
        assert json.loads(out)["starts"] is True
    elif fmt == "jsonl":
        assert json.loads(out.splitlines()[0])["verdict"] == "vLLM should start"
    elif fmt == "csv":
        assert next(csv.DictReader(out.splitlines()))["starts"] == "True"
    else:
        assert out.startswith("| model | gpu |") and "vLLM should start" in out


def test_output_file_format_from_extension_and_directory(models, tmp_path):
    target = tmp_path / "res.csv"
    assert cli.main(["check", models[0], "--gpu", "A5000", "-o", str(target)]) == 0
    assert target.read_text().startswith("model,gpu,tp")
    folder = str(tmp_path / "out") + "/"   # a trailing slash means a directory, even a new one
    assert cli.main(["check", models[0], "--gpu", "A5000", "--format", "json", "-o", str(folder)]) == 0
    written = list((tmp_path / "out").iterdir())
    assert len(written) == 1 and written[0].suffix == ".json"


def test_append_csv_keeps_one_header(models, tmp_path):
    log = tmp_path / "log.csv"
    for gpu in ("A5000", "A6000"):
        assert cli.main(["check", models[0], "--gpu", gpu, "-o", str(log), "--append"]) == 0
    rows = list(csv.DictReader(log.open()))
    assert [r["gpu"] for r in rows] == ["A5000", "A6000"]
    with pytest.raises(ValueError, match="--append works with csv and jsonl"):
        output.write(str(tmp_path / "x.json"), "{}", "json", "x", append=True)


def test_model_list_csv_with_per_model_options(models, tmp_path, capsys):
    lst = tmp_path / "models.csv"
    lst.write_text(f"model,gpu,tp,max_model_len\n{models[0]},A5000,1,1024\n{models[1]},A6000,2,\n"
                   f"{tmp_path}/missing,A5000,,\n")
    code = cli.main(["check", "--models", str(lst), "--format", "json"])
    res = json.loads(capsys.readouterr().out)
    assert code == 2                                   # one entry could not be read
    assert [r.get("tp") for r in res[:2]] == [1, 2] and res[0]["max_model_len"] == 1024
    assert "error" in res[2]


def test_model_list_text_and_json(models, tmp_path):
    txt = tmp_path / "m.txt"
    txt.write_text(f"# my models\n{models[0]}\n\n{models[1]}  # second\n")
    assert [e["model"] for e in config.read_model_list(str(txt))] == models
    js = tmp_path / "m.json"
    js.write_text(json.dumps([models[0], {"model": models[1], "gpu-mem-gib": "24", "enforce_eager": "yes"}]))
    entries = config.read_model_list(str(js))
    assert entries[1] == {"model": models[1], "gpu_mem_gib": 24.0, "enforce_eager": True}
    js.write_text(json.dumps([{"model": models[0], "colour": "red"}]))
    with pytest.raises(ValueError, match="unknown option 'colour'"):
        config.read_model_list(str(js))


def test_model_and_models_together_is_an_error(models, tmp_path, capsys):
    lst = tmp_path / "m.txt"
    lst.write_text(models[0])
    assert cli.main(["check", models[0], "--models", str(lst), "--gpu", "A5000"]) == 2
    assert "not both" in capsys.readouterr().err


@pytest.mark.skipif(sys.version_info < (3, 11), reason="TOML needs Python 3.11")
def test_config_file_sets_defaults_and_command_line_wins(models, tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "kvplan.toml"
    cfg.write_text('gpu = "A5000"\nformat = "json"\n[check]\nmax_model_len = 1024\ntp = 2\n'
                   '[recommend]\ntask = "code"\n')
    assert cli.main(["--config", str(cfg), "check", models[0]]) == 0
    d = json.loads(capsys.readouterr().out)
    assert (d["tp"], d["max_model_len"], d["gpu"]) == (2, 1024, "A5000")
    assert cli.main(["--config", str(cfg), "check", models[0], "-tp", "1", "--gpu", "A6000"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert (d["tp"], d["gpu"]) == (1, "A6000")
    monkeypatch.setenv("KVPLAN_CONFIG", str(cfg))
    assert cli.main(["check", models[0]]) == 0
    assert json.loads(capsys.readouterr().out)["max_model_len"] == 1024


def test_config_file_errors(models, tmp_path, capsys):
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"check": {"colour": "red"}}))
    assert cli.main(["--config", str(bad), "check", models[0]]) == 2
    assert "unknown option(s) colour" in capsys.readouterr().err
    bad.write_text(json.dumps({"check": {"kv_cache_dtype": "int3"}}))
    assert cli.main(["--config", str(bad), "check", models[0]]) == 2
    assert "is not one of" in capsys.readouterr().err
    assert cli.main(["--config", str(tmp_path / "none.json"), "check", models[0]]) == 2
    assert "not found" in capsys.readouterr().err


GREEN, RED = "\033[32;1m", "\033[31;1m"


def test_colours_follow_the_verdict(models, capsys):
    assert cli.main(["--color", "always", "check", models[0], "--gpu", "A5000"]) == 0
    assert GREEN + "vLLM should start" in capsys.readouterr().out
    assert cli.main(["--color", "always", "check", models[0], "--gpu-mem-gib", "0.01"]) == 1
    out = capsys.readouterr().out
    assert RED + "vLLM will not start" in out and "\033[31merror" in out


def test_no_colour_in_pipes_files_or_data_formats(models, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert cli.main(["check", models[0], "--gpu", "A5000"]) == 0          # captured output is not a terminal
    assert "\033[" not in capsys.readouterr().out
    assert cli.main(["--color", "always", "check", models[0], "--gpu", "A5000", "--format", "csv"]) == 0
    assert "\033[" not in capsys.readouterr().out
    target = tmp_path / "r.txt"
    assert cli.main(["--color", "always", "check", models[0], "--gpu", "A5000", "-o", str(target)]) == 0
    assert "\033[" not in target.read_text()
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert cli.main(["check", models[0], "--gpu", "A5000"]) == 0
    assert "\033[" not in capsys.readouterr().out
