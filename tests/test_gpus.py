import json
import subprocess

import pytest

from helpers import llama_like_tensors, write_model_dir
from kvplan import cli
from kvplan.gpus import (GPUS_MIB, RESERVED_MIB, GPUDetectError, DetectedGPU, gpu_total_bytes, parse_nvidia_smi,
                         query_nvidia_smi, select_gpus)

MiB = 2**20


@pytest.mark.parametrize("name, mib", [
    # CUDA-visible totals: nvidia-smi memory.total minus memory.reserved, as read on the real GPUs.
    ("A5000", 24114), ("RTX A5000", 24114), ("NVIDIA RTX A5000", 24114),
    ("RTX5090-LAPTOP", 24137), ("NVIDIA GeForce RTX 5090 Laptop GPU", 24137), ("rtx5090m", 24137),
    ("A6000", 48541), ("NVIDIA RTX A6000", 48541),
    ("Tesla V100-PCIE-32GB", 32494), ("v100_32gb", 32494),
    # Not driver-verified: the nvidia-smi total.
    ("RTX4090", 24564), ("NVIDIA GeForce RTX 4090", 24564), ("V100-SXM2-16GB", 16384),
])
def test_presets_and_driver_names(name, mib):
    assert gpu_total_bytes(name) == mib * MiB


def test_a5000_and_4090_share_a_size_but_stay_separate():
    assert GPUS_MIB["A5000"] == GPUS_MIB["RTX4090"]
    assert gpu_total_bytes("A5000") != gpu_total_bytes("RTX4090")   # only the A5000's reservation is known


def test_reserved_entries_are_presets():
    assert set(RESERVED_MIB) <= set(GPUS_MIB)


def test_desktop_5090_is_not_confused_with_the_laptop_part():
    with pytest.raises(ValueError, match="unknown GPU 'RTX5090'"):
        gpu_total_bytes("RTX5090")


LAPTOP = "0, NVIDIA GeForce RTX 5090 Laptop GPU, 24463, 326, 4478\n"
SERVER = ("0, NVIDIA RTX A6000, 49140, 599, 1\n1, NVIDIA RTX A6000, 49140, 599, 20000\n"
          "2, NVIDIA RTX A5000, 24564, 450, 1\n3, NVIDIA RTX A5000, 24564, 450, 1\n")


def test_parse_handles_commas_in_names_and_missing_values():
    gpus = parse_nvidia_smi("0, Odd, Name GPU, 1000, [N/A], [N/A]\n\n1, NVIDIA GB10, [N/A], [N/A], [N/A]\n")
    assert gpus == [DetectedGPU(0, "Odd, Name GPU", 1000, None, None),
                    DetectedGPU(1, "NVIDIA GB10", None, None, None)]
    assert gpus[0].cuda_total_mib == 1000 and gpus[1].cuda_total_mib is None


def test_cuda_total_subtracts_reserved_memory():
    assert parse_nvidia_smi(SERVER)[0].cuda_total_mib == 48541   # CUDA reported 47.40 GiB on this GPU


@pytest.mark.parametrize("text", ["garbage", "x, A, 1, 2, 3", "0, A, 1, 2"])
def test_parse_rejects_unexpected_output(text):
    with pytest.raises(GPUDetectError, match="unexpected nvidia-smi output"):
        parse_nvidia_smi(text)


def fake_run(stdout="", returncode=0, stderr="", exc=None):
    def run(*a, **k):
        if exc:
            raise exc
        return subprocess.CompletedProcess(a, returncode, stdout, stderr)
    return run


@pytest.mark.parametrize("run, msg", [
    (fake_run(exc=FileNotFoundError()), "not on PATH"),
    (fake_run(exc=subprocess.TimeoutExpired("nvidia-smi", 20)), "did not answer"),
    (fake_run(returncode=9, stdout="NVIDIA-SMI has failed because it couldn't communicate with the driver"),
     "couldn't communicate"),
    (fake_run(returncode=6), "exit code 6"),
    (fake_run(stdout=""), "no GPUs"),
])
def test_query_failures_explain_what_to_do(run, msg):
    with pytest.raises(GPUDetectError, match=msg) as e:
        query_nvidia_smi(run)
    assert "--gpu-mem-gib" in str(e.value)


def test_select_defaults_to_first_tp_gpus_and_checks_indices():
    gpus = parse_nvidia_smi(SERVER)
    assert [g.index for g in select_gpus(gpus, 2)] == [0, 1]
    assert [g.index for g in select_gpus(gpus, 2, [2, 3])] == [2, 3]
    with pytest.raises(GPUDetectError, match="needs 8 GPUs"):
        select_gpus(gpus, 8)
    with pytest.raises(GPUDetectError, match="no GPU with index 7"):
        select_gpus(gpus, 1, [7])
    with pytest.raises(GPUDetectError, match="lists 1 GPU"):
        select_gpus(gpus, 2, [0])
    with pytest.raises(GPUDetectError, match="does not report the memory"):
        select_gpus(parse_nvidia_smi("0, NVIDIA GB10, [N/A], [N/A], [N/A]"), 1)


@pytest.fixture
def model_dir(tmp_path):
    cfg = {"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 8,
           "num_key_value_heads": 2, "vocab_size": 1000, "max_position_embeddings": 2048}
    return str(write_model_dir(tmp_path / "m", cfg, llama_like_tensors(2, 64, 8, 2, 128, 1000)))


def use_smi(monkeypatch, text):
    monkeypatch.setattr(cli, "query_nvidia_smi", lambda: parse_nvidia_smi(text))


def test_auto_counts_memory_other_processes_hold(monkeypatch, capsys, model_dir):
    # The laptop case: 0.92 x 24,137 MiB (CUDA's total) is more than what the desktop session leaves.
    use_smi(monkeypatch, LAPTOP)
    assert cli.main(["check", model_dir, "--gpu", "auto"]) == 1
    out = capsys.readouterr().out
    assert "4,478 MiB in use" in out and "Free memory on device" in out and "--gpu-memory-utilization to 0.79" in out
    assert cli.main(["check", model_dir, "--gpu", "auto", "--gpu-memory-utilization", "0.75"]) == 0


def test_auto_explicit_other_memory_wins(monkeypatch, capsys, model_dir):
    use_smi(monkeypatch, LAPTOP)
    assert cli.main(["check", model_dir, "--gpu", "auto", "--other-gpu-mem-gib", "0", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["other_process_bytes"] == 0 and out["detected_gpus"][0]["cuda_total_mib"] == 24137


def test_auto_tp_uses_smallest_and_busiest_gpu(monkeypatch, capsys, model_dir):
    use_smi(monkeypatch, SERVER)
    assert cli.main(["check", model_dir, "--gpu", "auto", "-tp", "2", "--json"]) == 1   # GPU 1 is busy
    out = json.loads(capsys.readouterr().out)
    assert out["gpu_total_bytes"] == 48541 * MiB and out["other_process_bytes"] == 20000 * MiB
    assert cli.main(["check", model_dir, "--gpu", "auto", "-tp", "2", "--gpu-index", "2,3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["gpu_total_bytes"] == 24114 * MiB


@pytest.mark.parametrize("argv, msg", [
    (["--gpu", "A6000", "--gpu-index", "0"], "only applies to --gpu auto"),
    (["--gpu", "auto", "--gpu-index", "a"], "comma-separated GPU numbers"),
    (["--gpu", "auto", "--gpu-index", "0,0"], "distinct GPU numbers"),
    (["--gpu", "auto", "-tp", "0"], "at least 1"),
    (["--gpu-mem-gib", "0"], "must be positive"),
    (["--gpu-mem-gib", "nan"], "must be a number"),
    (["--gpu", "L4", "--other-gpu-mem-gib", "-1"], "cannot be negative"),
    (["--gpu", "L4", "--other-gpu-mem-gib", "40"], "at least the GPU's total memory"),
    (["--gpu", "L4", "--overhead-gib", "-1"], "cannot be negative"),
])
def test_bad_gpu_arguments_fail_cleanly(monkeypatch, capsys, model_dir, argv, msg):
    use_smi(monkeypatch, SERVER)
    assert cli.main(["check", model_dir, *argv]) == 2
    assert msg in capsys.readouterr().err


def test_auto_without_driver_fails_cleanly(monkeypatch, capsys, model_dir):
    def missing():
        raise GPUDetectError("--gpu auto needs nvidia-smi, which is not on PATH; pass --gpu NAME or --gpu-mem-gib")
    monkeypatch.setattr(cli, "query_nvidia_smi", missing)
    assert cli.main(["check", model_dir, "--gpu", "auto"]) == 2
    assert "nvidia-smi" in capsys.readouterr().err
