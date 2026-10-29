"""GPU memory sizes.

vLLM sizes everything from the total memory CUDA reports (cudaMemGetInfo), which is a
little below the marketing figure (ECC and driver reservations). The values here are the
"memory.total" figures `nvidia-smi` prints, in MiB. They can differ from what CUDA
reports by a few hundred MiB, so for exact results pass --gpu-mem-gib with the value from
`python -c "import torch; print(torch.cuda.mem_get_info()[1] / 2**30)"` on the real GPU.

Presets are looked up by name only, so two GPUs with the same size (the RTX A5000 and the
RTX 4090 both report 24564 MiB) are separate entries and never mistaken for each other.

`--gpu auto` reads the local GPUs from `nvidia-smi` instead, including the memory other
processes already hold (a desktop session on a laptop GPU, another job on a shared server).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

GPUS_MIB = {
    "H200": 143771,
    "H100-80GB": 81559,
    "H100-94GB": 95830,
    "A100-80GB": 81920,
    "A100-40GB": 40960,
    "L40S": 46068,
    "L40": 46068,
    "A6000": 49140,         # driver-verified (RTX A6000, driver 595.91)
    "RTX6000-ADA": 49140,
    "A5000": 24564,         # driver-verified (RTX A5000, driver 595.91)
    "L4": 23034,
    "A10G": 23028,
    "A10": 23028,
    "RTX4090": 24564,
    "RTX3090": 24576,
    "RTX5090-LAPTOP": 24463,  # driver-verified (GeForce RTX 5090 Laptop GPU, driver 621.01)
    "RTX4080": 16376,
    "T4": 15360,
    "V100-32GB": 32768,     # driver-verified (Tesla V100-PCIE-32GB, driver 580.159)
    "V100-16GB": 16384,
}

# Other spellings, after normalisation (see _norm), mapped to a preset.
_ALIASES = {
    "RTXA6000": "A6000",
    "RTXA5000": "A5000",
    "RTX6000ADAGENERATION": "RTX6000-ADA",
    "RTX5090LAPTOP": "RTX5090-LAPTOP",
    "RTX5090M": "RTX5090-LAPTOP",
    "V100PCIE32GB": "V100-32GB",
    "V100SXM232GB": "V100-32GB",
    "V100PCIE16GB": "V100-16GB",
    "V100SXM216GB": "V100-16GB",
}

NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                    "--format=csv,noheader,nounits"]


class GPUDetectError(ValueError):
    """`--gpu auto` could not read the local GPUs."""


def _norm(name: str) -> str:
    key = re.sub(r"[\s_\-]", "", name.upper())
    for prefix in ("NVIDIA", "GEFORCE", "TESLA", "QUADRO"):
        key = key.removeprefix(prefix)
    return key.removesuffix("GPU")


def gpu_total_bytes(name: str) -> int:
    key = _norm(name)
    key = _ALIASES.get(key, key)
    for k, mib in GPUS_MIB.items():
        if _norm(k) == _norm(key):
            return mib * 2**20
    raise ValueError(f"unknown GPU '{name}'. Known: {', '.join(GPUS_MIB)}, or 'auto' to read the local GPU "
                     "with nvidia-smi; for any other GPU pass --gpu-mem-gib with its total memory")


@dataclass(frozen=True)
class DetectedGPU:
    index: int
    name: str
    total_mib: int | None   # None when nvidia-smi prints N/A (e.g. unified-memory systems)
    used_mib: int | None


def _mib(field: str) -> int | None:
    field = field.strip()
    try:
        return int(float(field))
    except ValueError:
        return None   # "[N/A]", "[Not Supported]", ...


def parse_nvidia_smi(text: str) -> list[DetectedGPU]:
    """Parse `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 4:
            raise GPUDetectError(f"unexpected nvidia-smi output line: {line.strip()!r}; pass --gpu NAME or "
                                 "--gpu-mem-gib instead")
        # The name may itself contain commas; index is first, the two memory fields are last.
        try:
            index = int(parts[0].strip())
        except ValueError:
            raise GPUDetectError(f"unexpected nvidia-smi output line: {line.strip()!r}; pass --gpu NAME or "
                                 "--gpu-mem-gib instead") from None
        gpus.append(DetectedGPU(index, ",".join(parts[1:-2]).strip(), _mib(parts[-2]), _mib(parts[-1])))
    return gpus


def query_nvidia_smi(run=subprocess.run) -> list[DetectedGPU]:
    hint = "pass --gpu NAME or --gpu-mem-gib instead"
    try:
        res = run(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        raise GPUDetectError(f"--gpu auto needs nvidia-smi, which is not on PATH (no NVIDIA driver here?); "
                             f"{hint}") from None
    except subprocess.TimeoutExpired:
        raise GPUDetectError(f"nvidia-smi did not answer within 20 s; {hint}") from None
    except OSError as e:
        raise GPUDetectError(f"could not run nvidia-smi ({e}); {hint}") from None
    if res.returncode != 0:
        msg = (res.stderr or res.stdout or "").strip().splitlines()
        raise GPUDetectError(f"nvidia-smi failed: {msg[0] if msg else f'exit code {res.returncode}'}; {hint}")
    gpus = parse_nvidia_smi(res.stdout)
    if not gpus:
        raise GPUDetectError(f"nvidia-smi found no GPUs; {hint}")
    return gpus


def select_gpus(gpus: list[DetectedGPU], tp: int, indices: list[int] | None = None) -> list[DetectedGPU]:
    """The GPUs vLLM would use: the given nvidia-smi indices, or the first `tp` GPUs."""
    if tp < 1:
        raise ValueError("--tensor-parallel-size must be at least 1")
    by_index = {g.index: g for g in gpus}
    if indices is None:
        if len(gpus) < tp:
            raise GPUDetectError(f"tensor_parallel_size={tp} needs {tp} GPUs, but nvidia-smi lists {len(gpus)}")
        chosen = gpus[:tp]
    else:
        missing = [i for i in indices if i not in by_index]
        if missing:
            raise GPUDetectError(f"no GPU with index {', '.join(map(str, missing))}; nvidia-smi lists "
                                 + ", ".join(f"{g.index} ({g.name})" for g in gpus))
        if len(indices) != tp:
            raise GPUDetectError(f"--gpu-index lists {len(indices)} GPU(s) but tensor_parallel_size is {tp}")
        chosen = [by_index[i] for i in indices]
    for g in chosen:
        if g.total_mib is None or g.total_mib <= 0:
            raise GPUDetectError(f"nvidia-smi does not report the memory of GPU {g.index} ({g.name}); "
                                 "pass --gpu-mem-gib with its total memory")
    return chosen
