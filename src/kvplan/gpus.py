"""GPU memory sizes.

vLLM sizes everything from the total memory CUDA reports (cudaMemGetInfo), which is a
little below the marketing figure (ECC and driver reservations). The values here are the
"memory.total" figures `nvidia-smi` prints, in MiB. They can differ from what CUDA
reports by a few hundred MiB, so for exact results pass --gpu-mem-gib with the value from
`python -c "import torch; print(torch.cuda.mem_get_info()[1] / 2**30)"` on the real GPU.
"""

from __future__ import annotations

GPUS_MIB = {
    "H200": 143771,
    "H100-80GB": 81559,
    "H100-94GB": 95830,
    "A100-80GB": 81920,
    "A100-40GB": 40960,
    "L40S": 46068,
    "L40": 46068,
    "A6000": 49140,
    "RTX6000-ADA": 49140,
    "L4": 23034,
    "A10G": 23028,
    "A10": 23028,
    "RTX4090": 24564,
    "RTX3090": 24576,
    "RTX4080": 16376,
    "T4": 15360,
    "V100-32GB": 32768,
    "V100-16GB": 16384,
}


def gpu_total_bytes(name: str) -> int:
    key = name.upper().replace(" ", "").replace("_", "-")
    for k, mib in GPUS_MIB.items():
        if k.replace("-", "") == key.replace("-", ""):
            return mib * 2**20
    raise ValueError(f"unknown GPU '{name}'. Known: {', '.join(GPUS_MIB)}; or pass --gpu-mem-gib")
