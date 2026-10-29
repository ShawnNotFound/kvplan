# Validation against vLLM

kvplan's predictions next to what vLLM printed at startup on real GPUs.

## Setup (2026-10-29)

- vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, NVIDIA driver 595.91.07, Ubuntu 22.04.
- GPUs: RTX A6000 (48 GB) and RTX A5000 (24 GB), one model per GPU, nothing else running on it.
- Each run is a fresh process: `vllm.LLM(model=..., gpu_memory_utilization=0.9, max_model_len=4096, dtype="bfloat16")`, plus `enforce_eager=True` for the eager rows, then one 8-token generation. Weights came from a local Hugging Face cache, offline.
- `LLM()` defaults to 8192 batched tokens (vLLM's offline default), so kvplan was run with `--max-num-batched-tokens 8192`, or 2048 for the two rows that set it. kvplan was given the GPU's total memory as CUDA reports it (`--gpu-mem-gib`, the same value the `A6000` and `A5000` presets now use).
- Measured values come from vLLM's log lines: "Model loading took", "Actual usage is … consumed memory (weights + non-torch), … peak activation, … CUDAGraph memory", "Available KV cache memory" and "GPU KV cache size: N tokens, Maximum concurrency …". Runtime is everything between the weights and the KV cache: non-torch memory plus the profiled peak.
- Gemma 2 2B was loaded from `google/gemma-2-2b-it` locally; kvplan read the same files. Llama 3.2 3B likewise from `meta-llama/Llama-3.2-3B-Instruct`.

## Results

Error is kvplan's KV-token count relative to vLLM's. "v0.2" is the current model; "v0.1" is the first release, given the same exact GPU memory.

| GPU | Model | Mode | TP | Batched tokens | max_model_len | Weights GiB (vLLM / kvplan) | Runtime GiB (vLLM / kvplan) | KV cache GiB (vLLM / kvplan) | KV tokens (vLLM / kvplan) | Max concurrency (vLLM / kvplan) | Error v0.2 | Error v0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX A6000 | Qwen/Qwen2.5-1.5B-Instruct | CUDA graphs | 1 | 8192 | 4096 | 2.98 / 2.88 | 1.14 / 1.21 | 38.54 / 38.57 | 1,443,440 / 1,444,544 | 352.40 / 352.67 | +0.08% | -0.21% |
| RTX A6000 | Qwen/Qwen2.5-1.5B-Instruct | eager | 1 | 8192 | 4096 | 2.98 / 2.88 | 0.80 / 0.80 | 38.88 / 38.99 | 1,456,160 / 1,460,048 | 355.51 / 356.46 | +0.27% | +0.21% |
| RTX A6000 | google/gemma-2-2b-it | CUDA graphs | 1 | 8192 | 4096 | 4.94 / 4.87 | 1.90 / 1.31 | 35.82 / 36.48 | 360,479 / 367,106 | 88.01 / 89.63 | +1.84% | +0.86% |
| RTX A6000 | google/gemma-2-2b-it | eager | 1 | 8192 | 4096 | 4.94 / 4.87 | 0.99 / 0.81 | 36.74 / 36.98 | 369,701 / 372,097 | 90.26 / 90.84 | +0.65% | -0.30% |
| RTX A6000 | meta-llama/Llama-3.2-3B-Instruct | CUDA graphs | 1 | 8192 | 4096 | 6.04 / 5.98 | 1.40 / 1.33 | 35.22 / 35.35 | 329,728 / 330,928 | 80.50 / 80.79 | +0.36% | -0.29% |
| RTX A6000 | meta-llama/Llama-3.2-3B-Instruct | eager | 1 | 8192 | 4096 | 6.04 / 5.98 | 0.74 / 0.75 | 35.89 / 35.93 | 335,968 / 336,336 | 82.02 / 82.11 | +0.11% | -0.74% |
| RTX A6000 | NousResearch/Meta-Llama-3.1-8B-Instruct | CUDA graphs | 1 | 8192 | 4096 | 15.00 / 14.96 | 1.75 / 1.81 | 25.91 / 25.89 | 212,240 / 212,080 | 51.82 / 51.78 | -0.08% | +0.19% |
| RTX A6000 | NousResearch/Meta-Llama-3.1-8B-Instruct | eager | 1 | 8192 | 4096 | 15.00 / 14.96 | 1.07 / 1.12 | 26.59 / 26.58 | 217,808 / 217,712 | 53.18 / 53.15 | -0.04% | -0.49% |
| RTX A5000 | NousResearch/Meta-Llama-3.1-8B-Instruct | CUDA graphs | 1 | 8192 | 4096 | 15.00 / 14.96 | 1.75 / 1.81 | 4.45 / 4.42 | 36,432 / 36,240 | 8.89 / 8.85 | -0.53% | +1.01% |
| RTX A5000 | NousResearch/Meta-Llama-3.1-8B-Instruct | eager | 1 | 8192 | 4096 | 15.00 / 14.96 | 1.06 / 1.12 | 5.13 / 5.11 | 42,016 / 41,872 | 10.26 / 10.22 | -0.34% | -2.67% |
| 2× RTX A5000 | NousResearch/Meta-Llama-3.1-8B-Instruct | CUDA graphs | 2 | 8192 | 4096 | 7.54 / 7.48 | 1.05 / 1.16 | 12.60 / 12.56 | 206,464 / 205,776 | 50.41 / 50.24 | -0.33% | -2.03% |
| 2× RTX A5000 | NousResearch/Meta-Llama-3.1-8B-Instruct | eager | 2 | 8192 | 4096 | 7.54 / 7.48 | 0.68 / 0.69 | 12.98 / 13.03 | 212,672 / 213,456 | 51.92 / 52.11 | +0.37% | -1.04% |
| RTX A6000 | google/gemma-2-2b-it | eager | 1 | 2048 | 4096 | 4.94 / 4.87 | 0.65 / 0.49 | 37.08 / 37.30 | 373,127 / 375,298 | 91.10 / 91.63 | +0.58% | -0.36% |
| RTX A6000 | NousResearch/Meta-Llama-3.1-8B-Instruct | eager | 1 | 2048 | 4096 | 15.00 / 14.96 | 0.44 / 0.47 | 27.22 / 27.23 | 222,992 / 223,088 | 54.44 / 54.46 | +0.04% | -0.74% |
| RTX A6000 | google/gemma-2-2b-it | eager | 1 | 8192 | 8192 | 4.94 / 4.87 | 0.99 / 0.81 | 36.74 / 36.98 | 370,062 / 372,460 | 45.17 / 45.47 | +0.65% | -0.30% |

Summary: v0.2 is within 0.7% of vLLM's KV-token count in 14 of 15 runs and within 1.9% in all of them (mean absolute error 0.42%); v0.1 was within 2.7% (mean 0.76%).

**Read this with care: the v0.2 overhead terms were fitted to these same runs**, so the v0.2 column is an in-sample result, not an independent test. What it does show is that one small formula covers four dense architectures, both CUDA-graph and eager mode, two batch sizes and TP 1 and 2 at once. Runs on other GPUs, vLLM versions and model families are needed before quoting it as kvplan's accuracy.

## What the measurements changed

- **Total memory.** vLLM's total is `nvidia-smi`'s `memory.total` minus `memory.reserved` (A6000: 49140 − 599 MiB = 47.40 GiB; A5000: 24564 − 450 MiB = 23.55 GiB), not `memory.total`. With the plain total, kvplan's budget was about 0.5 GiB too large on an A6000 (around +1.5% KV tokens there).
- **Where the CUDA-graph memory goes.** The available KV memory vLLM reports is exactly `requested − consumed − peak`; with CUDA graphs on, the graph pool (0.20–0.24 GiB) and torch.compile's buffers are inside that peak. kvplan v0.1 charged 0.5 GiB for graphs and underestimated the compiled peak, and the two errors partly cancelled.
- **Eager peak.** The profiling pass peaks at the MLP's gate/up output (about 2 × batched tokens × intermediate size × 2 bytes per GPU) or at the sampler's fp32 logits (max_num_seqs × vocab × 4 bytes), whichever is larger. v0.1's formula, based on the hidden size, was up to 0.33 GiB high.

## Sliding-window layers (Gemma 2)

Gemma 2 alternates full-attention and 4096-token sliding-window layers. vLLM 0.32's hybrid KV cache manager puts them in two KV cache groups that draw from one block pool, and kvplan reproduces that grouping: at max_model_len 8192, where the window is shorter than a request, kvplan is 0.65% from vLLM, the same as at 4096. The "bytes per token" kvplan prints for Gemma 2 (106,496) counts every layer, so it is the cost of a token inside the window; the capacity and concurrency figures account for the window.

## Known gaps

- Gemma 2 with CUDA graphs: the compiled peak is 0.45 GiB above the fit (+1.8% KV tokens); its non-torch memory is also about 0.15 GiB above the 0.25 GiB kvplan assumes.
- Not measured: MoE models (kvplan falls back to the older estimate and says so), MLA models, quantised checkpoints, fp8 KV cache, Hopper or Blackwell GPUs, more than 2 GPUs, and `vllm serve` itself (its defaults differ only in batch size, which the 2048-token rows cover).
- Tesla V100: vLLM 0.32's CUDA 13 wheels do not run on compute capability 7.0 ("no kernel image is available for execution on the device"), so no V100 numbers.
- The CUDA context before vLLM's snapshot measured 0.21–0.46 GiB; kvplan keeps 0.5 GiB for the start check so that it errs towards "will not start".

## Reproducing

The scripts that produced this table (one `LLM()` per process, log parsing, and a matching `kvplan check` for each run) are small; they will be added to the repository as the measurement script in the roadmap.
