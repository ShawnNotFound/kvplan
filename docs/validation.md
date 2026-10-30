# Validation against vLLM

kvplan's predictions next to what vLLM printed at startup on real GPUs.

## Setup (2026-10-29)

- vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, NVIDIA driver 595.91.07, Ubuntu 22.04.
- GPUs: RTX A6000 (48 GB) and RTX A5000 (24 GB), one model per GPU, nothing else running on it.
- Each run is a fresh process: `vllm.LLM(model=..., gpu_memory_utilization=0.9, max_model_len=4096, dtype="bfloat16")`, plus `enforce_eager=True` for the eager rows, then one 8-token generation. Weights came from a local Hugging Face cache, offline.
- `LLM()` defaults to 8192 batched tokens (vLLM's offline default), so kvplan was run with `--max-num-batched-tokens 8192`, or 2048 for the two rows that set it. kvplan was given the GPU's total memory as CUDA reports it (`--gpu-mem-gib`, the same value the `A6000` and `A5000` presets now use).
- Measured values come from vLLM's log lines: "Model loading took", "Actual usage is … consumed memory (weights + non-torch), … peak activation, … CUDAGraph memory", "Available KV cache memory" and "GPU KV cache size: N tokens, Maximum concurrency …". Runtime is everything between the weights and the KV cache: non-torch memory plus the profiled peak.
- Gemma 2 2B was loaded from `google/gemma-2-2b-it` locally; kvplan read the same files. Llama 3.2 3B likewise from `meta-llama/Llama-3.2-3B-Instruct`.

## In-sample runs (used for calibration)

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

## Out-of-sample runs: 2× RTX A6000 (2026-10-30)

These runs were not used for calibration; kvplan's constants are the ones fitted above, unchanged. Same setup as above (vLLM 0.32.0, `gpu_memory_utilization=0.9`, `max_model_len=4096`, bf16, 8192 batched tokens), tensor parallelism 2 across the two A6000s. Values are per GPU.

| GPU | Model | Mode | TP | Batched tokens | max_model_len | Weights GiB (vLLM / kvplan) | Runtime GiB (vLLM / kvplan) | KV cache GiB (vLLM / kvplan) | KV tokens (vLLM / kvplan) | Max concurrency (vLLM / kvplan) | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2× RTX A6000 | NousResearch/Meta-Llama-3.1-8B-Instruct | CUDA graphs | 2 | 8192 | 4096 | 7.54 / 7.48 | 1.06 / 1.16 | 34.07 / 34.02 | 558,192 / 557,456 | 136.28 / 136.10 | -0.13% |
| 2× RTX A6000 | NousResearch/Meta-Llama-3.1-8B-Instruct | eager | 2 | 8192 | 4096 | 7.54 / 7.48 | 0.68 / 0.69 | 34.45 / 34.49 | 564,432 / 565,136 | 137.80 / 137.97 | +0.12% |
| 2× RTX A6000 | Qwen/Qwen2.5-VL-7B-Instruct | CUDA graphs | 2 | 8192 | 4096 | 7.90 / 7.72 | 2.39 / 1.27 | 32.37 / 33.67 | 1,212,368 / 1,255,894 | 295.99 / 306.61 | +3.59% |
| 2× RTX A6000 | Qwen/Qwen2.5-VL-7B-Instruct | eager | 2 | 8192 | 4096 | 7.90 / 7.72 | 1.68 / 0.83 | 33.08 / 34.11 | 1,238,848 / 1,272,357 | 302.45 / 310.63 | +2.70% |
| 2× RTX A6000 | Qwen/Qwen2.5-Omni-7B | eager | 2 | 8192 | 4096 | 9.01 / 10.56 | 1.11 / 0.87 | 32.54 / 31.23 | 1,218,704 / 204,688 | 297.54 / 49.97 | -83.2% |

What they show:

- **Dense model at TP 2 on a new GPU (Llama 3.1 8B):** within 0.13% in both modes, in line with the in-sample runs.
- **Vision-language model (Qwen2.5-VL 7B):** kvplan overestimates the KV cache by 2.7–3.6%. vLLM profiles the vision encoder with one maximum-size video item (an encoder budget of 16,384 tokens), which raises the peak to 1.33 GiB (eager) and 2.04 GiB (CUDA graphs) against kvplan's text-only 0.58 and 1.02 GiB. vLLM's weights are also 0.18 GiB above kvplan's; one likely cause is that part of the vision tower is not split across GPUs, as kvplan assumes it is. kvplan does not model encoder profiling yet.
- **Qwen2.5-Omni 7B is wrong:** its config nests the language model under `thinker_config.text_config`, which kvplan does not read, so it falls back to weight shapes and gets the KV size per token wrong (163,840 bytes instead of 28,672). Its weight estimate is 1.55 GiB per GPU too high, most likely because it counts the talker and token2wav weights, which vLLM's text-output model does not load. kvplan already flags this case as a rough estimate; a fix needs support for the nested config.

## Re-run after fix: multimodal models on 2× RTX A6000 (2026-10-30)

After the out-of-sample runs above, kvplan was changed to read nested text configs, skip the checkpoint parts vLLM does not load, hold Qwen2.5-Omni's audio tower whole on every GPU, and add an approximate term for vLLM's multimodal encoder profiling (docs/accounting.md, sections 0, 3 and 8). These rows are **not** out-of-sample: the fixes were written after seeing the runs above, so they are labelled as re-runs after the fix. The overhead constants were not re-fitted, and the encoder term was not fitted to these runs.

vLLM was run again with the same settings; it reported exactly the same numbers as before. The original rows above are unchanged.

| GPU | Model | Mode | TP | Batched tokens | max_model_len | Weights GiB (vLLM / kvplan) | Runtime GiB (vLLM / kvplan) | KV cache GiB (vLLM / kvplan) | KV tokens (vLLM / kvplan) | Max concurrency (vLLM / kvplan) | Error before fix | Error after fix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2× RTX A6000 | Qwen/Qwen2.5-VL-7B-Instruct | CUDA graphs | 2 | 8192 | 4096 | 7.90 / 7.72 | 2.39 / 1.64 | 32.37 / 33.30 | 1,212,368 / 1,242,219 | 295.99 / 303.28 | +3.59% | +2.46% |
| 2× RTX A6000 | Qwen/Qwen2.5-VL-7B-Instruct | eager | 2 | 8192 | 4096 | 7.90 / 7.72 | 1.68 / 1.19 | 33.08 / 33.74 | 1,238,848 / 1,258,699 | 302.45 / 307.30 | +2.70% | +1.60% |
| 2× RTX A6000 | Qwen/Qwen2.5-Omni-7B | eager | 2 | 8192 | 4096 | 9.01 / 8.91 | 1.11 / 1.19 | 32.54 / 32.55 | 1,218,704 / 1,214,248 | 297.54 / 296.45 | -83.2% | -0.37% |

- **Qwen2.5-Omni 7B:** kvplan now reads `thinker_config.text_config` (28,672 KV bytes per token per GPU, as vLLM), skips 4.19 GiB of talker and token2wav weights, and keeps the audio tower whole on each GPU. It is within 0.4% of vLLM.
- **Qwen2.5-VL 7B:** the encoder term adds 0.37 GiB, which removes 31–41% of the error. vLLM's measured peak is still 0.5–0.75 GiB above kvplan's, and its weights are 0.18 GiB above (explained in the next section: the M-RoPE cache). kvplan prints a warning that the multimodal figures are approximate.
- The encoder term is not a good physical model yet: the same vision encoder and the same 16,384-token item give a 1.33 GiB peak in Qwen2.5-VL but 0.66 GiB in Qwen2.5-Omni (eager). It is labelled approximate for that reason.
- All 15 in-sample text-model runs give exactly the same kvplan numbers after the fix.

## Multimodal profiling, measured (2026-10-30, second round)

Goal: replace the approximate encoder term with one derived from vLLM's code, and check it on runs that were not used to build it.

**Method.** 34 vLLM 0.32 startups of cached multimodal models on the same machine (RTX A6000 and RTX A5000, `gpu_memory_utilization` 0.9, bf16, 8192 batched tokens, `max_model_len` 4096 unless stated). They vary `max_pixels` (default 12,845,056, 1,003,520 and 3,211,264), `--limit-mm-per-prompt` (default, image only, no video), `max_model_len` (4096 and 8192), TP 1 and 2, eager and CUDA graphs, and the GPU. A small instrumentation module loaded into every vLLM process (`sitecustomize`, no change to vLLM's behaviour) recorded torch's allocated and peak memory before and after the encoder pass inside `profile_run`, the dummy items vLLM built, and, for one run, every parameter and buffer on rank 0.

**Split, fixed before modelling.** Fit set: the A6000 TP 1 runs of Qwen2.5-VL-7B, Qwen2.5-Omni-7B, Qwen2.5-Omni-3B and Qwen2-Audio-7B. Holdout: every A5000 run, every TP 2 run, and every run of the MiniCPM-o 2.6 and Voxtral-Mini-3B families. The fit set was used to check the derivations in docs/accounting.md (sections 3 and 8) and to set one constant (below); no holdout run influenced the model.

Not runnable: Voxtral-Mini-3B (vLLM 0.32 fails to load it from this cache: `VoxtralEncoderConfig` has no `downsample_factor`, also with `tokenizer_mode mistral`), ultravox-v0.5 (needs Llama-3.1-8B-Instruct weights that are not cached) and VITA-1.5 (weights not cached).

**What was found.**

- Every dummy batch kvplan predicts for the Qwen family matches what vLLM built: modality, item count, tokens and encoder positions (16 configurations).
- The Qwen2.5-VL vision tower's peak is linear in encoder positions: 29.9 KB per position at TP 1 and 22.2 KB at TP 2 over 30,720–65,536 positions. The live-tensor derivation (accounting.md, section 8) gives 29.4 and 21.7 KB.
- Qwen2.5-Omni profiles a 729-token video, not the 16,384 tokens vLLM budgets for, because its HF processor resizes video frames; without video it profiles a 16,384-token image and the peak goes from 0.1 to 1.9 GiB.
- With CUDA graphs on, the encoder pass is unchanged in kind (it runs eagerly) and vLLM's reported peak adds its CUDA-graph estimate.
- **The 0.18 GiB weight gap on Qwen2.5-VL at TP 2:** parameters match kvplan exactly (7.724 GiB). vLLM also allocates the language model's M-RoPE cos/sin cache, a [512000, 128] bf16 buffer (4 × max_position_embeddings × head_dim): 0.122 GiB. kvplan now counts RoPE caches with the weights. The remaining 0.05 GiB is other memory still allocated at the end of loading; not modelled.
- **Non-torch memory is higher for multimodal models:** 0.39–0.57 GiB on the fit runs (median 0.45) against 0.22–0.46 for text models. kvplan could not trace it to a vLLM allocation, so it adds a fitted 0.20 GiB for models with an encoder (median of the fit set minus the 0.25 GiB text value).

Errors are kvplan's KV-token count against vLLM's. "Profiled peak" is vLLM's "peak activation" (including its CUDA-graph estimate) next to kvplan's activation + torch.compile + encoder + CUDA-graph terms.

### Fit set (A6000, TP 1)

| GPU | Model | Mode | Settings | Profiled items (vLLM) | Encoder peak GiB (measured) | Profiled peak GiB (vLLM / kvplan) | KV tokens (vLLM / kvplan) | Error |
|---|---|---|---|---|---|---|---|---|
| RTX A6000 | Qwen2.5-VL-7B | eager | defaults | 1 video × 16,384 | 1.82 | 1.80 / 1.79 | 463,952 / 463,485 | -0.10% |
| RTX A6000 | Qwen2.5-VL-7B | eager | max_pixels 1,003,520 | 2 video × 3,840 | 0.86 | 1.10 / 1.21 | 477,568 / 474,402 | -0.66% |
| RTX A6000 | Qwen2.5-VL-7B | eager | max_pixels 3,211,264 | 2 video × 4,096 | 0.91 | 1.09 / 1.21 | 477,792 / 474,339 | -0.72% |
| RTX A6000 | Qwen2.5-VL-7B | eager | limit image 1, video 0 | 1 image × 16,384 | 1.82 | 1.80 / 1.79 | 463,952 / 463,485 | -0.10% |
| RTX A6000 | Qwen2.5-VL-7B | eager | max_model_len 8192 | 1 video × 16,384 | 1.82 | 1.80 / 1.79 | 463,952 / 464,388 | +0.09% |
| RTX A6000 | Qwen2.5-VL-7B | eager | max_pixels 1,003,520; max_model_len 8192 | 1 video × 7,680 | 0.86 | 1.10 / 1.21 | 477,568 / 475,327 | -0.47% |
| RTX A6000 | Qwen2-Audio-7B | eager | defaults | 10 audio × 750 | 0.50 | 0.78 / 0.73 | 52,800 / 52,912 | +0.21% |
| RTX A6000 | Qwen2.5-VL-7B | CUDA graphs | defaults | 1 video × 16,384 | 2.14 | 2.53 / 2.43 | 450,320 / 451,675 | +0.30% |
| RTX A6000 | Qwen2.5-Omni-7B | eager | defaults | 1 video × 729 | 0.09 | 1.04 / 1.16 | 455,680 / 454,735 | -0.21% |
| RTX A6000 | Qwen2.5-Omni-7B | eager | limit video 0 | 1 image × 16,384 | 1.89 | 1.88 / 1.86 | 439,568 / 441,618 | +0.47% |
| RTX A6000 | Qwen2.5-Omni-7B | eager | max_pixels 1,003,520 | 2 video × 3,840 | 0.90 | 1.10 / 1.21 | 454,528 / 453,875 | -0.14% |
| RTX A6000 | Qwen2.5-Omni-3B | eager | defaults | 1 video × 729 | 0.09 | 0.61 / 0.67 | 951,472 / 950,001 | -0.15% |
| RTX A6000 | Qwen2.5-Omni-3B | eager | limit video 0 | 1 image × 16,384 | 1.89 | 1.88 / 1.86 | 913,936 / 915,463 | +0.17% |
| RTX A6000 | Qwen2.5-Omni-3B | eager | max_model_len 8192 | 1 video × 729 | 0.09 | 0.61 / 0.67 | 951,472 / 951,852 | +0.04% |
| RTX A6000 | Qwen2-Audio-7B | eager | max_model_len 8192 | 10 audio × 750 | 0.50 | 0.78 / 0.73 | 52,800 / 52,912 | +0.21% |
| RTX A6000 | Qwen2.5-Omni-3B | CUDA graphs | defaults | 1 video × 729 | 0.11 | 1.04 / 1.14 | 938,848 / 936,390 | -0.26% |

Within 0.72% in all 16 runs (mean absolute error 0.27%).

### Holdout

| GPU | Model | Mode | Settings | Profiled items (vLLM) | Encoder peak GiB (measured) | Profiled peak GiB (vLLM / kvplan) | KV tokens (vLLM / kvplan) | Error |
|---|---|---|---|---|---|---|---|---|
| RTX A6000 | MiniCPM-o-2.6 | eager | defaults | 8 video × 1,024 | 5.95 | 5.94 / 1.21 | 384,128 / 471,900 | +22.85% |
| RTX A6000 | MiniCPM-o-2.6 | eager | limit image 1, audio 0, video 0 | 12 image × 640 | 5.81 | 5.75 / 1.21 | 398,048 / 471,900 | +18.55% |
| RTX A5000 | Qwen2.5-VL-7B | eager | defaults | 1 video × 16,384 | 1.82 | 1.80 / 1.79 | 62,144 / 63,113 | +1.56% |
| RTX A5000 | Qwen2.5-VL-7B | eager | max_pixels 1,003,520 | 2 video × 3,840 | 0.86 | 1.10 / 1.21 | 75,744 / 74,046 | -2.24% |
| RTX A5000 | Qwen2.5-Omni-3B | eager | defaults | 1 video × 729 | 0.09 | 0.61 / 0.67 | 326,416 / 327,217 | +0.25% |
| RTX A5000 | Qwen2.5-Omni-3B | eager | limit video 0 | 1 image × 16,384 | 1.89 | 1.88 / 1.86 | 288,896 / 292,680 | +1.31% |
| RTX A5000 | Qwen2-Audio-7B | eager | defaults | 10 audio × 750 | 0.50 | 0.78 / 0.73 | 8,864 / 8,944 | +0.90% |
| RTX A5000 | MiniCPM-o-2.6 | eager | defaults | 8 video × 1,024 | 5.95 | – / 1.21 | vLLM: no memory for the KV cache / kvplan: 71,544 | kvplan says it starts |
| RTX A5000 | Qwen2.5-Omni-3B | eager | max_pixels 1,003,520 | 2 video × 3,840 | 0.90 | 0.89 / 0.87 | 318,320 / 321,432 | +0.98% |
| RTX A5000 | Qwen2.5-VL-7B | eager | limit image 1, video 0 | 1 image × 16,384 | 1.82 | 1.80 / 1.79 | 62,144 / 63,113 | +1.56% |
| RTX A5000 | MiniCPM-o-2.6 | eager | max_model_len 8192 | 8 video × 1,024 | 5.95 | – / 1.21 | vLLM: no memory for the KV cache / kvplan: 71,683 | kvplan says it starts |
| 2× RTX A6000 | Qwen2.5-Omni-3B | eager | defaults | 1 video × 729 | 0.07 | 0.38 / 0.34 | 2,142,448 / 2,139,179 | -0.15% |
| 2× RTX A6000 | Qwen2.5-VL-7B | eager | max_pixels 1,003,520 | 2 video × 3,840 | 0.66 | 0.72 / 0.63 | 1,262,560 / 1,258,428 | -0.33% |
| 2× RTX A5000 | Qwen2.5-VL-7B | eager | defaults | 1 video × 16,384 | 1.35 | 1.33 / 1.32 | 434,848 / 431,833 | -0.69% |
| 2× RTX A5000 | Qwen2-Audio-7B | eager | defaults | 10 audio × 750 | 0.50 | 0.60 / 0.39 | 48,208 / 48,880 | +1.39% |

- **Qwen family and Qwen2-Audio on held-out GPUs and TP:** within 2.24% in all 11 runs (mean absolute error 1.03%). The A5000 rows have small KV pools (3–11 GiB), so a 0.05 GiB error is 1–2% there.
- **Qwen2-Audio at TP 2:** kvplan's audio-encoder rule is 25% low, and at TP 2 the encoder sets the peak; +1.39%.
- **MiniCPM-o 2.6 is not modelled:** vLLM profiles 8 videos (or 12 images) and the encoder pass peaks at 5.8–5.9 GiB; kvplan's generic estimate is 1.2 GiB, so it is 19–23% high on the A6000 and says the model starts on an A5000 where vLLM runs out of memory. kvplan now gives the verdict "start not confirmed" (exit code 3) for every encoder it cannot size. For this run it says vLLM starts only if the profiling peak stays below 4.83 GiB; vLLM measured 5.94 GiB. With the image-only limit vLLM also skips loading MiniCPM-o's audio tower (0.6 GiB less weight); kvplan does not model that.
- **Bug found by the holdout:** Qwen2-Audio's `text_config` names only `model_type: qwen2`, so kvplan fell back to weight shapes and, at TP 2, kept the whole KV width on each GPU (-49.7%). kvplan now fills in transformers' `Qwen2Config` defaults (32 KV heads) and, when it does fall back to weight shapes, splits the KV width across GPUs. The table shows the fixed result.

### Effect on other models

Checked on the 204 ungated models of the top-300 Hugging Face text-generation list, 2 × H100-80GB, TP 2, from local copies of their configs and safetensors headers. No KV size per token or start/no-start result changed. The RoPE cache adds a median of 0.01 GiB per GPU (most 0.03 GiB or less, at most 0.375 GiB for a 1M-context model with two caches). Three multimodal models also get the encoder terms: Qwen2.5-VL-7B NVFP4 (-0.70 GiB of KV), and Qwen3-VL-30B-A3B AWQ and Kimi-K2.5 NVFP4 (-0.42 and -0.23 GiB, both flagged UNCERTAIN). 64 models fail as before (unsupported architectures).

With the "start not confirmed" verdict, 3 of the 140 models that kvplan sizes move from "should start" to "start not confirmed": Qwen3-VL-30B-A3B AWQ (vision encoder not sized), and GPT-Neo 125M and MiniCPM-SALA AWQ (no native vLLM implementation). The other 123 "should start" and 14 "will not start" verdicts are unchanged.

### Earlier runs with the current kvplan

The RoPE cache and the non-torch change also move the earlier rows. In-sample text runs: within -1.23% to +1.83% (mean absolute error 0.51%; was 0.42%): the A5000 Llama rows move from -0.5/-0.3% to -1.2/-1.0%, the rest by less than 0.3 points. The 2× A6000 runs: Llama 3.1 8B -0.22% / +0.03%, Qwen2.5-VL 7B +0.31% (graphs) / -0.51% (eager), Qwen2.5-Omni 7B +0.03%.

## MoE and quantised checkpoints (2026-10-30, third round)

Twelve runs of four small models downloaded for this (7.6 GB in total): IBM Granite 3.1 1B-A400M (MoE, 32 experts), Qwen2.5-1.5B-Instruct AWQ and GPTQ-Int4, and Qwen3-1.7B-FP8 (block-quantised fp8 weights, which vLLM runs through Marlin on Ampere). vLLM 0.32.0, `LLM()` with 8192 batched tokens, max_model_len 4096 unless stated, CUDA graphs unless stated. None of these runs were used to fit anything. Results with the current kvplan:

| GPU | Model | Mode | TP | Weights GiB (vLLM / kvplan) | KV cache GiB (vLLM / kvplan) | KV tokens (vLLM / kvplan) | Error |
|---|---|---|---|---|---|---|---|
| RTX A6000 | ibm-granite/granite-3.1-1b-a400m-instruct | CUDA graphs | 1 | 2.50 / 2.50 | 39.32 / 39.27 | 858,864 / 857,808 | -0.12% |
| RTX A6000 | ibm-granite/granite-3.1-1b-a400m-instruct (max_model_len 8192) | eager | 1 | 2.50 / 2.50 | 39.66 / 39.63 | 866,288 / 865,664 | -0.07% |
| RTX A5000 | ibm-granite/granite-3.1-1b-a400m-instruct | CUDA graphs | 1 | 2.50 / 2.50 | 17.85 / 17.80 | 390,032 / 388,896 | -0.29% |
| 2× RTX A6000 | ibm-granite/granite-3.1-1b-a400m-instruct | CUDA graphs | 2 | 1.26 / 1.26 | 40.65 / 40.66 | 1,775,808 / 1,776,384 | +0.03% |
| RTX A6000 | Qwen/Qwen2.5-1.5B-Instruct-AWQ | CUDA graphs | 1 | 1.10 / 1.08 | 40.47 / 40.37 | 1,515,424 / 1,511,904 | -0.23% |
| RTX A5000 | Qwen/Qwen2.5-1.5B-Instruct-AWQ | CUDA graphs | 1 | 1.10 / 1.08 | 19.01 / 18.91 | 711,776 / 708,048 | -0.52% |
| 2× RTX A6000 | Qwen/Qwen2.5-1.5B-Instruct-AWQ | CUDA graphs | 2 | 0.55 / 0.54 | 41.32 / 41.26 | 3,094,912 / 3,090,448 | -0.14% |
| RTX A6000 | Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 | CUDA graphs | 1 | 1.10 / 1.08 | 40.45 / 40.37 | 1,514,976 / 1,511,824 | -0.21% |
| RTX A5000 | Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 | CUDA graphs | 1 | 1.10 / 1.08 | 19.00 / 18.91 | 711,344 / 707,984 | -0.47% |
| RTX A6000 | Qwen/Qwen3-1.7B-FP8 | CUDA graphs | 1 | 1.92 / 1.90 | 39.48 / 39.66 | 369,600 / 371,344 | +0.47% |
| RTX A5000 | Qwen/Qwen3-1.7B-FP8 | CUDA graphs | 1 | 1.92 / 1.90 | 18.02 / 18.20 | 168,688 / 170,384 | +1.01% |
| RTX A5000 | Qwen/Qwen3-1.7B-FP8 | eager | 1 | 1.92 / 1.90 | 18.59 / 18.67 | 174,080 / 174,768 | +0.40% |

All twelve are within 1.01% (mean absolute error 0.33%).

- **Bug found:** the AWQ and FP8 checkpoints store `lm_head.weight` although `tie_word_embeddings` is set. vLLM ties `lm_head` to the embedding and does not load the stored copy; kvplan counted it, so its weights were up to 0.58 GiB high and the KV figure up to 2.8% low (AWQ on the A5000). The table shows the fixed result. On the ungated models of today's top-300 Hugging Face text-generation list (2 × H100-80GB, TP 2; 175 of 287 sized, the rest unsupported architectures or not reachable), the fix changes 8 models, all by 0.15–0.37 GiB less weight per GPU and at most 0.5% more KV tokens: Qwen3 0.6B and 1.7B (also the base bf16 checkpoints store `lm_head.weight`) and two fine-tunes of them, Qwen3Guard-Gen 0.6B, Qwen3-4B-Instruct-2507-FP8, Qwen2.5-3B-Instruct-AWQ and Llama-3.2-3B-Instruct-FP8-dynamic. No start/no-start verdict changes.
- The runtime terms do not match one by one: the profiled peak is 0.43–0.44 GiB for the MoE model with CUDA graphs (kvplan 0.28 GiB) but 0.09 GiB in eager mode, and 0.83–0.98 GiB for the quantised models with CUDA graphs (kvplan 0.38–0.55 GiB), while kvplan's CUDA-graph term is a little high. The totals agree to within 1%, so the fit is unchanged, and kvplan still notes that its runtime figures for MoE models are a rough estimate.
- **fp8 KV cache on Ampere:** six runs with `kv_cache_dtype="fp8"` (Qwen2.5 1.5B, Gemma 2 2B, Llama 3.2 3B, Llama 3.1 8B, Granite MoE; A6000 and A5000) all failed at startup. The default FlashInfer backend could not build its fp8 kernels for compute capability 8.6, and the Triton backend refuses: "FP8 KV cache is not supported by the Triton attention backend on NVIDIA RTX A6000 (compute capability 8.6); native FP8 (fp8e4nv) requires SM89+". kvplan now gives "start not confirmed" for an fp8 KV cache on a GPU it knows to be older than compute capability 8.9, and no longer suggests `--kv-cache-dtype fp8` there. The KV size of an fp8 cache is still unmeasured.

## What the measurements changed

- **Total memory.** vLLM's total is `nvidia-smi`'s `memory.total` minus `memory.reserved` (A6000: 49140 − 599 MiB = 47.40 GiB; A5000: 24564 − 450 MiB = 23.55 GiB), not `memory.total`. With the plain total, kvplan's budget was about 0.5 GiB too large on an A6000 (around +1.5% KV tokens there).
- **Where the CUDA-graph memory goes.** The available KV memory vLLM reports is exactly `requested − consumed − peak`; with CUDA graphs on, the graph pool (0.20–0.24 GiB) and torch.compile's buffers are inside that peak. kvplan v0.1 charged 0.5 GiB for graphs and underestimated the compiled peak, and the two errors partly cancelled.
- **Eager peak.** The profiling pass peaks at the MLP's gate/up output (about 2 × batched tokens × intermediate size × 2 bytes per GPU) or at the sampler's fp32 logits (max_num_seqs × vocab × 4 bytes), whichever is larger. v0.1's formula, based on the hidden size, was up to 0.33 GiB high.

## Sliding-window layers (Gemma 2)

Gemma 2 alternates full-attention and 4096-token sliding-window layers. vLLM 0.32's hybrid KV cache manager puts them in two KV cache groups that draw from one block pool, and kvplan reproduces that grouping: at max_model_len 8192, where the window is shorter than a request, kvplan is 0.65% from vLLM, the same as at 4096. The "bytes per token" kvplan prints for Gemma 2 (106,496) counts every layer, so it is the cost of a token inside the window; the capacity and concurrency figures account for the window.

## Known gaps

- Gemma 2 with CUDA graphs: the compiled peak is 0.45 GiB above the fit (+1.8% KV tokens); its non-torch memory is also about 0.15 GiB above the 0.25 GiB kvplan assumes.
- Multimodal models: the encoder pass is derived for the Qwen2.5-VL / Qwen2.5-Omni vision tower only; Whisper-style audio is approximate (25% low) and other encoders (MiniCPM-o, Gemma 3, Llava, InternVL, ...) are not sized: kvplan warns. The extra 0.20 GiB of non-torch memory for multimodal models is a fitted constant.
- Not measured: MoE models larger than Granite 1B-A400M, MLA models, quantised checkpoints other than AWQ, GPTQ and fp8 block quantisation at 1.5–1.7B, the size of an fp8 KV cache (it does not start on the Ampere GPUs used here), Hopper or Blackwell GPUs, more than 2 GPUs, and `vllm serve` itself (its defaults differ only in batch size, which the 2048-token rows cover).
- Tesla V100: vLLM 0.32's CUDA 13 wheels do not run on compute capability 7.0 ("no kernel image is available for execution on the device"), so no V100 numbers.
- The CUDA context before vLLM's snapshot measured 0.21–0.46 GiB; kvplan keeps 0.5 GiB for the start check so that it errs towards "will not start".

## Reproducing

The scripts that produced this table (one `LLM()` per process, log parsing, and a matching `kvplan check` for each run) are small; they will be added to the repository as the measurement script in the roadmap.
