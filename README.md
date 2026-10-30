# kvplan

Plan vLLM deployments without the GPU.

Tell kvplan a model, your GPUs and your vLLM settings. It tells you whether vLLM will start, how much KV cache you will get, and how many full-length requests fit at once. It reads the model's real files from Hugging Face (only the headers, never the weights), so it works for new models the day they are uploaded.

```
$ kvplan check unsloth/Meta-Llama-3.1-8B-Instruct --gpu L4

unsloth/Meta-Llama-3.1-8B-Instruct on 1 x L4 (max_model_len 131072, max_num_seqs 256, gpu_memory_utilization 0.92)

  budget         20.69 GiB of 22.49 GiB per GPU
  weights        14.96 GiB per GPU (exact, from the weight files)
  runtime        0.83 GiB estimated: activation peak 0.22 GiB, torch.compile 0.11 GiB, CUDA graphs 0.25 GiB, other 0.25 GiB
  KV cache       4.91 GiB available (131,072 bytes per token per GPU, auto, block size 16)
  capacity       40,208 tokens; max concurrency at 131,072 tokens per request: 0.31x

  RESULT         vLLM will not start
  error          one request at max_model_len=131072 needs 16.00 GiB of KV cache, but only 4.91 GiB is available; vLLM's estimate of the longest length that fits is 40192
  try            --max-model-len 40192 (or lower)
  try            --kv-cache-dtype fp8 halves the KV cache
  try            --tensor-parallel-size 2 on 2 GPUs splits weights and KV further
```

> **Status: alpha.** Weight and KV sizes are computed exactly from the model files. The runtime overheads (activation peak, torch.compile buffers, CUDA graphs) are fitted to vLLM 0.32 startup logs from 15 runs on RTX A6000 and A5000 GPUs; see [Accuracy](https://github.com/ShawnNotFound/kvplan#accuracy) for what that does and does not cover.

## Why

vLLM decides its KV cache size at startup, on the real GPU: it reserves `gpu_memory_utilization` of the **total** GPU memory, loads the weights, profiles a forward pass, and gives whatever is left to the KV cache. If one request at `max_model_len` doesn't fit, it refuses to start. You find this out after renting or buying the GPU.

Other tools approximate this with parameter count × bytes per parameter plus a fixed overhead. That misses quantisation scales, padded vocabularies, tied embeddings, sliding-window layers, latent (MLA) caches and how tensor parallelism splits or replicates KV heads. kvplan reproduces vLLM's own accounting instead. See [docs/accounting.md](https://github.com/ShawnNotFound/kvplan/blob/master/docs/accounting.md) for every rule and the vLLM source it comes from.

## Install

```
pip install kvplan
```

or the latest development version: `pip install git+https://github.com/ShawnNotFound/kvplan`.

No dependencies beyond Python 3.10+. For gated models (Llama, Gemma), set `HF_TOKEN` or run `huggingface-cli login`.

## Use

```
kvplan check MODEL (--gpu NAME | --gpu-mem-gib GIB) [vLLM options]
kvplan inspect MODEL
```

`MODEL` is a Hugging Face repo id or a local directory or file. `check` understands the vLLM options that change memory: `-tp/--tensor-parallel-size`, `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--block-size`, `--enforce-eager`. Add `--json` for scripts and CI; the exit code is 0 when vLLM should start and 1 when it won't.

For the most precise result, give the GPU memory exactly as CUDA reports it (`--gpu-mem-gib`); the `--gpu` presets use the slightly larger `nvidia-smi` totals.

On the machine that will run vLLM, `--gpu auto` reads the GPUs from `nvidia-smi`, including memory other processes already hold. vLLM reserves a fraction of *total* memory, so a desktop session on a laptop GPU or another job on a shared server can stop it from starting even when the model fits:

```
$ kvplan check Qwen/Qwen2.5-1.5B-Instruct --gpu auto
...
  gpu 0          NVIDIA GeForce RTX 5090 Laptop GPU: 24,463 MiB total, 4,478 MiB in use (nvidia-smi)
  RESULT         vLLM will not start
  error          gpu_memory_utilization=0.92 asks for 21.98 GiB, but only about 19.02 GiB is free after the CUDA context and other processes. ...
  try            lower --gpu-memory-utilization to 0.79
```

With tensor parallelism it uses the first `-tp` GPUs, or the ones you name with `--gpu-index 2,3`.

```
$ kvplan inspect openai/gpt-oss-20b
openai/gpt-oss-20b
  architecture   gpt_oss (read from config)
  layers         24: 12 sliding, 12 full (window 128)
  attention      64 heads, 8 KV heads, head_dim 64
  experts        32 (all resident in GPU memory)
  context        131072
  KV per token   49,152 bytes per GPU at tp=1 (auto)
  weights        12.82 GiB per GPU at tp=1 (checkpoint 12.82 GiB in 3 file(s))
```

## What it covers

| | |
|---|---|
| Model metadata | config.json, including nested language-model configs (`text_config`, `thinker_config`, `llm_config`, `language_config`) resolved as vLLM does; safetensors headers; GGUF headers. A tensor-shape fallback when config.json lacks fields (flagged as uncertain), and a refusal when config and weights disagree |
| Multimodal | checkpoint parts vLLM does not load (Omni speech generators) are skipped; an approximate term for vLLM's startup encoder profiling, flagged in the output |
| Attention | MHA, GQA, MQA, sliding-window and hybrid full/sliding layouts (Gemma 2/3, gpt-oss, Cohere2, Qwen2/3 with sliding window), MLA (DeepSeek V2/V3, Kimi K2) |
| Weights | exact bytes per tensor, 16-bit loading of fp32 checkpoints, quantised formats at their stored size, vocab padding, tied embeddings, MTP layers excluded, tensor-parallel sharding and KV-head replication |
| vLLM | utilisation budget, free-memory check, KV block grouping and rounding, max concurrency, the max_model_len check and its suggested length |

In a sample of the 300 most-downloaded text-generation models on Hugging Face, kvplan's sources describe 93% without a token and 98% with one (details in [docs/coverage.md](https://github.com/ShawnNotFound/kvplan/blob/master/docs/coverage.md)).

**Not yet supported:** models with Mamba or linear-attention layers (Qwen3-Next, Jamba, Nemotron-H, Falcon-H1), DeepSeek V3.2 sparse attention, Gemma 4 (full-attention layers with their own head size), encoder-decoder models, pipeline/expert/data parallelism, LoRA, speculative decoding, and llama.cpp. kvplan says so instead of guessing.

## Accuracy

Checked against vLLM 0.32.0 on an RTX A6000 and RTX A5000s with four dense models (Qwen2.5 1.5B, Gemma 2 2B, Llama 3.2 3B, Llama 3.1 8B), with and without CUDA graphs, at tensor parallelism 1 and 2. kvplan's KV-cache token count was within 0.7% of vLLM's in 14 of 15 runs and within 1.9% in all of them; before calibration (v0.1) it was within 2.7%. The calibration was fitted to these same runs, so these are in-sample numbers: other GPUs, vLLM versions, MoE and quantised models have not been measured yet. In out-of-sample runs on two RTX A6000s at tensor parallelism 2, Llama 3.1 8B was within 0.13%. The multimodal runs exposed two bugs: Qwen2.5-Omni was 83% low (its nested config was not read) and Qwen2.5-VL 7B was 2.7–3.6% high (vLLM's encoder profiling was not modelled). After fixing them, Qwen2.5-Omni is within 0.4% and Qwen2.5-VL is 1.6–2.5% high; the encoder term is approximate and kvplan says so in its output. The full table, method and known gaps are in [docs/validation.md](https://github.com/ShawnNotFound/kvplan/blob/master/docs/validation.md).

## Roadmap

- v0.2: calibrated runtime overheads (done for dense models, see Accuracy), a measurement script anyone can run on their GPU
- v0.3: llama.cpp profile, published validation table
- v0.4: `place`: several models across several machines with mixed GPUs

## Licence

Apache-2.0
