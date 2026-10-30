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

> **Status: alpha.** Weight and KV sizes are computed exactly from the model files. The runtime overheads (activation peak, torch.compile buffers, CUDA graphs) are fitted to vLLM 0.32 startup logs from text-model runs on RTX A6000 and A5000 GPUs, and the multimodal encoder memory is derived from vLLM's code and checked on 27 more runs; see [Accuracy](https://github.com/ShawnNotFound/kvplan#accuracy) for what that does and does not cover.

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

`MODEL` is a Hugging Face repo id or a local directory or file. `check` understands the vLLM options that change memory: `-tp/--tensor-parallel-size`, `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--block-size`, `--enforce-eager`. Add `--json` for scripts and CI. The exit code is 0 when vLLM should start and 1 when it won't. It is 3 ("start not confirmed", `"starts": null` in the JSON) when a term kvplan cannot size could decide the outcome: an encoder whose startup profiling it cannot size, an architecture vLLM has no native implementation of, or attention geometry read from weight shapes. The output then names the unsized term, the memory left under kvplan's estimates, and what to do. Exit code 2 is a kvplan error.

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
| Multimodal | checkpoint parts vLLM does not load (Omni speech generators) are skipped; vLLM's startup encoder profiling is reproduced for the Qwen2-VL family (dummy items, encoder memory), approximated for Whisper-style audio and flagged as uncertain for other encoders; `--limit-mm-per-prompt`, `--mm-processor-kwargs` and `--skip-mm-profiling` as in vLLM |
| Attention | MHA, GQA, MQA, sliding-window and hybrid full/sliding layouts (Gemma 2/3, gpt-oss, Cohere2, Qwen2/3 with sliding window), MLA (DeepSeek V2/V3, Kimi K2) |
| Weights | exact bytes per tensor, 16-bit loading of fp32 checkpoints, quantised formats at their stored size, vocab padding, tied embeddings, MTP layers excluded, tensor-parallel sharding and KV-head replication |
| vLLM | utilisation budget, free-memory check, KV block grouping and rounding, max concurrency, the max_model_len check and its suggested length |

In a sample of the 300 most-downloaded text-generation models on Hugging Face, kvplan's sources describe 93% without a token and 98% with one (details in [docs/coverage.md](https://github.com/ShawnNotFound/kvplan/blob/master/docs/coverage.md)).

**Not yet supported:** models with Mamba or linear-attention layers (Qwen3-Next, Jamba, Nemotron-H, Falcon-H1), DeepSeek V3.2 sparse attention, Gemma 4 (full-attention layers with their own head size), encoder-decoder models, pipeline/expert/data parallelism, LoRA, speculative decoding, and llama.cpp. kvplan says so instead of guessing.

## Accuracy

Checked against vLLM 0.32.0 on RTX A6000 and RTX A5000 GPUs.

- **Text models** (Qwen2.5 1.5B, Gemma 2 2B, Llama 3.2 3B, Llama 3.1 8B; CUDA graphs and eager; tensor parallelism 1 and 2): kvplan's KV-cache token count is within 1.9% of vLLM's in all 15 runs (mean 0.5%). These runs were used for calibration, so they are in-sample. On two RTX A6000s at tensor parallelism 2, which were not used, Llama 3.1 8B is within 0.22%.
- **Multimodal models:** kvplan reproduces the dummy images and videos vLLM profiles at startup for Qwen2-VL, Qwen2.5-VL and Qwen2.5-Omni, including the effect of `--mm-processor-kwargs` max_pixels and `--limit-mm-per-prompt`, and derives the vision encoder's memory from vLLM's code. On 16 fit runs (Qwen2.5-VL 7B, Qwen2.5-Omni 3B and 7B, Qwen2-Audio 7B on an A6000) it is within 0.72%; on 11 held-out runs (A5000, tensor parallelism 2) within 2.24%. Other encoders are not sized: for MiniCPM-o 2.6 kvplan was 19–23% high, and it now warns that the figure is uncertain for any encoder it cannot size.
- **MoE and quantised models:** on 12 runs of Granite 3.1 1B-A400M (MoE), Qwen2.5 1.5B AWQ and GPTQ, and Qwen3 1.7B FP8, none of them used for fitting, kvplan is within 1.01% (mean 0.33%). An fp8 KV cache did not start on the RTX A6000 or A5000 (compute capability 8.6) with vLLM 0.32, so kvplan says "start not confirmed" for it on GPUs older than 8.9.

Not measured yet: large MoE and MLA models, the size of an fp8 KV cache, other vLLM versions, Hopper or Blackwell GPUs. The full table, method and known gaps are in [docs/validation.md](https://github.com/ShawnNotFound/kvplan/blob/master/docs/validation.md).

## Roadmap

- v0.2: calibrated runtime overheads (done for dense models, see Accuracy), a measurement script anyone can run on their GPU
- v0.3: llama.cpp profile, published validation table
- v0.4: `place`: several models across several machines with mixed GPUs

## Licence

Apache-2.0
