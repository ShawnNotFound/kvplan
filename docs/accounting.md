# How kvplan reproduces vLLM's memory accounting

All references are to vLLM `main` at commit `434eadc` (2026-10-29). Paths are relative to the vLLM repository.

## 0. Which config describes the language model

Sections 0 and 8 and the parts of section 3 marked "(vLLM 0.32)" cite the installed vLLM 0.32.0 package (commit `5df63a720`) and transformers 5.19.0, the versions in [validation.md](validation.md).

vLLM reads the language model's geometry from `get_hf_text_config` (`vllm/transformers_utils/config.py`), which calls transformers' `PretrainedConfig.get_text_config()` (`transformers/configuration_utils.py`). kvplan does the same on the raw config.json:

| Layout | Examples | Where the text config is |
|---|---|---|
| flat | Llama, Qwen2.5-VL | config.json itself |
| `text_config` (or `text_encoder`, `decoder`, `generator`) | Llava, Gemma 3, Qwen3-VL | that sub-config; several of them at once is ambiguous and transformers raises, so kvplan refuses too |
| `thinker_config.text_config` | Qwen2.5-Omni, Qwen3-Omni, Qwen3-ASR | `Qwen2_5OmniConfig.get_text_config` and `Qwen3OmniMoeConfig.get_text_config` return the thinker's text config |
| `llm_config` | InternVL, Bagel | vLLM renames it to `text_config` (`_CONFIG_ATTRS_MAPPING`, `_maybe_remap_hf_config_attrs` in `vllm/transformers_utils/config.py`) |
| `language_config` | DeepSeek-VL2 | vLLM's `DeepseekVLV2Config` loads it as `text_config` (`vllm/transformers_utils/configs/deepseek_vl2.py`) |

A nested text config may leave fields out and rely on its config class's defaults (Llava's `text_config` often names only `model_type: llama`); kvplan fills in `LlamaConfig`'s defaults for `llama`. For other model types a missing field sends kvplan to the weight shapes (below).

**Cross-check with the weights.** After reading the config, kvplan finds the language model's layers in the weight names (ignoring vision, audio and speech-generator prefixes) and compares each layer's K-projection rows with `num_kv_heads × head_dim`, and the layer count with `num_hidden_layers`. If they disagree it stops with an error instead of picking one. Packed quantised weights (GPTQ, AWQ, 4-bit) are skipped, because their shapes are not the logical ones.

**Weight-shape fallback.** When the config has no head or layer counts, kvplan derives the KV width per layer from the language model's K projections. Head counts stay unknown, so tensor-parallel replication cannot be checked; the output says the figures are uncertain. kvplan also warns when config.json names an architecture vLLM 0.32 does not implement natively (`src/kvplan/vllm_architectures.py`, generated from `vllm/model_executor/models/registry.py`); vLLM may then refuse the model or use its Transformers backend.

## 1. The budget

vLLM measures free memory after creating the CUDA context, NCCL buffers and cuBLAS handles, then reserves

```
requested = ceil(total_gpu_memory × gpu_memory_utilization)
```

`vllm/v1/worker/utils.py` `request_memory`. The default utilisation is 0.92 (`vllm/config/cache.py`). It is a fraction of **total** memory as CUDA reports it, which is `nvidia-smi`'s `memory.total` minus `memory.reserved` (on an RTX A6000: 49140 − 599 MiB = 47.40 GiB, exactly what vLLM logged). `--gpu auto` reads both fields; the presets subtract the reserved part for the GPUs where it was read from the driver (A6000, A5000, RTX 5090 Laptop, V100-32GB). If free memory at that point is below `requested`, vLLM refuses to start ("Free memory on device ... is less than desired GPU memory utilization"). kvplan models the context and NCCL as estimates (0.5 GiB, plus 0.25 GiB with tensor parallelism) and adds `--other-gpu-mem-gib`.

## 2. What the KV cache gets

```
available_kv = requested − consumed (weights + non-torch) − profiled peak
```

`vllm/v1/worker/gpu_worker.py` `Worker.determine_available_memory`; `vllm/utils/mem_utils.py` `memory_profiling`. In vLLM 0.32 the terms are measured like this:

- **consumed** = the drop in free GPU memory (`cudaMemGetInfo`) from before the engine was created to after the profiling run. It holds the weights and buffers ("Model loading took … GiB", the torch memory allocated during loading), memory outside torch (cuBLAS/cuDNN workspaces, NCCL buffers) and torch memory still reserved after `empty_cache()`.
- **profiled peak** = the transient headroom of the profiling run (torch's peak allocation minus what is still allocated afterwards) plus, with CUDA graphs on, vLLM's CUDA-graph memory estimate.

vLLM logs both ("Actual usage is … GiB for consumed memory (weights + non-torch), … GiB for peak activation, and … GiB for CUDAGraph memory") and the available KV memory it reports equals `requested − consumed − peak`.

| Term | kvplan | Status |
|---|---|---|
| weights | summed per tensor from the weight-file headers, plus the RoPE caches, see section 3 | exact for parameters and buffers (checked against a per-parameter dump of Qwen2.5-VL-7B at TP 2); vLLM's loading figure is 0.01–0.11 GiB higher, see section 3 |
| non-torch | 0.25 GiB, 0.45 GiB for models with a vision or audio encoder | fitted: text models measured 0.22–0.46 GiB above the loading figure; multimodal models 0.35–0.57 GiB (median 0.45 on the A6000 TP 1 fit runs); cause not traced |
| multimodal encoder profiling | section 8 | derived for the Qwen2.5-VL vision tower; approximate for other encoders |
| eager peak | `max(2 × max_num_batched_tokens × intermediate_size × 2 × 2 B / tp, min(max_num_seqs, max_num_batched_tokens) × vocab × 4 B)`: the MLP gate/up output or the sampler's fp32 logits, whichever is larger | fitted: within 0.08 GiB on 5 dense models at 2048 and 8192 batched tokens |
| torch.compile buffers | `7 × max_num_batched_tokens × hidden × 2 B / tp`, 0 with `--enforce-eager` | fitted: within 0.12 GiB, except Gemma 2 (0.45 GiB low) |
| CUDA graphs | 0.25 GiB, 0 with `--enforce-eager` | measured 0.20–0.24 GiB |

MoE models and models without `intermediate_size` use the older, uncalibrated estimate (`max_num_batched_tokens × hidden × 2 B × 12 / tp` plus two copies of the fp32 logits), and the output says so. Any of these can be replaced with `--overhead-gib`. See [validation.md](validation.md) for the measurements.

Batch defaults follow `EngineArgs.get_batch_defaults` (`vllm/engine/arg_utils.py`), OpenAI-server column (`vllm serve`; the offline `LLM()` class uses larger values, 8192 tokens on most GPUs, so pass `--max-num-batched-tokens 8192` to plan for it): 2048 tokens / 256 sequences on most GPUs and on A100/GB10; 8192 / 1024 from 70 GiB; 16384 / 1024 from 160 GiB.

## 3. Weights per GPU

Read from safetensors headers (exact byte ranges) or GGUF tensor offsets. Per tensor:

- Unquantised floating-point weights are held in 16 bit (vLLM's `dtype="auto"` runs fp32 checkpoints in 16 bit). Quantised weights and their scales keep their stored size.
- Embedding and `lm_head` are vocab-parallel: rows are padded to a multiple of 64 (`vllm/model_executor/layers/vocab_parallel_embedding.py` `pad_vocab_size`) and split over TP ranks. A checkpoint without `lm_head` is tied and stored once. When `tie_word_embeddings` is set but the checkpoint still stores `lm_head.weight` (Qwen's AWQ and FP8 releases), vLLM ties `lm_head` to the embedding (`lm_head.tie_weights` in `vllm/model_executor/models/qwen2.py`) and skips the stored copy (`AutoWeightsLoader` in `models/utils.py`), so kvplan does not count it.
- K and V projections: split over TP ranks, unless there are fewer KV heads than ranks, in which case each rank holds one whole head (`vllm/model_executor/layers/linear.py` `QKVParallelLinear`, `num_kv_head_replicas`).
- Replicated on every rank: 1-D tensors (norms, biases), MoE routers, MLA's `q_a_proj` and `kv_a_proj_with_mqa`.
- Everything else (attention and MLP matrices, experts) is split 1/tp.
- RoPE caches: `get_rope` (`vllm/model_executor/layers/rotary_embedding/__init__.py`) builds one cos/sin cache per distinct RoPE configuration and shares it across layers. It holds `max_position × rotary_dim` values in the model dtype (`RotaryEmbedding.__init__`, `base.py`), with `max_position = max_position_embeddings`, ×4 for M-RoPE (Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni: `mrope.py`, `positions_factor`), × the factor for linear scaling (`linear_scaling_rope.py`), and `original_max_position_embeddings × factor` for YaRN (`yarn_scaling_rope.py`). Gemma 3 has a second cache for its sliding-window layers. For Qwen2.5-VL-7B this is a [512000, 128] bf16 buffer, 0.122 GiB on every GPU. kvplan counts it with the weights. vLLM's "Model loading took" figure is a further 0.01–0.11 GiB above parameters plus buffers (other tensors still allocated at the end of loading); kvplan does not model that part.
- Layers of the language model with an index at or above `num_hidden_layers` are multi-token-prediction heads (e.g. DeepSeek-V3 layer 61) and are not loaded. Encoder layers (a 32-layer audio tower next to a 28-layer text model) are not affected.
- Parts vLLM does not load (vLLM 0.32): the prefixes a model class maps to `None` in its `hf_to_vllm_mapper`. kvplan lists these per architecture (`NOT_LOADED_PREFIXES` in `src/kvplan/weights.py`): `talker.` and `token2wav.` for Qwen2.5-Omni (`qwen2_5_omni_thinker.py`), `talker.` and `code2wav.` for Qwen3-Omni and Qwen3-ASR (`qwen3_omni_moe_thinker.py`, `qwen3_asr.py`), `tts` for MiniCPM-o (`minicpmo.py`). For other architectures with tensors that look like a speech generator (`talker`, `token2wav`, `code2wav`, `tts`, `vocoder`, ...), kvplan counts them and warns that vLLM may not load them.
- Encoders built from the plain transformers module rather than vLLM's tensor-parallel layers are held whole on every GPU (vLLM 0.32): the audio tower of Qwen2.5-Omni (`Qwen2_5OmniAudioEncoder`) and of Qwen2-Audio (`Qwen2AudioEncoder`). Vision towers that use vLLM's parallel layers (Qwen2.5-VL) are split.

## 4. KV bytes per token per layer (per TP rank)

`vllm/v1/kv_cache_interface.py` `AttentionSpec.page_size_bytes`; `vllm/config/model.py` `get_num_kv_heads`.

| Layer | Bytes per token |
|---|---|
| MHA / GQA / MQA | `max(1, kv_heads // tp) × (head_dim + v_head_dim) × dtype` |
| MLA | `(kv_lora_rank + qk_rope_head_dim) × dtype`, not split by TP |

`dtype` is the model's 16-bit type for `--kv-cache-dtype auto` and 1 byte for fp8.

## 5. Which layers use a sliding window

If the config's `layer_types` lists per-layer types, vLLM follows them (`vllm/model_executor/models/llama.py`, `gemma2.py`, `gemma3.py`; `gpt_oss.py` uses even layers). If there is no `layer_types`, vLLM applies `sliding_window` to every layer (`vllm/engine/arg_utils.py` sets `CacheConfig.sliding_window`, and each attention layer falls back to it). transformers fills in `layer_types` for some model types when config.json omits it; kvplan reproduces those defaults:

| model_type | sliding layers |
|---|---|
| gemma2, gpt_oss | even layers (0, 2, 4, …) |
| gemma3, gemma3_text | all but every 6th (`sliding_window_pattern`) |
| cohere2 | all but every 4th; vLLM uses a window of `sliding_window + 1` |
| olmo3 | all but every 4th |
| qwen2, qwen3 | only if `use_sliding_window`: all layers when `max_window_layers` is 0; a mixed pattern makes vLLM refuse to start |
| qwen3_moe | all layers, only if `use_sliding_window` |
| qwen2_moe | none (vLLM runs full attention) |
| ministral, mistral, mixtral, phi3, starcoder2 and any other type without default `layer_types` | all layers, if `sliding_window` is set |

## 6. Context length and KV dtype defaults

- Maximum length: the smallest of `max_position_embeddings`, `n_positions`, `max_seq_len`, `seq_length`, `model_max_length`, `max_target_positions`, `max_sequence_length`, `max_seq_length`, `seq_len`, capped by the tokenizer's `model_max_length`, multiplied by the RoPE scaling factor unless the RoPE type already states the scaled length (su, longrope, llama3, yarn, deepseek_yarn, deepseek_llama_scaling) or the model is Gemma 3 (`vllm/config/model.py` `_get_and_verify_max_len`).
- Default `max_model_len`: that maximum, except LongRoPE models, which default to `original_max_position_embeddings`.
- `--kv-cache-dtype auto`: the model's 16-bit type, unless a modelopt `quantization_config` sets an fp8 KV cache (`vllm/utils/torch_utils.py` `resolve_kv_cache_dtype_string`).
- An fp8 KV cache needs compute capability 8.9 or newer on vLLM 0.32's Triton backend (`vllm/v1/attention/backends/triton_attn.py`); FlashAttention 2 has no fp8 KV path, and FlashInfer's fp8 kernels failed to build on RTX A6000 and A5000 (8.6). For a GPU kvplan knows to be older than 8.9, an fp8 KV cache gives "start not confirmed".
- MLA models use 64-token blocks (FlashMLA) or 128 on Blackwell (CUTLASS MLA); other models 16.

## 7. Blocks, groups and capacity

`vllm/v1/core/kv_cache_planning.py`, `vllm/v1/core/kv_cache_utils.py`.

- Page size per layer = `block_size × bytes per token` (block size 16 by default).
- Uniform models: one group; `bytes_per_block` is the sum of all layers' pages.
- Hybrid models (full + sliding): layers are bucketed by type; `group_size` = the smallest bucket, or the largest if `largest < 1.5 × smallest`; each bucket is split into `cdiv(n, group_size)` groups; `bytes_per_block = group_size × page_size` (`_get_kv_cache_groups_uniform_page_size`).
- `num_blocks = available_kv // bytes_per_block`.
- Blocks one request at `max_model_len` needs, per group: full `cdiv(L, block_size)`; sliding `cdiv(min(window − 1 + in_flight, L), block_size) + 1` (`SlidingWindowSpec.max_admission_blocks_per_request`), where `in_flight = 2 × max_num_batched_tokens` because async scheduling (on by default) keeps two batches in flight (`VllmConfig.max_in_flight_tokens`).
- Max concurrency = `num_blocks / Σ_groups blocks per request` and KV tokens = `int(concurrency × L)`, as vLLM logs them (`get_max_concurrency_for_kv_cache_config`, `get_kv_cache_capacity`).
- Start check: one full-length request needs `bytes_per_block × Σ_groups blocks per request` (this counts padding layers in hybrid models, `_max_memory_usage_bytes_from_groups`), which must fit in `available_kv − bytes_per_block` (one block is held back as the null block). Otherwise vLLM stops and suggests the longest length that fits, found by binary search (`check_enough_kv_cache_memory`, `estimate_max_model_len`). kvplan reports the same number.

## 8. Multimodal encoder profiling

vLLM 0.32 runs the multimodal encoder inside the profiling run, before it sizes the KV cache. `GPUModelRunner.profile_run` (`vllm/v1/worker/gpu/model_runner.py`) builds a `MultiModalBudget` (`vllm/multimodal/encoder_budget.py`), encodes its dummy items in one batch (`EncoderRunner.profile_encoder_cache`, `vllm/v1/worker/gpu/mm/encoder_runner.py`), keeps the outputs in the encoder cache, then runs the text pass, and clears the cache. The profiled peak is therefore

```
max(encoder pass, encoder outputs + text pass)
```

What the budget picks:

1. Every modality not set to 0 by `--limit-mm-per-prompt` gets a maximum number of tokens per item from the model's processing info (`get_mm_max_tokens_per_item`). For the Qwen2-VL family (`qwen2_vl.py`): an image has `max_pixels / (patch_size × spatial_merge_size)²` tokens (16,384 for Qwen2.5-VL's 12,845,056-pixel default), and a video has as many frames of that size as fit in `max_model_len`, at most 14, padded to an even count (`get_num_frames_with_most_features`). `--mm-processor-kwargs '{"max_pixels": N}'` changes both.
2. The modality with the most tokens is profiled; ties go to the later name ("video" before "image").
3. Encoder budget = `max(max_num_batched_tokens, largest item)` (`compute_mm_encoder_budget`, `vllm/v1/core/encoder_cache_manager.py`). The batch holds `max(1, min(budget // item, max_num_seqs × max(1, min(limit, max_model_len // item))))` items (`MultiModalBudget._get_max_items`).
4. The items come from the model's dummy-input builder and its HF processor, which can produce fewer tokens than step 1 budgets for. Qwen2.5-Omni budgets 16,384 tokens for a video, but its HF processor resizes video frames to at most 768 × 28 × 28 pixels, so the profiled video has 729 tokens and the encoder hardly shows in the peak. With `--limit-mm-per-prompt '{"video": 0}'` an image of 16,384 tokens is profiled instead and the peak rises from 0.1 to 1.9 GiB.

kvplan reproduces steps 1–4 for Qwen2-VL, Qwen2.5-VL and Qwen2.5-Omni (`src/kvplan/mm.py`); every dummy batch in the measurements matched (modality, item count, tokens and encoder positions). For Whisper-style audio encoders (Qwen2-Audio, Qwen2.5-Omni's audio tower) an item is 30 s: 3000 mel frames, 1500 encoder positions, 750 tokens. For other models kvplan cannot derive the item and assumes one item of `max_num_batched_tokens` tokens; the output then says the figure is uncertain.

Memory of the encoder pass, per encoder position, from the live tensors at its peak:

| Encoder | Rule | Status |
|---|---|---|
| Qwen2.5-VL / Qwen2.5-Omni vision tower (`Qwen2_5_VisionBlock`, `qwen2_5_vl.py`) | Inside the MLP the block holds the fused-norm input and residual (`2h`), `gate_up` (`2i/tp`), the activation output (`i/tp`) and `down_proj`'s output (`h`); with TP > 1 the all-reduce returns one more `h` tensor. All bf16, plus the pixel input (1,176 values per position: uint8 for Qwen2.5-VL, whose vLLM processor normalises on the GPU, bf16 for Qwen2.5-Omni). For Qwen2.5-VL (`h` 1280, `i` 3420): 29.4 KB per position at TP 1 and 21.7 KB at TP 2 | derived; measured 29.9 KB and 22.2 KB (over 30,720 to 65,536 positions) |
| Whisper-style audio (transformers' `Qwen2AudioEncoderLayer`, not split across GPUs) | residual and normed input (`2h`) plus `fc1` and GELU outputs (`2i`), bf16, plus the mel frames | approximate: 27 KB per position against 35 KB measured on Qwen2-Audio; the text pass set the peak in every run except TP 2 |
| others | the Qwen2.5-VL rule applied to the encoder config | approximate, and flagged as uncertain: MiniCPM-o 2.6 profiles 8 videos and peaks at 5.9 GiB, which kvplan does not predict |

Encoder outputs held during the text pass: `items × tokens × text hidden_size × 2 B` on every GPU.

The encoder runs eagerly even with CUDA graphs on; vLLM's reported peak then also contains its CUDA-graph estimate (0.22–0.24 GiB), which kvplan already counts. `--skip-mm-profiling` removes the encoder pass; kvplan accepts the same flag.

## Known gaps

- Overheads in section 2 are estimates.
- The `--gpu` presets are `nvidia-smi` totals, a few hundred MiB above what CUDA reports; use `--gpu-mem-gib` for exact numbers.
- MLA block size depends on the attention backend vLLM picks for the GPU; kvplan assumes FlashMLA (64) or CUTLASS MLA (128 on Blackwell). Override with `--block-size`.
- FP8 MLA checkpoints: vLLM adds a 16-bit copy of `kv_b_proj` (W_UK/W_UV), which kvplan counts.
- Quantisation kernels that repack or pad weights at load time (e.g. Marlin) can add a little memory that is not modelled.
- Not modelled at all: Mamba / linear-attention layers, chunked local attention (Llama 4), sparse MLA, MLA-like configs vLLM does not run as MLA (e.g. MiniCPM3), pipeline/expert/data parallelism, LoRA, speculative decoding. The multimodal encoder term (section 8) is derived only for the Qwen2.5-VL vision tower.
