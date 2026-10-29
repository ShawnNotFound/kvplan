# How kvplan reproduces vLLM's memory accounting

All references are to vLLM `main` at commit `434eadc` (2026-10-29). Paths are relative to the vLLM repository.

## 1. The budget

vLLM measures free memory after creating the CUDA context, NCCL buffers and cuBLAS handles, then reserves

```
requested = ceil(total_gpu_memory × gpu_memory_utilization)
```

`vllm/v1/worker/utils.py` `request_memory`. The default utilisation is 0.92 (`vllm/config/cache.py`). It is a fraction of **total** memory. If free memory at that point is below `requested`, vLLM refuses to start ("Free memory on device ... is less than desired GPU memory utilization"). kvplan models the context and NCCL as estimates (0.5 GiB, plus 0.25 GiB with tensor parallelism) and adds `--other-gpu-mem-gib`.

## 2. What the KV cache gets

```
available_kv = requested − weights − (activation peak + persistent buffers + non-torch growth) − CUDA-graph estimate
```

`vllm/v1/worker/gpu_worker.py` `Worker.determine_available_memory`; `vllm/utils/mem_utils.py` `memory_profiling`.

| Term | kvplan | Status |
|---|---|---|
| weights | summed per tensor from the weight-file headers, see section 3 | exact |
| sampler logits | `min(max_num_seqs, max_num_batched_tokens) × vocab × 4 B × 2` | estimate |
| activation peak | `max_num_batched_tokens × hidden × 2 B × 12 / tp` | estimate |
| CUDA graphs | 0.5 GiB, 0 with `--enforce-eager` | estimate |
| non-torch growth | 0.25 GiB | estimate |

The estimates can be replaced with `--overhead-gib`. They will be calibrated against real startup logs in v0.2.

Batch defaults follow `EngineArgs.get_batch_defaults` (`vllm/engine/arg_utils.py`), OpenAI-server column: 2048 tokens / 256 sequences on most GPUs and on A100/GB10; 8192 / 1024 from 70 GiB; 16384 / 1024 from 160 GiB.

## 3. Weights per GPU

Read from safetensors headers (exact byte ranges) or GGUF tensor offsets. Per tensor:

- Unquantised floating-point weights are held in 16 bit (vLLM's `dtype="auto"` runs fp32 checkpoints in 16 bit). Quantised weights and their scales keep their stored size.
- Embedding and `lm_head` are vocab-parallel: rows are padded to a multiple of 64 (`vllm/model_executor/layers/vocab_parallel_embedding.py` `pad_vocab_size`) and split over TP ranks. A checkpoint without `lm_head` is tied and stored once.
- K and V projections: split over TP ranks, unless there are fewer KV heads than ranks, in which case each rank holds one whole head (`vllm/model_executor/layers/linear.py` `QKVParallelLinear`, `num_kv_head_replicas`).
- Replicated on every rank: 1-D tensors (norms, biases), MoE routers, MLA's `q_a_proj` and `kv_a_proj_with_mqa`.
- Everything else (attention and MLP matrices, experts) is split 1/tp.
- Layers with an index at or above `num_hidden_layers` are multi-token-prediction heads (e.g. DeepSeek-V3 layer 61) and are not loaded.

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

## Known gaps

- Overheads in section 2 are estimates.
- The `--gpu` presets are `nvidia-smi` totals, a few hundred MiB above what CUDA reports; use `--gpu-mem-gib` for exact numbers.
- MLA block size depends on the attention backend vLLM picks for the GPU; kvplan assumes FlashMLA (64) or CUTLASS MLA (128 on Blackwell). Override with `--block-size`.
- FP8 MLA checkpoints: vLLM adds a 16-bit copy of `kv_b_proj` (W_UK/W_UV), which kvplan counts.
- Quantisation kernels that repack or pad weights at load time (e.g. Marlin) can add a little memory that is not modelled.
- Not modelled at all: Mamba / linear-attention layers, chunked local attention (Llama 4), sparse MLA, MLA-like configs vLLM does not run as MLA (e.g. MiniCPM3), pipeline/expert/data parallelism, LoRA, speculative decoding, multimodal encoder caches.
