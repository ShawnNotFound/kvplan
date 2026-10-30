# Changelog

## 0.1.0 (2026-10-30)

First release.

- `kvplan check`: predicts whether vLLM will start on a given GPU, how much memory goes to weights, runtime and KV cache, the KV-cache size in tokens and the maximum concurrency. Reads config.json, safetensors headers and GGUF files, locally or from Hugging Face, without downloading weights.
- `kvplan inspect`: shows the model geometry kvplan read (layers, heads, KV layout, weight sizes).
- `--gpu auto` reads the local GPU through nvidia-smi, including memory other processes hold; presets for common GPUs (A6000, A5000, RTX 5090 Laptop, V100 and others).
- Tensor parallelism, sliding-window and hybrid KV layouts, MLA, quantised checkpoints and `--json` output.
- Runtime overheads calibrated against vLLM 0.32 startup logs; see docs/validation.md. RoPE cos/sin caches are counted with the weights.
- Multimodal models: the language-model config is found the way vLLM finds it (`text_config`, `thinker_config.text_config`, `llm_config`, `language_config`); parts vLLM does not load, such as the Qwen Omni speech generators, are not counted. vLLM's startup encoder profiling is reproduced for Qwen2-VL, Qwen2.5-VL and Qwen2.5-Omni (which dummy items vLLM encodes, and the vision encoder's memory, derived from vLLM's code), approximated for Whisper-style audio encoders, and flagged as uncertain for other encoders. New options `--limit-mm-per-prompt`, `--mm-processor-kwargs` (max_pixels, min_pixels) and `--skip-mm-profiling`, as in vLLM.
- A third verdict, "start not confirmed" (exit code 3, `"starts": null`), when a term kvplan cannot size could decide the start: it names the term, shows the memory left under kvplan's estimates and suggests what to do.
- kvplan refuses when config.json and the weights disagree on the KV size, and warns when the geometry comes from weight shapes or the architecture is not one vLLM implements natively. Gemma 4, whose full-attention layers have their own head size, is refused instead of sized wrongly.
