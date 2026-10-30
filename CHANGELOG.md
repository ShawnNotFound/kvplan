# Changelog

## 0.1.0 (unreleased)

First release.

- `kvplan check`: predicts whether vLLM will start on a given GPU, how much memory goes to weights, runtime and KV cache, the KV-cache size in tokens and the maximum concurrency. Reads config.json, safetensors headers and GGUF files, locally or from Hugging Face, without downloading weights.
- `kvplan inspect`: shows the model geometry kvplan read (layers, heads, KV layout, weight sizes).
- `--gpu auto` reads the local GPU through nvidia-smi, including memory other processes hold; presets for common GPUs (A6000, A5000, RTX 5090 Laptop, V100 and others).
- Tensor parallelism, sliding-window and hybrid KV layouts, MLA, quantised checkpoints and `--json` output.
- Runtime overheads calibrated against vLLM 0.32 startup logs; see docs/validation.md.
