# Changelog

## Unreleased

- `kvplan recommend`: popular Hugging Face models that vLLM should start on your GPUs, each one fully checked. It asks (or takes as options) how many GPUs to use, the task (chat, code, vision, audio, omni), quality or speed, and the context length; pick one to see its full check or download it.
- `kvplan search`: find repos by a loose name such as `llama 3.1 8b instruct`.
- Model names without an owner (`Qwen2.5-7B-Instruct`) resolve to the most downloaded repo with that exact name. A misspelled or unknown name lists the closest popular models instead of failing, and in a terminal lets you pick one.
- A repo that does not exist was reported as "gated or private"; it is now reported as not found, with suggestions.
- `--gpu` defaults to `auto`: without a GPU option, kvplan plans for the GPUs on this machine, and explains what to pass when it finds none.
- `kvplan download` and `kvplan check --download`: download only the files vLLM needs, into the Hugging Face cache (with huggingface_hub) or a directory. A passing check now prints the download command and the matching `vllm serve` line.
- Output formats and files: `--format text|json|jsonl|csv|markdown` and `-o/--output` (a file, or a directory for one file per model), with `--append` for CSV and JSON Lines logs.
- `kvplan check --models FILE`: check a list of models (text, CSV with per-model option columns, or JSON; `-` for standard input) in one run, with one table or file for all of them.
- Config files (`~/.config/kvplan/config.toml` or `.json`, `KVPLAN_CONFIG`, `--config`): default options for every command or per command.
- Coloured terminal output: the verdict is green (should start), red (will not start) or yellow (not confirmed), errors are red and warnings yellow. `--color auto|always|never`; `NO_COLOR` and `FORCE_COLOR` are respected, and files, pipes and data formats stay plain.
- Python API: `kvplan.check`, `inspect`, `search`, `recommend` and `download` return plain data for use in other programs; the command line is now built on it.
- `kvplan login`: check a Hugging Face token and save it where huggingface_hub, vLLM and huggingface-cli read it. `HF_TOKEN_PATH` and `XDG_CACHE_HOME` are now respected when looking for a saved token.

## 0.1.1 (2026-10-30)

- The RoPE cache note is left out when it would read 0.00 GiB (small models). The cache is still counted with the weights.
- Tied models whose checkpoint still stores `lm_head.weight` (Qwen3 0.6B and 1.7B, Qwen's AWQ and FP8 releases) no longer count it: vLLM does not load it. kvplan's weights for these were up to 0.58 GiB high.
- An fp8 KV cache on a GPU older than compute capability 8.9 (A100, A6000, A5000, RTX 3090, T4, V100) gives "start not confirmed": vLLM 0.32 failed to start with it on RTX A6000 and A5000. kvplan no longer suggests `--kv-cache-dtype fp8` on those GPUs.
- "start not confirmed" now reads "kvplan could not check ..." instead of "could not size ...".
- Validated on MoE and quantised checkpoints (Granite MoE, AWQ, GPTQ, fp8 weights): 12 runs within 1.01%; see docs/validation.md.

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
