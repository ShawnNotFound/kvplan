# Metadata coverage on the Hugging Face Hub

Measured on 2026-10-29, read-only and without a token, with the scripts in the research notes (`hf-coverage/`). "Covered" means the files give everything the KV and weight maths needs.

| Sample | config.json alone | + safetensors / GGUF headers | + a user token (gated repos) | Not coverable |
|---|---:|---:|---:|---:|
| Top 300 text-generation models by downloads | 68.3% | 93.3% | 97.7% | 2.3% |
| 200 text-generation models created in the last 90 days | 52.5% | 92.5% | 96.5% | 3.5% |
| Top 100 GGUF repos | 10.0% | 98.0% | 98.0% | 2.0% |

Weighted by downloads, the top-300 sample is 94.8% covered without a token and 98.7% with one.

What is not coverable: very new or custom architectures whose config.json uses non-standard field names, GGUF files for architectures llama.cpp does not define yet, repos with no config or weights, and a few older encoder-decoder models. kvplan reports these instead of guessing.

Raw summary: [coverage-summary.json](coverage-summary.json).
