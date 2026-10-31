"""Load a model's spec and tensor list from the Hub or disk, with fallbacks.

Order: config.json + safetensors headers; then a GGUF header; then, when config.json
lacks attention fields, the safetensors tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import (MissingConfigFields, ModelSpec, check_against_tensors, spec_from_config, spec_from_gguf,
                    spec_from_tensors)
from .sources import Source, SourceError
from .weights_meta import Tensor, gguf_files, read_gguf, read_safetensors, safetensors_files


@dataclass
class LoadedModel:
    spec: ModelSpec
    tensors: list[Tensor]
    files: list[str]
    repo_files: list[str] = field(default_factory=list)   # every file in the repo or directory


def load(source: Source, gguf: str | None = None) -> LoadedModel:
    files = source.list_files()
    want_gguf = gguf is not None

    config = None if want_gguf else source.read_json("config.json")
    index = None if want_gguf else source.read_json("model.safetensors.index.json")
    shards = [] if want_gguf else safetensors_files(files, index)

    if config is not None and shards:
        tensors = read_safetensors(source, shards)
        tokenizer_config = source.read_json("tokenizer_config.json") if "tokenizer_config.json" in files else None
        preprocessor = (source.read_json("preprocessor_config.json")
                        if "preprocessor_config.json" in files else None)
        try:
            spec = spec_from_config(config, tokenizer_config, preprocessor)
        except MissingConfigFields as e:
            archs = config.get("architectures")
            spec = spec_from_tensors(tensors, config.get("model_type", "unknown"), reason=str(e),
                                     architecture=archs[0] if isinstance(archs, list) and archs else None)
        check_against_tensors(spec, tensors)
        return LoadedModel(spec, tensors, shards, files)

    parts = gguf_files(files, prefer=gguf or None)
    if parts:
        ggs = [read_gguf(source, p) for p in parts]
        spec = spec_from_gguf(ggs[0].metadata)
        spec.notes.append(f"read from GGUF file {parts[0]}" + (f" (+{len(parts) - 1} parts)" if len(parts) > 1 else "")
                          + "; vLLM's GGUF support is experimental")
        return LoadedModel(spec, [t for g in ggs for t in g.tensors], parts, files)

    if shards:
        tensors = read_safetensors(source, shards)
        return LoadedModel(spec_from_tensors(tensors), tensors, shards, files)
    if config is not None:
        if any(f.endswith((".bin", ".pt", ".pth")) for f in files):
            raise SourceError(f"{source.name}: only PyTorch .bin weights, whose sizes kvplan cannot read without "
                              "downloading them; kvplan needs safetensors or GGUF files")
        raise SourceError(f"{source.name}: has config.json but no safetensors or GGUF weights")
    raise SourceError(f"{source.name}: no config.json, safetensors or GGUF files found")
