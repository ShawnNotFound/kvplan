"""kvplan: plan vLLM deployments without the GPU.

Use it from Python as well as from the command line:

    import kvplan
    result = kvplan.check("Qwen/Qwen2.5-7B-Instruct", gpu="A6000", max_model_len=8192)
    print(result.starts, result.kv_tokens)

See kvplan.api for every function and its options.
"""

__version__ = "0.2.0"

from .api import (CheckResult, KvplanError, ModelNotFound, check, download, inspect,  # noqa: E402
                  recommend, search)

__all__ = ["check", "inspect", "search", "recommend", "download", "CheckResult", "ModelNotFound", "KvplanError",
           "__version__"]
