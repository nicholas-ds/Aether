# llama.cpp Backend Design

**Date:** 2026-04-03
**Status:** Approved

## Overview

Add a llama.cpp backend to Aether so GGUF models can be loaded and run alongside HuggingFace models. This is the planned second backend that triggers the abstraction of `BaseModel` into an interface with two concrete implementations.

## Structure

```
src/models/
  base.py              ← abstract ModelBackend(ABC)
  config.py            ← add n_gpu_layers, n_ctx fields
  backends/
    __init__.py
    huggingface.py     ← current BaseModel logic moved here as HuggingFaceBackend
    llama_cpp.py       ← new LlamaCppBackend
```

## Abstract Interface (`base.py`)

`BaseModel` is replaced by `ModelBackend(ABC)` with three abstract methods:

```python
class ModelBackend(ABC):
    @abstractmethod
    def load(self) -> bool: ...

    @abstractmethod
    def unload(self) -> bool: ...

    @abstractmethod
    def generate_response(self, messages, max_length, temperature, stream, enable_thinking) -> Generator: ...
```

`_parse_thinking` and `_stream_tokens` move to `ModelBackend` — they are format-agnostic `<think>` tag parsers shared by both backends.

## ModelConfig Changes

Two optional fields added for llama.cpp:

```python
n_gpu_layers: int = -1    # -1 = offload all layers to GPU
n_ctx: int = 32768        # context window size at load time
```

HuggingFace backend ignores these. All existing HF-specific fields stay as-is.

## HuggingFaceBackend (`backends/huggingface.py`)

Current `BaseModel` logic moved here verbatim. Class renamed to `HuggingFaceBackend`. No behavioral changes.

## LlamaCppBackend (`backends/llama_cpp.py`)

Loads via:
```python
Llama(model_path=str(config.local_model_path), n_gpu_layers=config.n_gpu_layers, n_ctx=config.n_ctx)
```

Generates via `create_chat_completion(messages, max_tokens, temperature, stream=True)`. Content chunks are piped through the shared `_stream_tokens` to parse `<think>` tags. Yields the same dict format as `HuggingFaceBackend`: `text`, `thinking`, `tokens_used`, `max_tokens`.

Token counts come from llama.cpp's usage info on the final streaming chunk. Context window size comes from `config.n_ctx`.

## Router Changes

Backend selected in `__init__` by file extension:

```python
path = config.model_config.local_model_path
if path and path.suffix == ".gguf":
    self.backend = LlamaCppBackend(config.model_config)
else:
    self.backend = HuggingFaceBackend(config.model_config)
```

`self.base_model` renamed to `self.backend`. `load_model`, `unload_model`, and `generate_response` delegate to `self.backend` unchanged.

## Data Flow

```
CLI → Router.generate_response()
        → backend.generate_response()
            → _stream_tokens()  (shared, in ModelBackend)
                → yield {text/thinking/tokens_used/max_tokens}
```

## Out of Scope

- Ollama or API backends (no concrete use case yet)
- Automatic format conversion between GGUF and safetensors
- Per-backend CLI flags beyond what ModelConfig already supports
