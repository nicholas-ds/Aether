# llama.cpp Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a llama.cpp backend so Aether can load and run GGUF models alongside HuggingFace models, with the Router selecting the backend automatically by file extension.

**Architecture:** `BaseModel` in `src/models/base.py` becomes abstract `ModelBackend(ABC)`. Current `BaseModel` logic moves to `HuggingFaceBackend` in `src/models/backends/huggingface.py`. New `LlamaCppBackend` in `src/models/backends/llama_cpp.py` handles GGUF files via `llama-cpp-python`. `Router` selects the backend by checking `local_model_path.suffix == ".gguf"`.

**Tech Stack:** `llama-cpp-python` (already in requirements.txt), `abc.ABC`, `unittest.mock` for tests, `pytest`

---

## File Map

**Create:**
- `src/models/backends/__init__.py` — package marker
- `src/models/backends/huggingface.py` — `HuggingFaceBackend(ModelBackend)`, current BaseModel logic moved here
- `src/models/backends/llama_cpp.py` — `LlamaCppBackend(ModelBackend)`
- `tests/conftest.py` — adds project root to sys.path for test imports
- `tests/test_model_backend.py` — abstract interface + shared thinking parser tests
- `tests/test_config.py` — ModelConfig new field tests
- `tests/test_huggingface.py` — HuggingFaceBackend tests (mocked HF libs)
- `tests/test_llama_cpp.py` — LlamaCppBackend tests (mocked Llama)
- `tests/test_router.py` — Router backend selection tests

**Modify:**
- `src/models/base.py` — replace `BaseModel` with abstract `ModelBackend(ABC)`; move `_parse_thinking` and `_stream_tokens` here
- `src/models/config.py` — add `n_gpu_layers: int = -1` and `n_ctx: int = 32768`
- `src/orchestrator/router.py` — import backends, select by extension, rename `base_model` → `backend`
- `scripts/inspect_raw_response.py` — update import from `BaseModel` to `HuggingFaceBackend`

---

### Task 1: Abstract ModelBackend + test infrastructure

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_model_backend.py`
- Modify: `src/models/base.py`

- [ ] **Step 1: Create test infrastructure**

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 2: Write failing tests for ModelBackend**

Create `tests/test_model_backend.py`:
```python
import pytest
from pathlib import Path
from src.models.base import ModelBackend
from src.models.config import ModelConfig


class ConcreteBackend(ModelBackend):
    def load(self) -> bool:
        return True
    def unload(self) -> bool:
        return True
    def generate_response(self, messages, max_length=512, temperature=0.7, stream=False, enable_thinking=None):
        yield {"text": "response"}


def make_config():
    return ModelConfig(model_name="test", local_model_path=Path("/fake/path"))


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        ModelBackend(make_config())


def test_concrete_subclass_instantiates():
    backend = ConcreteBackend(make_config())
    assert backend.config.model_name == "test"


def test_parse_thinking_with_tags():
    backend = ConcreteBackend(make_config())
    thinking, content = backend._parse_thinking("<think>reasoning here</think>actual response")
    assert thinking == "reasoning here"
    assert content == "actual response"


def test_parse_thinking_without_tags():
    backend = ConcreteBackend(make_config())
    thinking, content = backend._parse_thinking("just a response")
    assert thinking == ""
    assert content == "just a response"


def test_stream_tokens_text():
    backend = ConcreteBackend(make_config())
    tokens = ["hello ", "world"]
    results = list(backend._stream_tokens(iter(tokens)))
    assert results == [{"text": "hello "}, {"text": "world"}]


def test_stream_tokens_thinking():
    backend = ConcreteBackend(make_config())
    tokens = ["<think>", "reasoning", "</think>", "response"]
    results = list(backend._stream_tokens(iter(tokens)))
    assert {"thinking": "reasoning"} in results
    assert {"text": "response"} in results
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_model_backend.py -v
```

Expected: FAIL — `ImportError` or `cannot import name 'ModelBackend'`

- [ ] **Step 4: Replace `src/models/base.py` with abstract `ModelBackend`**

```python
from abc import ABC, abstractmethod
import logging

from .config import ModelConfig


class ModelBackend(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _parse_thinking(self, text: str) -> tuple[str, str]:
        if "</think>" in text:
            thinking, content = text.split("</think>", 1)
            return thinking.replace("<think>", "").strip(), content.strip()
        return "", text.strip()

    def _stream_tokens(self, tokens):
        in_thinking = False
        for raw_token in tokens:
            if "<think>" in raw_token:
                in_thinking = True
                remainder = raw_token.replace("<think>", "")
                if remainder:
                    yield {"thinking": remainder}
                continue
            if "</think>" in raw_token:
                in_thinking = False
                remainder = raw_token.replace("</think>", "")
                if remainder:
                    yield {"text": remainder}
                continue
            yield {"thinking": raw_token} if in_thinking else {"text": raw_token}

    @abstractmethod
    def load(self) -> bool: ...

    @abstractmethod
    def unload(self) -> bool: ...

    @abstractmethod
    def generate_response(self, messages: list[dict], max_length: int = 512, temperature: float = 0.7, stream: bool = False, enable_thinking: bool | None = None): ...
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_model_backend.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models/base.py tests/conftest.py tests/test_model_backend.py
git commit -m "feat: abstract ModelBackend interface with shared thinking parsers"
```

---

### Task 2: Add llama.cpp fields to ModelConfig

**Files:**
- Create: `tests/test_config.py`
- Modify: `src/models/config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config.py`:
```python
from pathlib import Path
from src.models.config import ModelConfig


def make_config(**kwargs):
    return ModelConfig(model_name="test", local_model_path=Path("/fake/path"), **kwargs)


def test_n_gpu_layers_default():
    assert make_config().n_gpu_layers == -1


def test_n_ctx_default():
    assert make_config().n_ctx == 32768


def test_n_gpu_layers_custom():
    assert make_config(n_gpu_layers=20).n_gpu_layers == 20


def test_n_ctx_custom():
    assert make_config(n_ctx=8192).n_ctx == 8192
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `TypeError: unexpected keyword argument 'n_gpu_layers'`

- [ ] **Step 3: Add fields to `src/models/config.py`**

Add two fields after `enable_thinking` in the `ModelConfig` dataclass:

```python
    enable_thinking: bool = True
    n_gpu_layers: int = -1
    n_ctx: int = 32768
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/config.py tests/test_config.py
git commit -m "feat: add n_gpu_layers and n_ctx to ModelConfig for llama.cpp"
```

---

### Task 3: HuggingFaceBackend

**Files:**
- Create: `src/models/backends/__init__.py`
- Create: `src/models/backends/huggingface.py`
- Create: `tests/test_huggingface.py`
- Modify: `scripts/inspect_raw_response.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_huggingface.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.models.backends.huggingface import HuggingFaceBackend
from src.models.config import ModelConfig


def make_config(path=None):
    if path:
        return ModelConfig(model_name="test", local_model_path=Path(path))
    return ModelConfig(model_name="test", use_huggingface=True, huggingface_model_id="org/model")


def test_load_success(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = ModelConfig(model_name="test", local_model_path=model_dir)
    with patch("src.models.backends.huggingface.AutoTokenizer"), \
         patch("src.models.backends.huggingface.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = MagicMock()
        backend = HuggingFaceBackend(config)
        result = backend.load()
    assert result is True
    assert backend.model is not None


def test_load_missing_path():
    config = ModelConfig(model_name="test", local_model_path=Path("/does/not/exist"))
    backend = HuggingFaceBackend(config)
    result = backend.load()
    assert result is False


def test_unload_clears_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = ModelConfig(model_name="test", local_model_path=model_dir)
    with patch("src.models.backends.huggingface.AutoTokenizer"), \
         patch("src.models.backends.huggingface.AutoModelForCausalLM") as MockModel:
        MockModel.from_pretrained.return_value = MagicMock()
        backend = HuggingFaceBackend(config)
        backend.load()
        result = backend.unload()
    assert result is True
    assert backend.model is None


def test_generate_raises_when_not_loaded():
    backend = HuggingFaceBackend(make_config())
    with pytest.raises(RuntimeError, match="Model not loaded"):
        list(backend.generate_response([{"role": "user", "content": "hi"}]))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_huggingface.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.backends'`

- [ ] **Step 3: Create `src/models/backends/__init__.py`**

Create as an empty file.

- [ ] **Step 4: Create `src/models/backends/huggingface.py`**

```python
import logging
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ..base import ModelBackend
from ..config import ModelConfig


class HuggingFaceBackend(ModelBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    def _validate_environment(self) -> None:
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Install the correct PyTorch build for your GPU: "
                "https://pytorch.org/get-started/locally/"
            )
        if self.config.local_model_path and not self.config.local_model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.config.local_model_path}")

    def load(self) -> bool:
        try:
            self._validate_environment()
            model_path = self.config.local_model_path if self.config.local_model_path else self.config.huggingface_model_id
            self.logger.info(f"Loading model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=str(self.config.cache_dir)
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=self.config.torch_dtype,
                device_map={"": 0},
                quantization_config=self.config.quantization_config,
                trust_remote_code=True,
                local_files_only=self.config.local_model_path is not None,
                cache_dir=str(self.config.cache_dir),
                low_cpu_mem_usage=True,
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def generate_response(self, messages: list[dict], max_length: int = 512, temperature: float = 0.7, stream: bool = False, enable_thinking: bool | None = None):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        thinking_on = enable_thinking if enable_thinking is not None else self.config.enable_thinking

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_on,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            max_context = self.model.config.max_position_embeddings

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.95,
                top_k=20,
                min_p=0,
                do_sample=True,
                use_cache=True,
            )

            if stream:
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
                result = {}
                def run():
                    result["output"] = self.model.generate(**generation_kwargs, streamer=streamer)
                thread = Thread(target=run)
                thread.start()
                yield from self._stream_tokens(streamer)
                thread.join()
                yield {"tokens_used": len(result["output"][0]), "max_tokens": max_context}
                return

            outputs = self.model.generate(**generation_kwargs)
            decoded = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True,
            )
            thinking, content = self._parse_thinking(decoded)
            yield {"text": content, "thinking": thinking, "tokens_used": len(outputs[0]), "max_tokens": max_context}
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            raise

    def unload(self) -> bool:
        try:
            if self.model:
                self.model = None
                self.tokenizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error unloading model: {str(e)}")
            return False
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_huggingface.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Update `scripts/inspect_raw_response.py`**

Replace the import and instantiation lines:

```python
from src.models.backends.huggingface import HuggingFaceBackend
```

Replace `model = BaseModel(config)` with:

```python
model = HuggingFaceBackend(config)
```

Remove the old import line `from src.models.base import BaseModel`.

- [ ] **Step 7: Commit**

```bash
git add src/models/backends/__init__.py src/models/backends/huggingface.py tests/test_huggingface.py scripts/inspect_raw_response.py
git commit -m "feat: HuggingFaceBackend extracted from BaseModel"
```

---

### Task 4: LlamaCppBackend

**Files:**
- Create: `src/models/backends/llama_cpp.py`
- Create: `tests/test_llama_cpp.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_llama_cpp.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.models.backends.llama_cpp import LlamaCppBackend
from src.models.config import ModelConfig


def make_config(**kwargs):
    return ModelConfig(model_name="test", local_model_path=Path("/fake/model.gguf"), **kwargs)


def test_load_success():
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        backend = LlamaCppBackend(make_config())
        result = backend.load()
    assert result is True
    MockLlama.assert_called_once_with(
        model_path="/fake/model.gguf",
        n_gpu_layers=-1,
        n_ctx=32768,
        verbose=False,
    )


def test_load_missing_path():
    config = ModelConfig(model_name="test", local_model_path=Path("/does/not/exist.gguf"))
    backend = LlamaCppBackend(config)
    result = backend.load()
    assert result is False


def test_load_uses_config_values():
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        backend = LlamaCppBackend(make_config(n_gpu_layers=20, n_ctx=8192))
        backend.load()
    MockLlama.assert_called_once_with(
        model_path="/fake/model.gguf",
        n_gpu_layers=20,
        n_ctx=8192,
        verbose=False,
    )


def test_unload_clears_model():
    with patch("src.models.backends.llama_cpp.Llama"):
        backend = LlamaCppBackend(make_config())
        backend.load()
        result = backend.unload()
    assert result is True
    assert backend.model is None


def test_generate_raises_when_not_loaded():
    backend = LlamaCppBackend(make_config())
    with pytest.raises(RuntimeError, match="Model not loaded"):
        list(backend.generate_response([{"role": "user", "content": "hi"}]))


def test_generate_non_stream():
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        mock_model = MagicMock()
        MockLlama.return_value = mock_model
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 10},
        }
        backend = LlamaCppBackend(make_config())
        backend.load()
        results = list(backend.generate_response([{"role": "user", "content": "hi"}], stream=False))
    assert any("text" in r for r in results)
    assert any("tokens_used" in r for r in results)


def test_generate_stream():
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        mock_model = MagicMock()
        MockLlama.return_value = mock_model
        mock_model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        ])
        mock_model.n_tokens = 15
        backend = LlamaCppBackend(make_config())
        backend.load()
        results = list(backend.generate_response([{"role": "user", "content": "hi"}], stream=True))
    text_chunks = [r["text"] for r in results if "text" in r]
    assert "hello" in text_chunks
    assert " world" in text_chunks
    token_result = next(r for r in results if "tokens_used" in r)
    assert token_result["max_tokens"] == 32768
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llama_cpp.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.backends.llama_cpp'`

- [ ] **Step 3: Create `src/models/backends/llama_cpp.py`**

```python
import logging
from llama_cpp import Llama

from ..base import ModelBackend
from ..config import ModelConfig


class LlamaCppBackend(ModelBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def load(self) -> bool:
        try:
            if not self.config.local_model_path or not self.config.local_model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {self.config.local_model_path}")
            self.model = Llama(
                model_path=str(self.config.local_model_path),
                n_gpu_layers=self.config.n_gpu_layers,
                n_ctx=self.config.n_ctx,
                verbose=False,
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def unload(self) -> bool:
        try:
            if self.model:
                self.model = None
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error unloading model: {str(e)}")
            return False

    def generate_response(self, messages: list[dict], max_length: int = 512, temperature: float = 0.7, stream: bool = False, enable_thinking: bool | None = None):
        if not self.model:
            raise RuntimeError("Model not loaded")

        if stream:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_length,
                temperature=temperature,
                stream=True,
            )

            def content_tokens():
                for chunk in response:
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content

            yield from self._stream_tokens(content_tokens())
            yield {"tokens_used": self.model.n_tokens, "max_tokens": self.config.n_ctx}
            return

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_length,
            temperature=temperature,
            stream=False,
        )
        content = response["choices"][0]["message"]["content"]
        thinking, text = self._parse_thinking(content)
        tokens_used = response["usage"]["total_tokens"]
        yield {"text": text, "thinking": thinking, "tokens_used": tokens_used, "max_tokens": self.config.n_ctx}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llama_cpp.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/backends/llama_cpp.py tests/test_llama_cpp.py
git commit -m "feat: LlamaCppBackend for GGUF model inference"
```

---

### Task 5: Router backend selection

**Files:**
- Create: `tests/test_router.py`
- Modify: `src/orchestrator/router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_router.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.orchestrator.router import Router, RouterConfig
from src.models.config import ModelConfig


def test_gguf_path_selects_llama_cpp(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    config = ModelConfig(model_name="test", local_model_path=gguf_file)
    with patch("src.orchestrator.router.LlamaCppBackend") as MockLlama, \
         patch("src.orchestrator.router.HuggingFaceBackend") as MockHF:
        Router(RouterConfig(model_config=config))
    MockLlama.assert_called_once()
    MockHF.assert_not_called()


def test_non_gguf_path_selects_huggingface(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = ModelConfig(model_name="test", local_model_path=model_dir)
    with patch("src.orchestrator.router.LlamaCppBackend") as MockLlama, \
         patch("src.orchestrator.router.HuggingFaceBackend") as MockHF:
        Router(RouterConfig(model_config=config))
    MockHF.assert_called_once()
    MockLlama.assert_not_called()


def test_none_path_selects_huggingface():
    config = ModelConfig(model_name="test", use_huggingface=True, huggingface_model_id="org/model")
    with patch("src.orchestrator.router.LlamaCppBackend") as MockLlama, \
         patch("src.orchestrator.router.HuggingFaceBackend") as MockHF:
        Router(RouterConfig(model_config=config))
    MockHF.assert_called_once()
    MockLlama.assert_not_called()


def test_load_delegates_to_backend():
    config = ModelConfig(model_name="test", use_huggingface=True, huggingface_model_id="org/model")
    with patch("src.orchestrator.router.HuggingFaceBackend") as MockHF:
        mock_backend = MagicMock()
        MockHF.return_value = mock_backend
        mock_backend.load.return_value = True
        router = Router(RouterConfig(model_config=config))
        result = router.load_model()
    assert result is True
    mock_backend.load.assert_called_once()


def test_generate_delegates_to_backend():
    config = ModelConfig(model_name="test", use_huggingface=True, huggingface_model_id="org/model")
    with patch("src.orchestrator.router.HuggingFaceBackend") as MockHF:
        mock_backend = MagicMock()
        MockHF.return_value = mock_backend
        mock_backend.generate_response.return_value = iter([{"text": "hi"}])
        router = Router(RouterConfig(model_config=config))
        list(router.generate_response(messages=[{"role": "user", "content": "hello"}]))
    mock_backend.generate_response.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_router.py -v
```

Expected: FAIL — `ImportError` because Router still imports `BaseModel`

- [ ] **Step 3: Update `src/orchestrator/router.py`**

```python
from dataclasses import dataclass
import logging

from ..models.backends.huggingface import HuggingFaceBackend
from ..models.backends.llama_cpp import LlamaCppBackend
from ..models.config import ModelConfig


@dataclass
class RouterConfig:
    model_config: ModelConfig


class Router:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        path = config.model_config.local_model_path
        if path and path.suffix == ".gguf":
            self.backend = LlamaCppBackend(config.model_config)
        else:
            self.backend = HuggingFaceBackend(config.model_config)

    def load_model(self) -> bool:
        return self.backend.load()

    def unload_model(self) -> bool:
        return self.backend.unload()

    def generate_response(self, messages: list[dict], max_length: int = 512, temperature: float = 0.7, stream: bool = False, enable_thinking: bool | None = None):
        return self.backend.generate_response(
            messages=messages,
            max_length=max_length,
            temperature=temperature,
            stream=stream,
            enable_thinking=enable_thinking,
        )
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/router.py tests/test_router.py
git commit -m "feat: Router selects backend by file extension"
```

---

### Task 6: Smoke test with GGUF model

- [ ] **Step 1: Launch CLI with the GGUF model**

```bash
python scripts/chat.py --model-path "models/local/mlabonne_Qwen3-14B-abliterated-Q5_K_S.gguf" --model-name "mlabonne_Qwen3-14B-abliterated"
```

Expected output:
```
Loading model...
Model loaded successfully!

Chat CLI (type 'exit' to quit)
==================================================
```

If it fails with a llama.cpp error, check that `llama-cpp-python` is installed with CUDA support (see note below).

> **Note on CUDA support for llama.cpp:** The default `pip install llama-cpp-python` builds for CPU only. For GPU acceleration, install with:
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
> ```
> Replace `cu121` with your CUDA version (e.g. `cu118`, `cu124`). Check with `nvcc --version`.

- [ ] **Step 2: Send a test message and verify response streams**

Type: `Hello, are you working?`

Expected: A response streams to the terminal followed by a context percentage line like `[context: 2.3%]`.

- [ ] **Step 3: Verify thinking parses correctly**

Run with `--show-thinking`:
```bash
python scripts/chat.py --model-path "models/local/mlabonne_Qwen3-14B-abliterated-Q5_K_S.gguf" --model-name "mlabonne_Qwen3-14B-abliterated" --show-thinking
```

Expected: Thinking tokens appear before the response text.
