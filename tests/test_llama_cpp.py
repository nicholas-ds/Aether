import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.models.backends.llama_cpp import LlamaCppBackend
from src.models.config import ModelConfig


def make_config(path=None, **kwargs):
    p = path or Path("/fake/model.gguf")
    return ModelConfig(model_name="test", local_model_path=p, **kwargs)


def test_load_success(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        backend = LlamaCppBackend(make_config(path=gguf_file))
        result = backend.load()
    assert result is True
    MockLlama.assert_called_once_with(
        model_path=gguf_file.as_posix(),
        n_gpu_layers=-1,
        n_ctx=16384,
        verbose=False,
    )


def test_load_missing_path():
    config = ModelConfig(model_name="test", local_model_path=Path("/does/not/exist.gguf"))
    with patch("src.models.backends.llama_cpp.Llama"):
        backend = LlamaCppBackend(config)
        result = backend.load()
    assert result is False


def test_load_uses_config_values(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        backend = LlamaCppBackend(make_config(path=gguf_file, n_gpu_layers=20, n_ctx=16384))
        backend.load()
    MockLlama.assert_called_once_with(
        model_path=gguf_file.as_posix(),
        n_gpu_layers=20,
        n_ctx=16384,
        verbose=False,
    )


def test_unload_clears_model(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    with patch("src.models.backends.llama_cpp.Llama"):
        backend = LlamaCppBackend(make_config(path=gguf_file))
        backend.load()
        result = backend.unload()
    assert result is True
    assert backend.model is None


def test_generate_raises_when_not_loaded():
    backend = LlamaCppBackend(make_config())
    with pytest.raises(RuntimeError, match="Model not loaded"):
        list(backend.generate_response([{"role": "user", "content": "hi"}]))


def test_generate_non_stream(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        mock_model = MagicMock()
        MockLlama.return_value = mock_model
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 10},
        }
        backend = LlamaCppBackend(make_config(path=gguf_file))
        backend.load()
        results = list(backend.generate_response([{"role": "user", "content": "hi"}], stream=False))
    text_result = next(r for r in results if "text" in r)
    assert text_result["text"] == "hello"
    assert text_result["tokens_used"] == 10
    assert text_result["max_tokens"] == 16384


def test_generate_stream(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()
    with patch("src.models.backends.llama_cpp.Llama") as MockLlama:
        mock_model = MagicMock()
        MockLlama.return_value = mock_model
        mock_model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        ])
        mock_model.n_tokens = 15
        backend = LlamaCppBackend(make_config(path=gguf_file))
        backend.load()
        results = list(backend.generate_response([{"role": "user", "content": "hi"}], stream=True))
    text_chunks = [r["text"] for r in results if "text" in r]
    assert "hello" in text_chunks
    assert " world" in text_chunks
    token_result = next(r for r in results if "tokens_used" in r)
    assert token_result["max_tokens"] == 16384
