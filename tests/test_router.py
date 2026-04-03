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
