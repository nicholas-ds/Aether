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
