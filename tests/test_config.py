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


def test_enable_thinking_default():
    assert make_config().enable_thinking is True
