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
