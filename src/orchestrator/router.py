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
