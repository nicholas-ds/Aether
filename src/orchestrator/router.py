from dataclasses import dataclass
import logging

from ..models.base import BaseModel
from ..models.config import ModelConfig


@dataclass
class RouterConfig:
    model_config: ModelConfig


class Router:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.base_model = BaseModel(config.model_config)
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> bool:
        return self.base_model.load()

    def unload_model(self) -> bool:
        return self.base_model.unload()

    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7, stream: bool = False):
        return self.base_model.generate_response(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            stream=stream
        )
