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
