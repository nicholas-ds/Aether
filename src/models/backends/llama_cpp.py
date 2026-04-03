import os
import sys

if sys.platform == "win32":
    _cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(_cuda_base):
        for _ver in os.listdir(_cuda_base):
            for _sub in ("bin\\x64", "bin"):
                _p = os.path.join(_cuda_base, _ver, _sub)
                if os.path.isdir(_p):
                    os.add_dll_directory(_p)

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from ..base import ModelBackend
from ..config import ModelConfig


class LlamaCppBackend(ModelBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def load(self) -> bool:
        try:
            if Llama is None:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Install with: pip install llama-cpp-python"
                )
            if not self.config.local_model_path or not self.config.local_model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {self.config.local_model_path}")
            self.model = Llama(
                model_path=self.config.local_model_path.as_posix(),
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

    def generate_response(self, messages: list[dict], max_length: int = 4096, temperature: float = 0.7, stream: bool = False, enable_thinking: bool | None = None):
        if not self.model:
            raise RuntimeError("Model not loaded")
        # enable_thinking is not used: Qwen3 emits <think> tags naturally via its chat template

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
