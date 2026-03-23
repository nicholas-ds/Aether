import logging
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

from .config import ModelConfig


class BaseModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)

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

    def generate_response(self, messages: list[dict], max_length: int = 512, temperature: float = 0.7, stream: bool = False) -> Dict[str, Any]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

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
                thread = Thread(target=self.model.generate, kwargs={**generation_kwargs, "streamer": streamer})
                thread.start()
                for token in streamer:
                    yield {"text": token}
                thread.join()
                return

            outputs = self.model.generate(**generation_kwargs)
            decoded = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True,
            )

            if "</think>" in decoded:
                thinking, content = decoded.split("</think>", 1)
                thinking = thinking.replace("<think>", "").strip()
                content = content.strip()
            else:
                thinking = ""
                content = decoded.strip()

            return {"text": content, "thinking": thinking}
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
