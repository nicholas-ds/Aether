from dataclasses import dataclass
from typing import Optional
import torch
from pathlib import Path
from transformers import BitsAndBytesConfig


@dataclass
class ModelConfig:
    model_name: str
    local_model_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    use_huggingface: bool = False
    huggingface_model_id: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    quantization_config: Optional[BitsAndBytesConfig] = None
    enable_thinking: bool = True
    n_gpu_layers: int = -1
    n_ctx: int = 32768

    def __post_init__(self):
        if not self.use_huggingface and not self.local_model_path:
            raise ValueError("Either local_model_path must be provided or use_huggingface must be True with huggingface_model_id")
        if self.use_huggingface and not self.huggingface_model_id:
            raise ValueError("huggingface_model_id must be provided when use_huggingface is True")
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "aether" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.quantization_config is None:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
