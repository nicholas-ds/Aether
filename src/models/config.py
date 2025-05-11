from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    """Configuration for base model loading."""
    model_name: str
    model_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
