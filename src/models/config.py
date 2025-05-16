from dataclasses import dataclass
from typing import Optional
import torch
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for base model loading."""
    # Model identification
    model_name: str  # Name of the model (e.g., "Yi-13B", "Deepseek-13B")
    
    local_model_path: Optional[Path] = None  # Path to local model files
    
    use_huggingface: bool = False  # Whether to use Hugging Face instead of local model
    huggingface_model_id: Optional[str] = None  # Hugging Face model ID if using HF
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_in_4bit: bool = True  # Changed from load_in_8bit to load_in_4bit
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.use_huggingface and not self.local_model_path:
            raise ValueError("Either local_model_path must be provided or use_huggingface must be True with huggingface_model_id")
        
        if self.use_huggingface and not self.huggingface_model_id:
            raise ValueError("huggingface_model_id must be provided when use_huggingface is True")
