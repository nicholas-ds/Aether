import logging
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

class BaseModel:
    """Base model manager for loading and managing the foundation model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
    
    def load(self) -> bool:
        """
        Load the specified model and its tokenizer.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading model: {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=self.config.torch_dtype,
                device_map="auto"
            )
            
            if self.model is None or self.tokenizer is None:
                self.logger.error("Model or tokenizer failed to load")
                return False
                
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
            return False
    
    def get_model_and_tokenizer(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Get the loaded model and tokenizer."""
        return self.model, self.tokenizer
