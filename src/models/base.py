import logging
from typing import Optional, Tuple, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

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
        Prioritizes local model loading, falls back to Hugging Face if configured.
        """
        try:
            if not self.config.use_huggingface:
                return self._load_local_model()
            else:
                return self._load_huggingface_model()
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
            return False
    
    def _load_local_model(self) -> bool:
        """Load model from local path."""
        if not self.config.local_model_path or not self.config.local_model_path.exists():
            self.logger.error(f"Local model path does not exist: {self.config.local_model_path}")
            return False
            
        self.logger.info(f"Loading model from local path: {self.config.local_model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.local_model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            quantization_config = None
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.config.local_model_path),
                trust_remote_code=True,
                torch_dtype=self.config.torch_dtype,
                device_map="auto",
                local_files_only=True,
                quantization_config=quantization_config,
                max_memory={0: "14GiB"}  # Limit GPU memory usage to 14GB
            )
            
            if self.model is None or self.tokenizer is None:
                self.logger.error("Model or tokenizer failed to load")
                return False
                
            self.logger.info("Local model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading local model: {str(e)}")
            return False
    
    def _load_huggingface_model(self) -> bool:
        """Load model from Hugging Face."""
        if not self.config.huggingface_model_id:
            self.logger.error("No Hugging Face model ID provided")
            return False
            
        self.logger.info(f"Loading model from Hugging Face: {self.config.huggingface_model_id}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.huggingface_model_id,
                trust_remote_code=True
            )
            
            quantization_config = None
            if self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.huggingface_model_id,
                trust_remote_code=True,
                torch_dtype=self.config.torch_dtype,
                device_map="auto",
                quantization_config=quantization_config
            )
            
            if self.model is None or self.tokenizer is None:
                self.logger.error("Model or tokenizer failed to load")
                return False
                
            self.logger.info("Hugging Face model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Hugging Face model: {str(e)}")
            return False
    
    def get_model_and_tokenizer(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Get the loaded model and tokenizer."""
        return self.model, self.tokenizer

    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt to generate from
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Dict containing the generated text and metadata
            
        Raises:
            RuntimeError: If model or tokenizer is not loaded
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "text": response_text,
                "metadata": {
                    "temperature": temperature,
                    "max_length": max_length,
                    "model_name": self.config.model_name
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            raise
