import logging
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

logger = logging.getLogger(__name__)

def download_model(
    model_id: str,
    output_dir: Path,
    torch_dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
    load_in_8bit: bool = False
) -> bool:
    """
    Download a model from Hugging Face and save it locally.
    
    Args:
        model_id: The Hugging Face model ID (e.g., "Qwen/Qwen3-14B")
        output_dir: Directory where the model will be saved
        torch_dtype: Optional torch dtype for the model
        revision: Optional specific model revision/branch to download
        trust_remote_code: Whether to trust remote code from the model
        load_in_8bit: Whether to load the model in 8-bit precision
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model {model_id} to {output_dir}")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision
        )
        tokenizer.save_pretrained(output_dir)
        logger.info("Tokenizer downloaded successfully")
        
        # Configure quantization if enabled
        quantization_config = None
        if load_in_8bit:
            if not torch.cuda.is_available():
                logger.warning("8-bit quantization requested but CUDA not available. Falling back to full precision.")
                load_in_8bit = False
            else:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize bitsandbytes: {str(e)}. Falling back to full precision.")
                    load_in_8bit = False
                    quantization_config = None
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype or torch.float16,
            trust_remote_code=trust_remote_code,
            revision=revision,
            quantization_config=quantization_config,
            device_map="auto"  # Automatically handle device placement
        )
        model.save_pretrained(output_dir)
        logger.info("Model downloaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def verify_model_files(model_dir: Path) -> bool:
    """
    Verify that all necessary model files are present in the directory.
    
    Args:
        model_dir: Directory containing the model files
    
    Returns:
        bool: True if all required files are present, False otherwise
    """
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    try:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            logger.error(f"Model directory does not exist: {model_dir}")
            return False
            
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        
        if missing_files:
            logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
            
        logger.info("All required model files are present")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model files: {str(e)}")
        return False 