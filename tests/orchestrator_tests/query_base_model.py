import sys
import os
from pathlib import Path
import torch
import gc

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Use absolute imports
from models.config import ModelConfig
from models.base import BaseModel

# Check GPU availability first
if not torch.cuda.is_available():
    print("ERROR: No GPU available. This model requires GPU to run.")
    sys.exit(1)

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent.parent

# Create model configuration pointing to our downloaded Qwen model
model_path = project_root / "models" / "local" / "Qwen3-14B"
print(f"Looking for model files in: {model_path}")
print(f"Directory exists: {model_path.exists()}")
if model_path.exists():
    print("Files in directory:")
    for file in model_path.iterdir():
        print(f"  - {file.name}")

# Configure the model with explicit GPU settings
model_config = ModelConfig(
    model_name="Qwen3-14B",
    local_model_path=model_path,
    use_huggingface=False,  # Use local model
    torch_dtype=torch.float16,  # Use float16 for better memory efficiency
    load_in_4bit=True,  # Enable 4-bit quantization
    device="cuda"  # Force GPU usage
)

# Clear GPU cache before loading
torch.cuda.empty_cache()
print("\nCleared GPU cache")

# Load the model
print("\nAttempting to load model...")
base_model = BaseModel(model_config)
if not base_model.load():
    print("Failed to load model")
    sys.exit(1)

print("Model loaded successfully!")
print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Test basic generation
print("\nTesting basic generation:")
try:
    response = base_model.generate_response(
        "Hello, how are you?",
        max_length=200,
        temperature=0.7
    )
    print(f"Response: {response['text']}")
except Exception as e:
    print(f"Error during generation: {str(e)}")

# Test with a more complex prompt
print("\nTesting with a more complex prompt:")
try:
    response = base_model.generate_response(
        "Write a short poem about artificial intelligence",
        max_length=300,
        temperature=0.8
    )
    print(f"Response: {response['text']}")
except Exception as e:
    print(f"Error during generation: {str(e)}")

# Print final GPU memory usage
print(f"\nFinal GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Clean up
print("\nCleaning up...")
del base_model
torch.cuda.empty_cache()
gc.collect()
print("Cleanup complete")
