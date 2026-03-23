import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import ModelConfig
from src.models.base import BaseModel

MODEL_NAME = "Qwen3-14B"
PROMPT = "What is 2 + 2?"

config = ModelConfig(
    model_name=MODEL_NAME,
    local_model_path=PROJECT_ROOT / "models" / "local" / MODEL_NAME,
)
model = BaseModel(config)

print("Loading model...")
model.load()
print("Done.\n")

messages = [{"role": "user", "content": PROMPT}]

for enable_thinking in (True, False):
    print(f"{'='*60}")
    print(f"enable_thinking={enable_thinking}")
    print(f"{'='*60}")

    text = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = model.tokenizer(text, return_tensors="pt").to(model.model.device)

    outputs = model.model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        do_sample=True,
    )

    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

    print("\n--- raw token IDs ---")
    print(output_ids)

    print("\n--- decoded (skip_special_tokens=False) ---")
    print(model.tokenizer.decode(output_ids, skip_special_tokens=False))

    print("\n--- decoded (skip_special_tokens=True) ---")
    print(model.tokenizer.decode(output_ids, skip_special_tokens=True))
    print()

model.unload()
