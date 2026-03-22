import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import ModelConfig
from src.orchestrator.router import Router, RouterConfig


def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM")
    parser.add_argument("--model-name", default="Qwen3-14B")
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    model_path = args.model_path or (PROJECT_ROOT / "models" / "local" / args.model_name)

    config = ModelConfig(
        model_name=args.model_name,
        local_model_path=model_path,
    )
    router = Router(RouterConfig(model_config=config))

    print("Loading model...")
    if not router.load_model():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    print("Model loaded successfully!")

    try:
        print("\nChat CLI (type 'exit' to quit)")
        print("=" * 50)

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            try:
                print("\nAssistant: ", end="", flush=True)
                for token in router.generate_response(prompt=user_input, stream=True):
                    print(token["text"], end="", flush=True)
                print()
            except Exception as e:
                print(f"\nError: {e}")
    finally:
        print("\nUnloading model...")
        router.unload_model()
        print("Goodbye!")


if __name__ == "__main__":
    main()
