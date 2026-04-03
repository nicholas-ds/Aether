import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import ModelConfig
from src.orchestrator.router import Router, RouterConfig


SYSTEM_PROMPT = (
    "You are Aether, an expert teacher and mentor. "
    "Think deeply before responding. "
    "Keep responses short and conversational — like two people talking, not a lecture. "
    "Be encouraging but honest. Don't tell the user what they want to hear if it's wrong. "
    "If you don't know something, say so. Don't guess."
)


def append_user_message(messages: list[dict], content: str) -> list[dict]:
    return messages + [{"role": "user", "content": content}]


def append_assistant_message(messages: list[dict], content: str) -> list[dict]:
    return messages + [{"role": "assistant", "content": content}]


def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM")
    parser.add_argument("--model-name", default="mlabonne_Qwen3-14B-abliterated-Q5_K_S.gguf")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--show-thinking", action="store_true", default=False)
    parser.add_argument("--thinking", action="store_true", default=False)
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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        max_length = 16384 if args.thinking else 4096

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            messages = append_user_message(messages, user_input)

            try:
                print("\nAether: ", end="", flush=True)
                response_text = ""
                tokens_used = None
                max_tokens = None
                for chunk in router.generate_response(messages=messages, stream=True, enable_thinking=args.thinking, max_length=max_length):
                    if "text" in chunk:
                        print(chunk["text"], end="", flush=True)
                        response_text += chunk["text"]
                    elif "thinking" in chunk:
                        if args.show_thinking:
                            print(chunk["thinking"], end="", flush=True)
                    elif "tokens_used" in chunk:
                        tokens_used = chunk["tokens_used"]
                        max_tokens = chunk["max_tokens"]
                print()
                if tokens_used is not None:
                    print(f"[context: {round(tokens_used / max_tokens * 100, 1)}%]")
                messages = append_assistant_message(messages, response_text)
            except Exception as e:
                messages = messages[:-1]
                print(f"\nError: {e}")
    finally:
        print("\nUnloading model...")
        router.unload_model()
        print("Goodbye!")


if __name__ == "__main__":
    main()
