# =============================================================================
# chat.py -- Interactive Chat with Your Trained LLM
# =============================================================================
#
# USAGE:
#   python -m src.chat --checkpoint data/checkpoints/finetune/best.pt
#   python -m src.chat --checkpoint data/checkpoints/finetune/best.pt --temperature 0.8
#
# HOW IT WORKS:
#   1. Load fine-tuned model and tokenizer
#   2. Enter interactive loop
#   3. User types a message
#   4. Format as: <|user|>message<|end_turn|><|assistant|>
#   5. Model generates until <|end_turn|> or max tokens
#   6. Display response, maintain conversation history
#
# =============================================================================

import argparse

import torch

from src.model import LLM, ModelConfig
from src.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat with your LLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    # Load checkpoint
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Setup device
    device = _detect_device()
    dtype = _detect_dtype(device)

    # Create model and load weights
    model_config = ModelConfig.from_dict(config["model"])
    model = LLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Tokenizer with chat tokens
    tokenizer = Tokenizer(add_chat_tokens=True)

    print(f"\nModel: {model.param_count():,} parameters")
    print(f"Device: {device} ({dtype})")
    print("=" * 60)
    print("Chat with your LLM! Commands:")
    print("  'quit' or Ctrl+C to exit")
    print("  'clear' to reset conversation")
    print(f"  Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print("=" * 60)

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history = []
            print("[Conversation cleared]")
            continue

        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})

        # Format conversation for the model
        prompt_ids = _format_conversation(conversation_history, tokenizer)

        # Truncate if too long (keep most recent turns)
        max_prompt_len = model_config.seq_length - args.max_tokens
        while len(prompt_ids) > max_prompt_len and len(conversation_history) > 1:
            conversation_history.pop(0)
            prompt_ids = _format_conversation(conversation_history, tokenizer)

        # Generate response
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        # Extract only the new tokens (after the prompt)
        new_tokens = output_ids[0, len(prompt_ids):].tolist()

        # Stop at end_turn token or EOS
        end_turn_id = tokenizer.encode(tokenizer.SPECIAL_TOKENS["end_turn"])[0]
        if end_turn_id in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(end_turn_id)]
        if tokenizer.eos_token_id in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(tokenizer.eos_token_id)]

        response = tokenizer.decode(new_tokens).strip()
        conversation_history.append({"role": "assistant", "content": response})

        print(f"\nAssistant: {response}")


def _format_conversation(history: list, tokenizer: Tokenizer) -> list:
    """Format conversation history into token IDs for the model."""
    ids = []
    for msg in history:
        role_token = tokenizer.SPECIAL_TOKENS[msg["role"]]
        ids.extend(tokenizer.encode(role_token))
        ids.extend(tokenizer.encode(msg["content"]))
        ids.extend(tokenizer.encode(tokenizer.SPECIAL_TOKENS["end_turn"]))
    # Add the assistant prompt to start generation
    ids.extend(tokenizer.encode(tokenizer.SPECIAL_TOKENS["assistant"]))
    return ids


def _detect_device() -> torch.device:
    """Detect best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _detect_dtype(device: torch.device) -> torch.dtype:
    """Detect best dtype for inference."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif device.type == "cuda":
        return torch.float16
    return torch.float32


if __name__ == "__main__":
    main()
