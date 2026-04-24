# =============================================================================
# evaluate.py -- Perplexity Evaluation and Sample Generation
# =============================================================================
#
# WHAT THIS DOES:
#   1. Compute perplexity on a validation set
#   2. Generate sample completions from fixed prompts
#   3. Can be run standalone or called during training
#
# USAGE:
#   python -m eval.evaluate --checkpoint data/checkpoints/pretrain/best.pt
#   python -m eval.evaluate --checkpoint data/checkpoints/pretrain/best.pt --val_data data/val.bin
#
# WHAT IS PERPLEXITY?
#   Perplexity = exp(average cross-entropy loss)
#
#   Intuition: "How many tokens is the model choosing between?"
#     - Random model: perplexity = vocab_size (50257) -- no idea!
#     - Good model: perplexity < 20 -- pretty confident
#     - Perfect model: perplexity = 1 -- always right
#
#   Lower is better. It's the standard metric for language models.
#
# =============================================================================

import argparse
import math

import torch

from src.model import LLM, ModelConfig
from src.tokenizer import Tokenizer
from src.data import PretrainDataset, create_dataloader


# Fixed prompts for qualitative evaluation
EVAL_PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy,",
    "The best way to learn programming is",
    "Once upon a time, there was a",
    "Artificial intelligence will",
    "The capital of France is",
]


def compute_perplexity(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 100,
) -> float:
    """
    Compute perplexity on a dataset.

    Returns:
        Perplexity (lower is better)
    """
    model.eval()
    losses = []
    use_amp = dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                _, loss = model(x, y)

            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return perplexity


def generate_samples(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    prompts: list = None,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> list:
    """
    Generate text completions for a list of prompts.

    Returns:
        List of (prompt, completion) tuples
    """
    if prompts is None:
        prompts = EVAL_PROMPTS

    model.eval()
    results = []
    use_amp = dtype in (torch.float16, torch.bfloat16)

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        # Extract generated part
        generated = tokenizer.decode(output_ids[0].tolist())
        results.append((prompt, generated))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="data/val.bin")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model_config = ModelConfig.from_dict(config["model"])
    model = LLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = Tokenizer()

    print(f"Model: {model.param_count():,} parameters")
    print(f"Device: {device} ({dtype})")

    # Compute perplexity
    import os
    if os.path.exists(args.val_data):
        print(f"\n{'='*60}")
        print("PERPLEXITY EVALUATION")
        print(f"{'='*60}")

        val_dataset = PretrainDataset(args.val_data, config["model"]["seq_length"])
        val_loader = create_dataloader(val_dataset, batch_size=8, num_workers=2)

        perplexity = compute_perplexity(model, val_loader, device, dtype, args.max_batches)
        print(f"\n  Perplexity: {perplexity:.2f}")
        print(f"  (Lower is better. Random = ~50257, Good < 20)")
    else:
        print(f"\nSkipping perplexity (val data not found: {args.val_data})")

    # Generate samples
    print(f"\n{'='*60}")
    print("SAMPLE GENERATIONS")
    print(f"{'='*60}")

    results = generate_samples(
        model, tokenizer, device, dtype,
        temperature=args.temperature,
    )

    for prompt, completion in results:
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {completion}")
        print(f"  {'-'*40}")


if __name__ == "__main__":
    main()
