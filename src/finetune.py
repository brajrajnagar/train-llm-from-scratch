# =============================================================================
# finetune.py -- Instruction Fine-tuning Entry Point
# =============================================================================
#
# WHY FINE-TUNE?
#   After pretraining, the model can predict next tokens but doesn't know
#   how to follow instructions or have conversations.
#
#   Pretraining teaches: "What word comes next?"
#   Fine-tuning teaches:  "How to be a helpful chatbot"
#
# KEY DIFFERENCES FROM PRETRAINING:
#   1. Lower learning rate (2e-5 vs 6e-4) -- don't destroy pretrained knowledge
#   2. Fewer steps (1000-3000 vs 50000) -- small dataset, fast convergence
#   3. Loss mask -- only train on ASSISTANT responses, not user prompts
#   4. Special tokens -- <|user|>, <|assistant|>, <|end_turn|> added to vocab
#
# USAGE:
#   Single GPU:
#     python -m src.finetune --config configs/160m.yaml \
#       --checkpoint data/checkpoints/pretrain/best.pt
#
#   Multi-GPU:
#     torchrun --nproc_per_node=3 -m src.finetune --config configs/160m.yaml \
#       --checkpoint data/checkpoints/pretrain/best.pt
#
# =============================================================================

import argparse

import yaml
import torch

from src.model import LLM, ModelConfig
from src.tokenizer import Tokenizer
from src.data import FinetuneDataset, create_dataloader
from src.distributed import (
    setup_distributed,
    wrap_model_distributed,
    cleanup_distributed,
)
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for chat")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config (same model architecture as pretraining)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--finetune_data", type=str, default="data/finetune.jsonl",
                        help="Path to fine-tuning data (JSONL)")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()

    # 1. Load config from YAML (same model architecture as pretraining)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override training params for fine-tuning
    # (Lower LR, fewer steps, smaller batch -- protect pretrained knowledge)
    config["training"]["max_steps"] = args.max_steps
    config["training"]["learning_rate"] = args.learning_rate
    config["training"]["min_lr"] = args.learning_rate * 0.1
    config["training"]["warmup_steps"] = args.warmup_steps
    config["training"]["grad_accum_steps"] = 2
    config["training"]["batch_size"] = 4
    config["checkpoint_dir"] = "data/checkpoints/finetune"
    config["eval_interval"] = 200
    config["save_interval"] = 500

    # 2. Setup distributed
    ctx = setup_distributed()

    # 3. Create tokenizer with chat tokens
    tokenizer = Tokenizer(add_chat_tokens=True)
    ctx.print(f"\nTokenizer: {tokenizer.vocab_size} tokens "
              f"(+{tokenizer.vocab_size - 50257} chat tokens)")

    # 4. Create model and load pretrained weights
    model_config = ModelConfig.from_dict(config["model"])
    model = LLM(model_config)

    # Load pretrained checkpoint (on CPU first, then move to device)
    ctx.print(f"Loading pretrained checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Resize embedding for new special tokens (chat format tokens)
    if tokenizer.vocab_size > model_config.vocab_size:
        old_vocab = model_config.vocab_size
        new_vocab = tokenizer.vocab_size
        model.resize_token_embeddings(new_vocab)
        ctx.print(f"Resized embeddings: {old_vocab} -> {new_vocab}")

    ctx.print(f"Model: {model.param_count():,} parameters")

    # 5. Wrap model (FSDP/compile)
    gradient_ckpt = config.get("gradient_checkpointing", False)
    compile_model = config.get("compile", True)
    model = wrap_model_distributed(model, ctx, gradient_ckpt, compile_model)

    # 6. Optimizer (lower LR for fine-tuning, less weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,  # Lower than pretraining (0.1)
        fused=(ctx.device.type == "cuda"),
    )

    # 7. Dataset
    train_dataset = FinetuneDataset(
        args.finetune_data,
        config["model"]["seq_length"],
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        distributed=(ctx.strategy == "fsdp"),
    )

    # 8. Train (no validation set for fine-tuning -- dataset is small)
    ctx.print(f"\nStarting fine-tuning...")
    ctx.print(f"  Max steps: {args.max_steps}")
    ctx.print(f"  Learning rate: {args.learning_rate}")
    ctx.print(f"  Training examples: {len(train_dataset)}")
    ctx.print("")

    trainer = Trainer(model, optimizer, train_loader, None, config, ctx)
    trainer.train()

    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
