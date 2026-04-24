# =============================================================================
# pretrain.py -- Pretraining Entry Point
# =============================================================================
#
# USAGE:
#   Single GPU:  python -m src.pretrain --config configs/160m.yaml
#   Multi-GPU:   torchrun --nproc_per_node=3 -m src.pretrain --config configs/160m.yaml
#   Resume:      [same command] --resume data/checkpoints/pretrain/latest.pt
#
# WHAT HAPPENS:
#   1. Load config from YAML
#   2. Auto-detect hardware (GPU count, BF16 support, etc.)
#   3. Create model, optimizer, dataloaders
#   4. Train with cosine LR schedule, gradient accumulation, FSDP
#   5. Save checkpoints periodically
#
# =============================================================================

import argparse

import yaml
import torch

from src.model import LLM, ModelConfig
from src.data import PretrainDataset, create_dataloader
from src.distributed import (
    setup_distributed,
    wrap_model_distributed,
    cleanup_distributed,
    load_checkpoint_distributed,
)
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain LLM from scratch")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    # 1. Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 2. Setup distributed (auto-detects hardware)
    ctx = setup_distributed()

    # 3. Create model
    model_config = ModelConfig.from_dict(config["model"])
    model = LLM(model_config)
    ctx.print(f"\nModel created: {model.param_count():,} parameters")
    ctx.print(f"  d_model={model_config.d_model}, n_heads={model_config.n_heads}, "
              f"n_kv_heads={model_config.n_kv_heads}, n_blocks={model_config.n_blocks}, "
              f"d_ff={model_config.d_ff}, seq_length={model_config.seq_length}")

    # 4. Wrap with FSDP/compile (moves model to device)
    gradient_ckpt = config.get("gradient_checkpointing", False)
    compile_model = config.get("compile", True)
    model = wrap_model_distributed(model, ctx, gradient_ckpt, compile_model)

    # 5. Create optimizer
    #    AdamW with fused=True is faster on CUDA (fuses parameter updates)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
        weight_decay=config["training"]["weight_decay"],
        fused=(ctx.device.type == "cuda"),
    )

    # 6. Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint_distributed(model, optimizer, args.resume, ctx)
        ctx.print(f"Resumed from step {start_step}")

    # 7. Create datasets and dataloaders
    ctx.print(f"\nLoading data...")
    train_dataset = PretrainDataset(
        config["data"]["train_path"],
        config["model"]["seq_length"],
    )
    val_dataset = PretrainDataset(
        config["data"]["val_path"],
        config["model"]["seq_length"],
    )
    ctx.print(f"  Train: {len(train_dataset):,} examples")
    ctx.print(f"  Val:   {len(val_dataset):,} examples")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        distributed=(ctx.strategy == "fsdp"),
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        distributed=(ctx.strategy == "fsdp"),
    )

    # 8. Train!
    ctx.print(f"\nStarting pretraining...")
    ctx.print(f"  Max steps: {config['training']['max_steps']}")
    ctx.print(f"  Effective batch size: "
              f"{config['training']['batch_size']} * {ctx.world_size} * "
              f"{config['training']['grad_accum_steps']} = "
              f"{config['training']['batch_size'] * ctx.world_size * config['training']['grad_accum_steps']}")
    ctx.print(f"  Tokens per step: "
              f"{config['training']['batch_size'] * ctx.world_size * config['training']['grad_accum_steps'] * config['model']['seq_length']:,}")
    ctx.print("")

    trainer = Trainer(model, optimizer, train_loader, val_loader, config, ctx)
    trainer.train(start_step=start_step)

    # 9. Cleanup
    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
