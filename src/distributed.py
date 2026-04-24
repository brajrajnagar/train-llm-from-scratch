# =============================================================================
# distributed.py -- Hardware Auto-Detection and Distributed Training
# =============================================================================
#
# THIS FILE SOLVES: "Make it work on ANY hardware"
#
# The same code must run on:
#   3x B200 GPUs  ->  FSDP (Fully Sharded Data Parallel)
#   1x RTX 4090   ->  Single GPU (no distribution overhead)
#   Apple M3 Max  ->  MPS backend
#   Old laptop    ->  CPU (slow but functional)
#
# AUTO-DETECTION HIERARCHY:
#   +------------------------------------------------------+
#   |  Multiple CUDA GPUs?  ->  FSDP (best for multi-GPU)  |
#   |  Single CUDA GPU?     ->  Single GPU mode             |
#   |  Apple Silicon?       ->  MPS backend                 |
#   |  None of the above?   ->  CPU fallback                |
#   +------------------------------------------------------+
#
#   +------------------------------------------------------+
#   |  BF16 supported?      ->  Use BF16 (best for LLMs)   |
#   |  FP16 supported?      ->  Use FP16 (older GPUs)      |
#   |  Neither?             ->  Use FP32 (CPU/old HW)      |
#   +------------------------------------------------------+
#
# WHY FSDP INSTEAD OF DataParallel?
#   DataParallel (what learn-gpt-from-scratch uses):
#     - All GPUs hold a FULL copy of the model
#     - Gradients gathered on GPU 0 (memory bottleneck!)
#     - Python GIL limits actual parallelism
#
#   FSDP (Fully Sharded Data Parallel):
#     - Model is SHARDED across GPUs (each holds 1/N of weights)
#     - Weights gathered on-demand, then discarded
#     - Gradients reduced and sharded immediately
#     - Memory usage ~ 1/N (can train much larger models!)
#
#   Example (160M model, 3 GPUs):
#     DataParallel: Each GPU holds 160M params -> 480M total GPU memory
#     FSDP:         Each GPU holds ~53M params -> 160M total GPU memory
#
# =============================================================================

import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    """Holds all distributed training state."""
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32
    strategy: str = "cpu"      # 'fsdp', 'single_gpu', 'mps', 'cpu'
    is_main_process: bool = True

    def print(self, *args, **kwargs):
        """Print only on the main process (avoids duplicate output)."""
        if self.is_main_process:
            print(*args, **kwargs)


def setup_distributed() -> DistributedContext:
    """
    Auto-detect hardware and set up the optimal training strategy.

    When launched with torchrun:
      Environment variables RANK, WORLD_SIZE, LOCAL_RANK are set automatically.
      We initialize the NCCL process group for GPU communication.

    When launched with plain python:
      No env vars -> single device mode.
    """
    ctx = DistributedContext()

    # Check if we're in a torchrun multi-GPU launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        ctx.rank = int(os.environ["RANK"])
        ctx.local_rank = int(os.environ["LOCAL_RANK"])
        ctx.world_size = int(os.environ["WORLD_SIZE"])
        ctx.is_main_process = (ctx.rank == 0)

        # Initialize process group for GPU-to-GPU communication
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(ctx.local_rank)

        ctx.device = torch.device(f"cuda:{ctx.local_rank}")
        ctx.strategy = "fsdp"

    elif torch.cuda.is_available():
        ctx.device = torch.device("cuda")
        ctx.strategy = "single_gpu"

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        ctx.device = torch.device("mps")
        ctx.strategy = "mps"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    else:
        ctx.device = torch.device("cpu")
        ctx.strategy = "cpu"

    # Detect best dtype for this hardware
    if ctx.device.type == "cuda" and torch.cuda.is_bf16_supported():
        ctx.dtype = torch.bfloat16
    elif ctx.device.type == "cuda":
        ctx.dtype = torch.float16
    else:
        ctx.dtype = torch.float32

    ctx.print(f"Hardware detected:")
    ctx.print(f"  Strategy:   {ctx.strategy}")
    ctx.print(f"  World size: {ctx.world_size}")
    ctx.print(f"  Dtype:      {ctx.dtype}")
    ctx.print(f"  Device:     {ctx.device}")
    if ctx.device.type == "cuda":
        ctx.print(f"  GPU:        {torch.cuda.get_device_name(ctx.device)}")

    return ctx


def wrap_model_distributed(
    model: torch.nn.Module,
    ctx: DistributedContext,
    gradient_checkpointing: bool = False,
    compile_model: bool = True,
) -> torch.nn.Module:
    """
    Wrap the model with the appropriate distributed strategy.
    Returns the wrapped model.
    """
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        _enable_gradient_checkpointing(model)
        ctx.print("Gradient checkpointing: ENABLED")

    if ctx.strategy == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial
        from src.model import TransformerBlock

        # Auto-wrap policy: each TransformerBlock is a separate FSDP unit
        # This gives FSDP the right granularity for sharding
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )

        # Mixed precision: compute in BF16/FP16, communicate in same dtype
        mp_policy = MixedPrecision(
            param_dtype=ctx.dtype,
            reduce_dtype=ctx.dtype,
            buffer_dtype=ctx.dtype,
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=ctx.local_rank,
            use_orig_params=True,  # Needed for torch.compile compatibility
        )
        ctx.print(f"Model wrapped with FSDP (sharding across {ctx.world_size} GPUs)")

    elif ctx.strategy in ("single_gpu", "mps"):
        model = model.to(ctx.device)

    else:  # cpu
        model = model.to(ctx.device)

    # Try torch.compile (significant speedup on CUDA)
    if compile_model and ctx.device.type == "cuda":
        try:
            model = torch.compile(model)
            ctx.print("torch.compile: ENABLED")
        except Exception as e:
            ctx.print(f"torch.compile: SKIPPED ({e})")
    else:
        ctx.print(f"torch.compile: SKIPPED (not supported on {ctx.device.type})")

    return model


def _enable_gradient_checkpointing(model: torch.nn.Module):
    """
    Enable gradient checkpointing on transformer blocks.

    WHY GRADIENT CHECKPOINTING?
      Normal: Save ALL intermediate activations -> fast but uses lots of memory
      Checkpointed: Save only block boundaries -> recompute intermediates in backward
      Trade-off: ~30% slower training, but ~40% less memory

      Useful when: model is too large to fit in GPU memory with full activations
    """
    from torch.utils.checkpoint import checkpoint
    from src.model import TransformerBlock

    for block in model.blocks:
        original_forward = block.forward

        def make_checkpointed(orig_fn):
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(orig_fn, *args, use_reentrant=False, **kwargs)
            return checkpointed_forward

        block.forward = make_checkpointed(original_forward)


def save_checkpoint_distributed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: dict,
    path: str,
    ctx: DistributedContext,
):
    """
    Save checkpoint, handling FSDP state dict gathering.

    With FSDP, model weights are sharded across GPUs.
    We need to gather them to rank 0 before saving (only rank 0 saves).
    """
    if ctx.strategy == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # Gather full state dict to rank 0 (offload to CPU to save GPU memory)
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)

        if ctx.is_main_process:
            checkpoint = {
                "step": step,
                "model_state_dict": model_state,
                "optimizer_state_dict": optim_state,
                "val_loss": val_loss,
                "config": config,
            }
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path} (step {step})")

    else:
        # Single device: straightforward save
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path} (step {step})")


def load_checkpoint_distributed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    ctx: DistributedContext,
) -> tuple:
    """
    Load checkpoint, handling FSDP state dict distribution.

    Returns (start_step, val_loss).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if ctx.strategy == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # Load full state dict into FSDP model (automatically shards)
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optim_state = FSDP.optim_state_dict_to_load(
                    model, optimizer, checkpoint["optimizer_state_dict"]
                )
                optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_step = checkpoint.get("step", 0) + 1
    val_loss = checkpoint.get("val_loss", float("inf"))

    return start_step, val_loss


def cleanup_distributed(ctx: DistributedContext):
    """Clean up distributed process group."""
    if ctx.strategy == "fsdp" and dist.is_initialized():
        dist.destroy_process_group()
