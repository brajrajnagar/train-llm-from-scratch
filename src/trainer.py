# =============================================================================
# trainer.py -- Training Loop, LR Schedule, Logging
# =============================================================================
#
# THE TRAINING LOOP AT SCALE:
#
#   For each step:
#     +------------------------------------------------------+
#     |  1. GET BATCH           From memory-mapped .bin       |
#     |  2. FORWARD PASS        model(x) -> logits, loss      |
#     |  3. SCALE LOSS          loss / grad_accum_steps        |
#     |  4. BACKWARD PASS       loss.backward()                |
#     |  5. ACCUMULATE?         Repeat 1-4 for N micro-steps   |
#     |  6. CLIP GRADIENTS      Prevent instability            |
#     |  7. OPTIMIZER STEP      Update weights                 |
#     |  8. LR SCHEDULE         Cosine with warmup             |
#     |  9. LOG METRICS         Loss, LR, tokens/sec           |
#     |  10. CHECKPOINT         Save periodically              |
#     +------------------------------------------------------+
#
# GRADIENT ACCUMULATION (new concept from learn-gpt-from-scratch):
#
#   Problem: We want a large effective batch size (e.g., 96 sequences),
#            but we can only fit 8 sequences in GPU memory at once.
#
#   Solution: Process 8 sequences, accumulate gradients, repeat 12 times,
#             THEN update weights. Effect is the same as batch_size=96!
#
#     Micro-step 1:  forward(8 seqs) -> backward -> accumulate grads
#     Micro-step 2:  forward(8 seqs) -> backward -> accumulate grads
#     ...
#     Micro-step 12: forward(8 seqs) -> backward -> accumulate grads
#     -> clip gradients -> optimizer.step() -> zero grads
#
# COSINE LR SCHEDULE WITH WARMUP:
#
#   LR |  /\
#      | /  \_________
#      |/             \_____
#      |                    \___
#      +------------------------ steps
#      warmup    cosine decay
#
#   1. Warmup: LR ramps linearly from 0 to peak (prevents early instability)
#   2. Cosine decay: LR smoothly decreases to min_lr
#   This is the standard schedule used by GPT-3, Llama, etc.
#
# =============================================================================

import math
import os
import time

import torch
from tqdm import tqdm

from src.distributed import save_checkpoint_distributed
from src.tokenizer import Tokenizer


class Trainer:
    """
    Training loop with gradient accumulation, mixed precision,
    cosine LR schedule, logging, and checkpointing.
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        train_dataloader,
        val_dataloader,
        config: dict,
        ctx,  # DistributedContext
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.ctx = ctx

        # Training config
        tc = config["training"]
        self.max_steps = tc["max_steps"]
        self.grad_accum_steps = tc["grad_accum_steps"]
        self.grad_clip = tc["grad_clip"]
        self.learning_rate = tc["learning_rate"]
        self.min_lr = tc["min_lr"]
        self.warmup_steps = tc["warmup_steps"]

        # Logging/checkpointing config
        self.log_interval = config.get("log_interval", 10)
        self.eval_interval = config.get("eval_interval", 500)
        self.eval_iters = config.get("eval_iters", 100)
        self.save_interval = config.get("save_interval", 2000)
        self.checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")

        # Mixed precision:
        #   BF16: no scaler needed (enough dynamic range)
        #   FP16: GradScaler prevents underflow
        #   FP32: no AMP at all
        self.use_amp = ctx.dtype in (torch.float16, torch.bfloat16)
        self.scaler = torch.amp.GradScaler(
            device=ctx.device.type,
            enabled=(ctx.dtype == torch.float16),
        )

        # Metrics
        self.step = 0
        self.best_val_loss = float("inf")
        self.tokens_processed = 0
        self.loss_history = []  # Recent losses for trend tracking

        # For sample generation during eval
        self.tokenizer = Tokenizer()
        self.sample_prompts = [
            "The meaning of life is",
            "In a distant galaxy, scientists discovered",
        ]

        # Training start time
        self.t_start = None

    def get_lr(self, step: int) -> float:
        """
        Cosine learning rate schedule with linear warmup.

        Step < warmup_steps:  LR ramps linearly from 0 -> learning_rate
        Step >= warmup_steps: LR decays via cosine from learning_rate -> min_lr
        """
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps

        if step >= self.max_steps:
            return self.min_lr

        # Cosine decay: smoothly decrease from peak to min
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.learning_rate - self.min_lr) * cosine_decay

    def train(self, start_step: int = 0):
        """Main training loop."""
        self.step = start_step
        self.model.train()
        self.t_start = time.time()

        train_iter = iter(self.train_dataloader)
        seq_length = self.config["model"]["seq_length"]
        batch_size = self.config["training"]["batch_size"]
        tokens_per_step = batch_size * seq_length * self.grad_accum_steps * self.ctx.world_size
        total_tokens_target = tokens_per_step * self.max_steps

        pbar = tqdm(
            range(start_step, self.max_steps),
            desc="Training",
            unit="step",
            disable=not self.ctx.is_main_process,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}",
            ncols=120,
        )

        for step in pbar:
            self.step = step
            t0 = time.time()

            # Update learning rate for this step
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # === Gradient accumulation loop ===
            total_loss = 0.0
            for micro_step in range(self.grad_accum_steps):
                # Get next batch (reset iterator at end of epoch)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    if hasattr(self.train_dataloader, "sampler") and \
                       hasattr(self.train_dataloader.sampler, "set_epoch"):
                        self.train_dataloader.sampler.set_epoch(step)
                    train_iter = iter(self.train_dataloader)
                    batch = next(train_iter)

                x, y = batch
                x = x.to(self.ctx.device)
                y = y.to(self.ctx.device)

                # Forward pass with mixed precision
                with torch.autocast(
                    device_type=self.ctx.device.type,
                    dtype=self.ctx.dtype,
                    enabled=self.use_amp,
                ):
                    _, loss = self.model(x, y)
                    loss = loss / self.grad_accum_steps  # Scale for accumulation

                total_loss += loss.item()

                # Backward pass (scaler handles FP16, no-op for BF16/FP32)
                self.scaler.scale(loss).backward()

            # === End of accumulation: clip, step, zero ===

            # Gradient clipping (prevents exploding gradients)
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            else:
                grad_norm = None

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            # === Metrics ===
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = tokens_per_step / dt
            self.tokens_processed += tokens_per_step

            # Track loss history for trend
            self.loss_history.append(total_loss)
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)

            # Logging
            if step % self.log_interval == 0:
                # Compute loss trend (avg of last 50 vs last 10)
                avg_recent = sum(self.loss_history[-10:]) / len(self.loss_history[-10:])
                if len(self.loss_history) >= 50:
                    avg_older = sum(self.loss_history[-50:-10]) / 40
                    trend = "↓" if avg_recent < avg_older else "↑" if avg_recent > avg_older * 1.01 else "→"
                else:
                    trend = ""

                pct_tokens = self.tokens_processed / total_tokens_target * 100
                elapsed = t1 - self.t_start
                eta_str = self._format_time((elapsed / max(step - start_step, 1)) * (self.max_steps - step)) if step > start_step else "..."

                pbar.set_postfix_str(
                    f"loss={total_loss:.4f}{trend} | lr={lr:.1e} | "
                    f"{tokens_per_sec/1000:.0f}K tok/s | "
                    f"{pct_tokens:.1f}% data | ETA {eta_str}"
                )
                if self.ctx.is_main_process:
                    self._log_step(step, total_loss, lr, tokens_per_sec, grad_norm, dt)

            # Evaluation
            if step > 0 and step % self.eval_interval == 0:
                val_loss = self.evaluate()
                perplexity = math.exp(min(val_loss, 20))  # Cap to avoid overflow
                elapsed = time.time() - self.t_start if self.t_start else 0

                eval_msg = (
                    f"\n{'='*60}\n"
                    f"  EVAL @ step {step:,} | elapsed {self._format_time(elapsed)}\n"
                    f"  Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}\n"
                    f"  Train Loss (avg last 10): {sum(self.loss_history[-10:])/len(self.loss_history[-10:]):.4f}\n"
                    f"  Best Val Loss: {min(self.best_val_loss, val_loss):.4f}\n"
                    f"  Tokens Processed: {self.tokens_processed:,}\n"
                    f"{'='*60}"
                )
                self.ctx.print(eval_msg)

                # Write eval to log file too
                if self.ctx.is_main_process:
                    log_path = os.path.join(self.checkpoint_dir, "train.log")
                    with open(log_path, "a") as f:
                        f.write(eval_msg + "\n")

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best.pt", step, val_loss)
                    self.ctx.print(f"  *** New best val loss: {val_loss:.4f} ***")

                # Generate samples so you can eyeball quality in the logs
                self._generate_samples(step)

                self.model.train()

            # Periodic checkpoint
            if step > 0 and step % self.save_interval == 0:
                self._save("latest.pt", step, total_loss)

        # Final save
        self._save("latest.pt", self.step, total_loss)
        self.ctx.print("\nTraining complete!")
        self.ctx.print(f"  Total tokens processed: {self.tokens_processed:,}")
        self.ctx.print(f"  Best val loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute average validation loss."""
        self.model.eval()
        losses = []

        if self.val_dataloader is None:
            return float("inf")

        val_iter = iter(self.val_dataloader)

        for _ in range(self.eval_iters):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            x, y = batch
            x = x.to(self.ctx.device)
            y = y.to(self.ctx.device)

            with torch.autocast(
                device_type=self.ctx.device.type,
                dtype=self.ctx.dtype,
                enabled=self.use_amp,
            ):
                _, loss = self.model(x, y)

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses) if losses else float("inf")

        # Average across all processes in distributed training
        if self.ctx.strategy == "fsdp" and torch.distributed.is_initialized():
            loss_tensor = torch.tensor(avg_loss, device=self.ctx.device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    def _save(self, filename: str, step: int, val_loss: float):
        """Save checkpoint via distributed-aware function."""
        path = os.path.join(self.checkpoint_dir, filename)
        save_checkpoint_distributed(
            self.model, self.optimizer, step, val_loss, self.config, path, self.ctx
        )

    @staticmethod
    def _format_time(seconds):
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"

    def _log_step(self, step, loss, lr, tokens_per_sec, grad_norm, dt):
        """Log training metrics with visual loss bar."""
        grad_str = f"{grad_norm:.4f}" if grad_norm is not None else "N/A"

        # Visual loss bar: maps loss 0-12 to a 20-char bar
        bar_width = 20
        loss_clamped = max(0, min(loss, 12))
        filled = int((1 - loss_clamped / 12) * bar_width)
        loss_bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = time.time() - self.t_start if self.t_start else 0

        log_line = (
            f"step={step:>6d} | loss={loss:.4f} [{loss_bar}] | lr={lr:.2e} | "
            f"grad_norm={grad_str} | {tokens_per_sec/1000:.0f}K tok/s | "
            f"dt={dt:.2f}s | tok={self.tokens_processed:,} | "
            f"elapsed={self._format_time(elapsed)}"
        )
        # Write to log file
        log_path = os.path.join(self.checkpoint_dir, "train.log")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

    @torch.no_grad()
    def _generate_samples(self, step: int):
        """
        Generate sample completions after eval so you can eyeball quality.

        All ranks must participate in forward passes (FSDP requirement),
        but only rank 0 logs the results.
        """
        self.model.eval()
        seq_length = self.config["model"]["seq_length"]

        lines = [f"\n--- Sample generations at step {step} ---"]
        for prompt in self.sample_prompts:
            token_ids = self.tokenizer.encode(prompt)
            idx = torch.tensor([token_ids], dtype=torch.long, device=self.ctx.device)

            for _ in range(100):  # max_new_tokens
                idx_cond = idx[:, -seq_length:]
                with torch.autocast(
                    device_type=self.ctx.device.type,
                    dtype=self.ctx.dtype,
                    enabled=self.use_amp,
                ):
                    logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / 0.8  # temperature
                probs = torch.softmax(logits, dim=-1)
                # top-k 40
                topk_probs, topk_indices = torch.topk(probs, 40, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                next_token = torch.gather(topk_indices, -1, ix)
                idx = torch.cat([idx, next_token], dim=1)

            if self.ctx.is_main_process:
                text = self.tokenizer.decode(idx[0].tolist())
                lines.append(f"  PROMPT: {prompt}")
                lines.append(f"  OUTPUT: {text}")
                lines.append("")

        if self.ctx.is_main_process:
            lines.append("--- End samples ---\n")
            sample_text = "\n".join(lines)
            self.ctx.print(sample_text)

            log_path = os.path.join(self.checkpoint_dir, "train.log")
            with open(log_path, "a") as f:
                f.write(sample_text + "\n")
