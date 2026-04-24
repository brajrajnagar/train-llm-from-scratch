#!/usr/bin/env python3
# =============================================================================
# run.py -- Your Starting Point
# =============================================================================
#
# New here? Just run:
#
#   python run.py
#
# This will guide you through every step of training your own LLM.
#
# =============================================================================

import os
import sys
import subprocess

# Use the same Python that's running this script
PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def print_header():
    print()
    print("=" * 60)
    print("  Train Your Own LLM From Scratch")
    print("=" * 60)
    print()
    print("  This tool walks you through the entire pipeline:")
    print()
    print("    Step 1: Download training data")
    print("    Step 2: Prepare data (tokenize)")
    print("    Step 3: Pretrain the model")
    print("    Step 4: Fine-tune for chat")
    print("    Step 5: Chat with your model")
    print("    Step 6: Evaluate your model")
    print()


def print_status():
    """Show what's been completed so far."""
    checks = {
        "Data downloaded": os.path.exists("data/raw"),
        "Data prepared": os.path.exists("data/train.bin") and os.path.exists("data/val.bin"),
        "Model pretrained": os.path.exists("data/checkpoints/pretrain/best.pt"),
        "Model fine-tuned": os.path.exists("data/checkpoints/finetune/best.pt"),
    }

    print("  Progress:")
    for label, done in checks.items():
        status = "done" if done else "not yet"
        marker = "[x]" if done else "[ ]"
        print(f"    {marker} {label} ({status})")
    print()


def detect_gpus():
    """Detect available GPUs."""
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            name = torch.cuda.get_device_name(0)
            return count, name
    except ImportError:
        pass
    return 0, None


def pick_config():
    """Let user choose a model config."""
    gpu_count, gpu_name = detect_gpus()

    print()
    print("  Choose a model size:")
    print()
    print("    [1]  50M params  -- Quick test, ~30 min on GPU")
    print("                        (Start here to verify everything works)")
    print()
    print("    [2] 160M params  -- Sweet spot, ~4 hours on multi-GPU")
    print("                        (Good enough for a basic chatbot)")
    print()
    print("    [3] 400M params  -- Better quality, ~24-48 hours on multi-GPU")
    print("                        (Only if you have the compute budget)")
    print()

    if gpu_count > 0:
        print(f"  Your hardware: {gpu_count}x {gpu_name}")
    else:
        print("  Your hardware: CPU only (training will be very slow)")
    print()

    choice = input("  Your choice [1/2/3] (default: 1): ").strip()

    configs = {"1": "configs/50m.yaml", "2": "configs/160m.yaml", "3": "configs/400m.yaml"}
    return configs.get(choice, "configs/50m.yaml")


def run_command(cmd, description):
    """Run a shell command, showing output in real time."""
    print()
    print(f"  Running: {description}")
    print(f"  Command: {cmd}")
    print("-" * 60)
    result = subprocess.run(cmd, shell=True)
    print("-" * 60)
    if result.returncode != 0:
        print(f"  Command failed with exit code {result.returncode}")
        return False
    return True


def step_download():
    """Step 1: Download training data."""
    print()
    print("=" * 60)
    print("  Step 1: Download Training Data")
    print("=" * 60)
    print()
    print("  This downloads FineWeb-Edu from HuggingFace.")
    print("  It's a curated dataset of educational web text.")
    print()
    print("  Options:")
    print("    [1] FineWeb-Edu 10B tokens (~20GB download, recommended)")
    print("    [2] OpenWebText (~12GB download)")
    print("    [3] Skip (I already have data)")
    print()

    choice = input("  Your choice [1/2/3] (default: 1): ").strip()

    if choice == "3":
        print("  Skipped.")
        return True
    elif choice == "2":
        return run_command(
            f"{PYTHON} -c \""
            "from datasets import load_dataset; "
            "ds = load_dataset('openwebtext', split='train'); "
            "ds.save_to_disk('data/raw/openwebtext'); "
            "print(f'Downloaded {len(ds):,} examples')\"",
            "Downloading OpenWebText",
        )
    else:
        return run_command(
            f"bash scripts/01_download_data.sh",
            "Downloading FineWeb-Edu (10B token sample)",
        )


def step_prepare():
    """Step 2: Prepare data (tokenize)."""
    print()
    print("=" * 60)
    print("  Step 2: Prepare Data (Tokenize)")
    print("=" * 60)
    print()
    print("  This tokenizes the raw text using GPT-2's BPE tokenizer")
    print("  and saves it as efficient binary files (.bin).")
    print()
    print("  Output: data/train.bin and data/val.bin")
    print()

    ready = input("  Ready to prepare data? [Y/n]: ").strip().lower()
    if ready == "n":
        return False

    return run_command(
        f"bash scripts/02_prepare_data.sh",
        "Tokenizing dataset",
    )


def step_pretrain():
    """Step 3: Pretrain the model."""
    print()
    print("=" * 60)
    print("  Step 3: Pretrain the Model")
    print("=" * 60)
    print()
    print("  This trains the model to predict the next token.")
    print("  After pretraining, it can complete text but can't chat yet.")
    print()

    if not os.path.exists("data/train.bin"):
        print("  ERROR: data/train.bin not found. Run Step 2 first.")
        return False

    config = pick_config()
    print(f"\n  Using config: {config}")

    gpu_count, _ = detect_gpus()
    if gpu_count > 1:
        gpu_input = input(f"\n  How many GPUs to use? [1-{gpu_count}] (default: {gpu_count}): ").strip()
        try:
            num_gpus = int(gpu_input) if gpu_input else gpu_count
            num_gpus = min(num_gpus, gpu_count)
        except ValueError:
            num_gpus = gpu_count

        gpu_ids = input(f"  Which GPU IDs? (e.g., 0,1,2,3) (default: 0-{num_gpus-1}): ").strip()
        if not gpu_ids:
            gpu_ids = ",".join(str(i) for i in range(num_gpus))

        cmd = f"CUDA_VISIBLE_DEVICES={gpu_ids} {PYTHON.replace('python', 'torchrun')} --nproc_per_node={num_gpus} -m src.pretrain --config {config}"
    else:
        cmd = f"{PYTHON} -m src.pretrain --config {config}"

    # Check for resume
    latest = "data/checkpoints/pretrain/latest.pt"
    if os.path.exists(latest):
        resume = input(f"\n  Found existing checkpoint. Resume training? [Y/n]: ").strip().lower()
        if resume != "n":
            cmd += f" --resume {latest}"

    return run_command(cmd, "Pretraining")


def step_finetune():
    """Step 4: Fine-tune for chat."""
    print()
    print("=" * 60)
    print("  Step 4: Fine-tune for Chat")
    print("=" * 60)
    print()
    print("  This teaches the model to follow instructions and chat.")
    print("  It uses the Dolly-15K dataset (15,000 instruction-response pairs).")
    print()

    checkpoint = "data/checkpoints/pretrain/best.pt"
    if not os.path.exists(checkpoint):
        print(f"  ERROR: Pretrained checkpoint not found at {checkpoint}")
        print("  Run Step 3 first.")
        return False

    config = pick_config()
    print(f"\n  Using config: {config}")

    gpu_count, _ = detect_gpus()
    if gpu_count > 1:
        gpu_input = input(f"\n  How many GPUs to use? [1-{gpu_count}] (default: {gpu_count}): ").strip()
        try:
            num_gpus = int(gpu_input) if gpu_input else gpu_count
            num_gpus = min(num_gpus, gpu_count)
        except ValueError:
            num_gpus = gpu_count

        gpu_ids = input(f"  Which GPU IDs? (e.g., 0,1,2,3) (default: 0-{num_gpus-1}): ").strip()
        if not gpu_ids:
            gpu_ids = ",".join(str(i) for i in range(num_gpus))

        cmd = (f"CUDA_VISIBLE_DEVICES={gpu_ids} {PYTHON.replace('python', 'torchrun')} "
               f"--nproc_per_node={num_gpus} -m src.finetune "
               f"--config {config} --checkpoint {checkpoint}")
    else:
        cmd = f"{PYTHON} -m src.finetune --config {config} --checkpoint {checkpoint}"

    return run_command(cmd, "Fine-tuning")


def step_chat():
    """Step 5: Chat with your model."""
    print()
    print("=" * 60)
    print("  Step 5: Chat with Your Model!")
    print("=" * 60)
    print()

    # Try fine-tuned first, then pretrained
    if os.path.exists("data/checkpoints/finetune/best.pt"):
        checkpoint = "data/checkpoints/finetune/best.pt"
        print(f"  Using fine-tuned model: {checkpoint}")
    elif os.path.exists("data/checkpoints/pretrain/best.pt"):
        checkpoint = "data/checkpoints/pretrain/best.pt"
        print(f"  Using pretrained model: {checkpoint}")
        print("  (Note: without fine-tuning, the model will complete text")
        print("   but won't follow instructions well)")
    else:
        print("  ERROR: No checkpoint found. Run Step 3 (and optionally 4) first.")
        return False

    print()
    return run_command(
        f"{PYTHON} -m src.chat --checkpoint {checkpoint}",
        "Starting chat",
    )


def step_evaluate():
    """Step 6: Evaluate your model."""
    print()
    print("=" * 60)
    print("  Step 6: Evaluate Your Model")
    print("=" * 60)
    print()

    if os.path.exists("data/checkpoints/finetune/best.pt"):
        checkpoint = "data/checkpoints/finetune/best.pt"
    elif os.path.exists("data/checkpoints/pretrain/best.pt"):
        checkpoint = "data/checkpoints/pretrain/best.pt"
    else:
        print("  ERROR: No checkpoint found. Run Step 3 first.")
        return False

    print(f"  Checkpoint: {checkpoint}")
    print()
    return run_command(
        f"{PYTHON} -m eval.evaluate --checkpoint {checkpoint}",
        "Evaluating model",
    )


def main():
    print_header()
    print_status()

    steps = {
        "1": ("Download data", step_download),
        "2": ("Prepare data", step_prepare),
        "3": ("Pretrain", step_pretrain),
        "4": ("Fine-tune", step_finetune),
        "5": ("Chat", step_chat),
        "6": ("Evaluate", step_evaluate),
        "q": ("Quit", None),
    }

    while True:
        print("  What would you like to do?")
        print()
        for key, (label, _) in steps.items():
            print(f"    [{key}] {label}")
        print()

        choice = input("  Your choice: ").strip().lower()

        if choice == "q":
            print("\n  Goodbye!\n")
            break
        elif choice in steps and steps[choice][1] is not None:
            steps[choice][1]()
            print()
            print_status()
        else:
            print("  Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()
