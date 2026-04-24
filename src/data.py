# =============================================================================
# data.py -- Data Loading and Preprocessing
# =============================================================================
#
# THE DATA PIPELINE:
#
#   Phase 1: Download (01_download_data.sh)
#     HuggingFace datasets -> raw text on disk
#
#   Phase 2: Prepare (this file, CLI mode)
#     Raw text -> tokenize with GPT-2 BPE -> save as .bin (uint16 memmap)
#
#   Phase 3: Train (this file, Dataset classes)
#     .bin file -> memory-mapped reads -> random chunks -> DataLoader
#
# WHY MEMORY-MAPPED FILES?
#   - OpenWebText is ~18GB of text -> ~9B tokens -> ~18GB as uint16
#   - Can't fit all in RAM, definitely can't fit in GPU memory
#   - Memory mapping: OS loads pages on demand, like virtual memory
#   - Random access is fast (no sequential reading needed)
#   - Multiple workers can read the same file simultaneously
#
# =============================================================================

import os
import json
import struct
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm


# =============================================================================
# Binary file format for tokenized data
# =============================================================================
#
# Header: 8 bytes
#   magic (4 bytes): b'TLLM' (Train LLM)
#   version (2 bytes): uint16, currently 1
#   dtype_code (2 bytes): uint16 code (0 = uint16, 1 = uint32)
# Data: N * sizeof(dtype) bytes
#   Token IDs stored sequentially

MAGIC = b"TLLM"
VERSION = 1
HEADER_SIZE = 8


def write_tokenized_bin(token_ids_iter, output_path: str, total_tokens: int = None):
    """
    Write tokenized data to a binary file.

    Args:
        token_ids_iter: Iterator yielding lists/arrays of token IDs
        output_path: Path to write .bin file
        total_tokens: Optional total for progress bar
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        # Write header
        f.write(MAGIC)
        f.write(struct.pack("<H", VERSION))   # version
        f.write(struct.pack("<H", 0))         # dtype_code: 0 = uint16

        # Write token IDs
        written = 0
        pbar = tqdm(token_ids_iter, desc=f"Writing {output_path}", unit=" chunks")
        for chunk in pbar:
            arr = np.array(chunk, dtype=np.uint16)
            f.write(arr.tobytes())
            written += len(arr)
            if total_tokens:
                pbar.set_postfix(tokens=f"{written:,}/{total_tokens:,}")

    print(f"Wrote {written:,} tokens to {output_path}")
    return written


def read_tokenized_bin(path: str) -> np.memmap:
    """
    Open a tokenized .bin file as a memory-mapped numpy array.

    Returns:
        np.memmap of shape (num_tokens,) with dtype uint16
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid file format: expected {MAGIC!r}, got {magic!r}")
        version = struct.unpack("<H", f.read(2))[0]
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
        dtype_code = struct.unpack("<H", f.read(2))[0]

    dtype = np.uint16 if dtype_code == 0 else np.uint32
    itemsize = np.dtype(dtype).itemsize
    file_size = os.path.getsize(path)
    num_tokens = (file_size - HEADER_SIZE) // itemsize

    # Memory-map the data section (skip header)
    data = np.memmap(path, dtype=dtype, mode="r", offset=HEADER_SIZE, shape=(num_tokens,))
    return data


# =============================================================================
# Datasets
# =============================================================================

class PretrainDataset(Dataset):
    """
    Memory-mapped dataset for pretraining.

    Returns random contiguous chunks of (seq_length + 1) tokens,
    split into input (first seq_length) and target (last seq_length).

    WHY +1?
      Input:  [tok_0, tok_1, ..., tok_{T-1}]
      Target: [tok_1, tok_2, ..., tok_T]
      We need T+1 contiguous tokens to create one training example.

    WHY RANDOM SAMPLING?
      With billions of tokens, DistributedSampler would try to shuffle
      billions of indices (hangs). Instead, we pick random offsets each
      time __getitem__ is called -- the idx is ignored. This gives us
      uniform coverage without materializing a huge index list.
    """

    def __init__(self, data_path: str, seq_length: int, epoch_length: int = 100_000):
        self.data = read_tokenized_bin(data_path)
        self.seq_length = seq_length
        self.n_tokens = len(self.data)
        # Fixed epoch length so DataLoader/DistributedSampler don't choke
        self.epoch_length = epoch_length

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> tuple:
        # Random offset into the token stream
        max_start = self.n_tokens - self.seq_length - 1
        idx = np.random.randint(0, max_start)
        chunk = self.data[idx : idx + self.seq_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class FinetuneDataset(Dataset):
    """
    Dataset for instruction fine-tuning.

    Loads pre-tokenized conversations with loss masks from a JSONL file.
    Only computes loss on assistant responses (not user prompts).

    Each JSONL line: {"token_ids": [...], "loss_mask": [...]}

    Returns:
      input_ids:  (seq_length,) token IDs, padded/truncated
      targets:    (seq_length,) target IDs, -1 for masked positions
    """

    def __init__(self, data_path: str, seq_length: int):
        self.seq_length = seq_length
        self.examples = []

        with open(data_path, "r") as f:
            for line in f:
                example = json.loads(line.strip())
                self.examples.append(example)

        print(f"Loaded {len(self.examples)} fine-tuning examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple:
        example = self.examples[idx]
        token_ids = example["token_ids"]
        loss_mask = example["loss_mask"]

        # Truncate if too long
        if len(token_ids) > self.seq_length + 1:
            token_ids = token_ids[: self.seq_length + 1]
            loss_mask = loss_mask[: self.seq_length + 1]

        # Pad if too short
        pad_len = (self.seq_length + 1) - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [0] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)

        # Input is all but last token, targets are shifted by 1
        x = token_ids[:-1]
        y = token_ids[1:]
        mask = loss_mask[1:]  # Align mask with targets

        # Set targets to -1 where loss mask is 0 (ignored by cross_entropy)
        y[mask == 0] = -1

        return x, y


# =============================================================================
# DataLoader creation
# =============================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    distributed: bool = False,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader with optional distributed sampling.

    For distributed training (FSDP/DDP), uses DistributedSampler
    to ensure each GPU sees different data.
    """
    sampler = None
    shuffle = True

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid uneven batch sizes in distributed training
    )


# =============================================================================
# CLI: Data preparation
# =============================================================================

def tokenize_dataset(
    dataset_name: str,
    dataset_subset: str,
    output_dir: str,
    val_fraction: float = 0.005,
):
    """
    Download a HuggingFace dataset, tokenize it, and save as .bin files.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "HuggingFaceFW/fineweb-edu")
        dataset_subset: Dataset subset/config name (e.g., "sample-10BT")
        output_dir: Directory for train.bin and val.bin
        val_fraction: Fraction of data to hold out for validation
    """
    from datasets import load_dataset, load_from_disk
    from src.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    os.makedirs(output_dir, exist_ok=True)

    # Try loading from local disk first (saved by 01_download_data.sh)
    local_path = os.path.join(output_dir, "raw", dataset_subset if dataset_subset else "data")
    if os.path.exists(local_path):
        print(f"Loading dataset from disk: {local_path}")
        dataset = load_from_disk(local_path)
    else:
        print(f"Loading dataset from HuggingFace: {dataset_name} (subset: {dataset_subset})")
        if dataset_subset:
            dataset = load_dataset(dataset_name, name=dataset_subset, split="train")
        else:
            dataset = load_dataset(dataset_name, split="train")

    print(f"Dataset loaded: {len(dataset)} examples")

    # Tokenize with multiprocessing
    def tokenize_fn(examples):
        return {"token_ids": [tokenizer.encode(text) for text in examples["text"]]}

    print("Tokenizing...")
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Split into train/val
    total = len(dataset)
    val_size = max(int(total * val_fraction), 1)
    train_size = total - val_size

    print(f"Split: {train_size} train, {val_size} val")

    dataset = dataset.shuffle(seed=42)
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, total))

    # Write to .bin files
    def token_iter(data):
        for example in data:
            yield example["token_ids"]

    # Count total tokens for progress bar
    print("Writing train.bin...")
    write_tokenized_bin(
        token_iter(train_data),
        os.path.join(output_dir, "train.bin"),
    )

    print("Writing val.bin...")
    write_tokenized_bin(
        token_iter(val_data),
        os.path.join(output_dir, "val.bin"),
    )

    print("Data preparation complete!")


def prepare_finetune_data(
    dataset_name: str,
    output_dir: str,
    seq_length: int,
):
    """
    Download an instruction dataset, tokenize conversations, save as .jsonl.

    Supported datasets:
      - "databricks/databricks-dolly-15k"
      - "tatsu-lab/alpaca"
    """
    from datasets import load_dataset
    from src.tokenizer import Tokenizer

    tokenizer = Tokenizer(add_chat_tokens=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading instruction dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(dataset)} examples")

    output_path = os.path.join(output_dir, "finetune.jsonl")
    written = 0
    skipped = 0

    with open(output_path, "w") as f:
        for example in tqdm(dataset, desc="Processing"):
            # Convert to chat format based on dataset structure
            messages = _convert_to_chat(example, dataset_name)
            if not messages:
                skipped += 1
                continue

            # Tokenize with loss mask
            token_ids, loss_mask = tokenizer.encode_chat(messages)

            # Skip if too long (would be entirely truncated)
            if len(token_ids) > seq_length * 2:
                skipped += 1
                continue

            f.write(json.dumps({"token_ids": token_ids, "loss_mask": loss_mask}) + "\n")
            written += 1

    print(f"Wrote {written} examples to {output_path} (skipped {skipped})")


def _convert_to_chat(example: dict, dataset_name: str) -> list:
    """Convert a dataset example to chat format [{"role": ..., "content": ...}]."""
    if "dolly" in dataset_name:
        # Dolly format: instruction, context, response
        instruction = example.get("instruction", "")
        context = example.get("context", "")
        response = example.get("response", "")

        if not instruction or not response:
            return []

        user_msg = instruction
        if context:
            user_msg += f"\n\nContext: {context}"

        return [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]

    elif "alpaca" in dataset_name:
        # Alpaca format: instruction, input, output
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")

        if not instruction or not output:
            return []

        user_msg = instruction
        if inp:
            user_msg += f"\n\nInput: {inp}"

        return [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]

    else:
        # Generic: try common field names
        user = example.get("prompt", example.get("instruction", example.get("question", "")))
        assistant = example.get("response", example.get("output", example.get("answer", "")))

        if not user or not assistant:
            return []

        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]


# =============================================================================
# Main: run data preparation from command line
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation for LLM training")
    parser.add_argument(
        "--task", choices=["pretrain", "finetune"], required=True,
        help="'pretrain': tokenize text corpus to .bin. 'finetune': prepare instruction data."
    )
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_subset", type=str, default="sample-10BT")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--val_fraction", type=float, default=0.005)
    args = parser.parse_args()

    if args.task == "pretrain":
        tokenize_dataset(
            args.dataset,
            args.dataset_subset,
            args.output_dir,
            args.val_fraction,
        )
    elif args.task == "finetune":
        prepare_finetune_data(
            args.dataset,
            args.output_dir,
            args.seq_length,
        )
