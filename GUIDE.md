# Learning Guide

New here? This guide tells you **what to read** and **in what order** to understand the full LLM training pipeline.

## The Big Picture

Training an LLM chatbot has 3 phases:

```
Phase 1: PRETRAIN          Phase 2: FINE-TUNE         Phase 3: CHAT
"Learn language"           "Learn to be helpful"      "Talk to humans"

Web text (billions         Instruction pairs          User types,
of tokens) --> Model       (15K examples) --> Model   model responds
learns next-token          learns to follow           interactively
prediction                 instructions
```

## Reading Order

Read the files in this order. Each builds on the previous one.

### 1. Start Here: Model Architecture

**Read:** `src/model.py`

This is the heart of the project. It implements a modern LLM (Llama-style) with 4 key innovations over GPT-2:

| What | Class to Read | Why It Matters |
|------|--------------|----------------|
| Normalization | `RMSNorm` | Faster than LayerNorm, no mean centering |
| Position encoding | `RotaryPositionalEmbedding` | Handles any sequence length, no learned table |
| Attention | `GroupedQueryAttention` | Shares KV heads = 30% less memory |
| Feed-forward | `SwiGLUFFN` | Gated activation = better training efficiency |

Then see how they're assembled: `TransformerBlock` -> `LLM`.

**After reading:** You understand what the model looks like inside.

### 2. Tokenization

**Read:** `src/tokenizer.py`

Short file. Wraps the GPT-2 BPE tokenizer. Key concept: `encode_chat()` returns both token IDs and a **loss mask** -- this is how fine-tuning only trains on assistant responses.

**After reading:** You understand how text becomes numbers (and back).

### 3. Data Pipeline

**Read:** `src/data.py`

How data flows from raw text to training batches:
```
HuggingFace dataset
  --> tokenize with BPE
    --> save as .bin file (uint16, memory-mapped)
      --> PretrainDataset (random chunks via random offset sampling)
        --> DataLoader (batches for GPU)
```

Two key concepts:
- **Memory-mapped files** -- the dataset is too large for RAM, so we let the OS load pages on demand.
- **Random offset sampling** -- with ~10B tokens, we can't shuffle all indices (too slow). Instead, each `__getitem__` picks a random offset into the token stream, giving uniform coverage without materializing a huge index list.

**After reading:** You understand how billions of tokens get to the GPU efficiently.

### 4. Distributed Training

**Read:** `src/distributed.py`

How the code adapts to any hardware:
```
3x B200?     --> FSDP (model sharded across GPUs)
1x RTX 4090? --> Single GPU
Apple M3?    --> MPS backend
No GPU?      --> CPU (slow but works)
```

Key concept: **FSDP** -- instead of copying the full model to each GPU (DataParallel), FSDP *shards* it so each GPU holds 1/N of the weights.

**After reading:** You understand how multi-GPU training works.

### 5. Training Loop

**Read:** `src/trainer.py`

The actual training algorithm. Three new concepts beyond basic training:

1. **Gradient accumulation** -- simulate large batches by accumulating gradients over multiple forward passes
2. **Cosine LR schedule** -- warmup then smooth decay, used by GPT-3/Llama
3. **Mixed precision** -- compute in BF16/FP16 for 2x speed

**After reading:** You understand every step of the training loop.

### 6. Putting It Together

**Read:** `src/pretrain.py` then `src/finetune.py`

These are short files that wire everything together. Compare them to see the key differences:

| | Pretraining | Fine-tuning |
|--|------------|-------------|
| Data | Web text (~10B tokens) | Instructions (15K examples) |
| Batch size | 64 per GPU | 4 per GPU |
| Learning rate | 6e-4 (high) | 2e-5 (low) |
| Steps | 50,000 | 2,000 |
| Loss | All tokens | Only assistant responses |
| Result | Text completer | Chatbot |

### 7. Chat Interface

**Read:** `src/chat.py`

How inference works: format conversation with special tokens, generate until `<|end_turn|>`, manage history.

### 8. Evaluation

**Read:** `eval/evaluate.py`

How to measure model quality: **perplexity** (lower = better) and sample generation.

### 9. Configuration

**Read:** `configs/50m.yaml`, `configs/160m.yaml`, `configs/400m.yaml`

Compare the three configs to see how model size scales: more layers, wider dimensions, longer sequences.

## Running the Pipeline

Don't want to figure out commands? Just run:

```bash
python run.py
```

It will walk you through each step interactively.

Or run each step manually:

```bash
# Step 1: Download data
bash scripts/01_download_data.sh

# Step 2: Tokenize to .bin
bash scripts/02_prepare_data.sh

# Step 3: Pretrain (auto-detects GPUs)
bash scripts/03_pretrain.sh

# Step 4: Fine-tune for chat
bash scripts/04_finetune.sh

# Step 5: Chat!
bash scripts/05_chat.sh
```

## Key Concepts Cheat Sheet

| Concept | What It Means | Where in Code |
|---------|--------------|---------------|
| BPE tokenization | Split text into subwords, not characters | `src/tokenizer.py` |
| RoPE | Encode position by rotating Q/K vectors | `src/model.py:RotaryPositionalEmbedding` |
| GQA | Share KV heads across query heads | `src/model.py:GroupedQueryAttention` |
| SwiGLU | Gated FFN: `SiLU(gate) * up` | `src/model.py:SwiGLUFFN` |
| RMSNorm | Normalize by RMS only (no mean) | `src/model.py:RMSNorm` |
| FSDP | Shard model across GPUs | `src/distributed.py` |
| Gradient accumulation | Simulate large batches | `src/trainer.py:Trainer.train` |
| Cosine LR schedule | Warmup then smooth decay | `src/trainer.py:Trainer.get_lr` |
| Loss masking | Only train on assistant responses | `src/tokenizer.py:encode_chat` |
| Weight tying | Embedding and output share weights | `src/model.py:LLM.__init__` |
| Memory mapping | Load data pages on demand from disk | `src/data.py:read_tokenized_bin` |
| Random sampling | Pick random offsets into token stream (avoids shuffling billions of indices) | `src/data.py:PretrainDataset` |
