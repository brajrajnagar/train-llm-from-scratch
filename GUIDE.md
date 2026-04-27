# Complete Learning Guide: Train an LLM from Scratch

> **New here?** This guide walks you through **every concept** needed to understand and build a modern LLM chatbot from scratch. No prior deep learning experience required — we explain everything step by step.

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Model Architecture](#phase-1-model-architecture)
4. [Phase 2: Tokenization](#phase-2-tokenization)
5. [Phase 3: Data Pipeline](#phase-3-data-pipeline)
6. [Phase 4: Distributed Training](#phase-4-distributed-training)
7. [Phase 5: Training Loop](#phase-5-training-loop)
8. [Phase 6: Pretraining vs Fine-tuning](#phase-6-pretraining-vs-fine-tuning)
9. [Phase 7: Chat & Inference](#phase-7-chat--inference)
10. [Phase 8: Evaluation](#phase-8-evaluation)
11. [Quick Start](#quick-start)
12. [Troubleshooting](#troubleshooting)

---

## The Big Picture

### What Are We Building?

We're building a **chatbot** — a model that can have conversations with humans. Think ChatGPT, but much smaller and running on your own hardware.

### The 3-Phase Training Process

Training a modern LLM happens in **three distinct phases**, each with a different goal:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAINING          PHASE 2: FINE-TUNING       PHASE 3: CHAT    │
│  "Learn Language"              "Learn Instructions"       "Talk to Users"   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Input: Billions of words        Input: 15,000 examples   Input: Your text │
│  from the internet               of Q&A pairs                            │
│                                                                            │
│  Task: Predict the next          Task: Given a          Output: Helpful   │
│  word in a sentence              question, predict      response in chat  │
│                                  the answer             format            │
│                                                                            │
│  Result: A model that            Result: A model that   Result: You can   │
│  "knows language" but            "knows how to be       have natural      │
│  doesn't follow instructions     helpful"               conversations     │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Three Phases?

| Phase | Why It Exists | Analogy |
|-------|---------------|---------|
| **Pretraining** | Teaches grammar, facts, reasoning patterns | Like reading every book in a library |
| **Fine-tuning** | Teaches following instructions and chat format | Like training to be a helpful customer service rep |
| **Chat** | Lets you interact with the model | Like actually talking to customers |

---

## Prerequisites

### What You Need to Know

| Concept | Do You Need It? | Quick Explanation |
|---------|-----------------|-------------------|
| Python | Yes | We write everything in Python |
| Basic math | Yes | Addition, multiplication, exponents |
| Neural networks | Helpful but not required | We explain as we go |
| PyTorch | Helpful but not required | We explain the code |

### What You Need Installed

```bash
# Install dependencies
pip install -r requirements.txt

# That's it! The scripts handle everything else.
```

---

## Phase 1: Model Architecture

### Start Here: `src/model.py`

This file defines **what the model looks like** — its "brain structure."

### The Evolution: GPT-2 → Llama

Our model is based on **Llama** (Meta's model), which improved upon **GPT-2** (OpenAI's model). Here's what changed:

| Component | GPT-2 (Old) | Llama (New) | Why It's Better |
|-----------|-------------|-------------|-----------------|
| **Normalization** | LayerNorm | RMSNorm | 10-15% faster, simpler math |
| **Position Encoding** | Learned embeddings | RoPE (Rotary) | Works for any sequence length |
| **Attention** | Multi-Head (MHA) | Grouped Query (GQA) | 30% less memory |
| **Activation** | GELU | SwiGLU | Better training efficiency |
| **Linear Layers** | Has bias | No bias | Simpler, works just as well |

### The 5 Key Components

Let's understand each piece:

#### 1. RMSNorm (Normalization)

**What it does:** Normalizes the input values so they don't get too large or too small.

**Old way (LayerNorm):**
```python
# Subtract mean, divide by standard deviation, then scale and shift
y = gamma * (x - mean) / std + beta  # 2 parameters: gamma, beta
```

**New way (RMSNorm):**
```python
# Just divide by root-mean-square, then scale
y = gamma * x / sqrt(mean(x²) + eps)  # 1 parameter: gamma only
```

**Why it matters:** Fewer operations = faster training. Used in Llama, Mistral, Gemma.

---

#### 2. RoPE (Rotary Position Embedding)

**What it does:** Tells the model where each word is in the sentence.

**Old way (GPT-2):**
```python
# Learn a separate embedding for each position
position_embeddings = nn.Embedding(max_length, d_model)
# Problem: Can't handle sequences longer than training!
```

**New way (RoPE):**
```python
# Rotate query and key vectors based on position
# Position 0: small rotation
# Position 100: larger rotation
# Position 1000: even larger rotation
```

**Visual explanation:**
```
Imagine each dimension pair as a 2D arrow:

Position 0:  →     (0° rotation)
Position 1:  ↗     (small rotation)
Position 2:  ↑     (more rotation)
Position 3:  ↖     (even more rotation)

When computing attention, the angle difference = position difference!
```

**Why it matters:** Works for ANY sequence length, even longer than training.

---

#### 3. Grouped Query Attention (GQA)

**What it does:** Makes attention more memory-efficient.

**Old way (Multi-Head Attention):**
```
12 Query heads  →  12 Key heads  →  12 Value heads
Q1 Q2 ... Q12       K1 K2 ... K12     V1 V2 ... V12
```

**New way (GQA):**
```
12 Query heads  →  4 Key heads  →  4 Value heads
Q1-Q3 share K1,V1
Q4-Q6 share K2,V2
Q7-Q9 share K3,V3
Q10-Q12 share K4,V4
```

**Why it matters:**
- 30% less memory for the KV cache (important for long sequences)
- Faster inference
- Same quality as full attention

---

#### 4. SwiGLU (Activation Function)

**What it does:** Decides which information to pass through the network.

**Old way (GELU):**
```python
# Simple activation
output = Linear2(GELU(Linear1(x)))
```

**New way (SwiGLU):**
```python
# Gated activation — like a "valve" controlling information
gate = Linear_gate(x)      # "How much to let through"
up = Linear_up(x)          # "What to let through"
output = Linear_down(SiLU(gate) * up)
```

**SiLU function:** `SiLU(x) = x * sigmoid(x)` — smooth curve between 0 and x

**Why it matters:** Better control over information flow = more efficient training.

---

#### 5. Transformer Block

**Putting it all together:**

```
Input
  │
  ├────────────────────────────────────┐
  │  RMSNorm → GQA → Add Residual      │  ← x = x + GQA(RMSNorm(x))
  └────────────────────────────────────┘
  │
  ├────────────────────────────────────┐
  │  RMSNorm → SwiGLU → Add Residual   │  ← x = x + SwiGLU(RMSNorm(x))
  └────────────────────────────────────┘
  │
Output
```

**Residual connection:** Adding the input back to the output helps gradients flow through deep networks.

---

### The Complete Model

```
Token IDs: [2980, 318, 257, ...]  ← "This is a..."
    │
    ▼
┌───────────────────────────┐
│ Token Embedding           │  Each token → 768-dim vector
│ (NO position embedding!)  │  RoPE handles position in attention
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ Transformer Block 1       │
│ Transformer Block 2       │  ← 20 blocks for 160M model
│ ...                       │
│ Transformer Block 20      │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ Final RMSNorm             │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ Output Projection         │  768 → 50,257 (vocab size)
│ (tied with embedding)     │  Same weights as input embedding!
└───────────────────────────┘
    │
    ▼
Logits → Softmax → Next token probabilities
```

**Weight tying:** The output layer uses the SAME weights as the input embedding. This:
- Saves memory (fewer parameters to learn)
- Often improves quality
- Is standard practice in language models

---

### Model Configurations

We provide three model sizes:

| Config | Parameters | d_model | Layers | Training Time | Quality |
|--------|------------|---------|--------|---------------|---------|
| `50m.yaml` | 50 million | 512 | 12 | ~1 hour | Basic |
| `160m.yaml` | 160 million | 768 | 20 | ~4-24 hours | Good |
| `400m.yaml` | 400 million | 1024 | 24 | ~1-2 days | Better |

**Recommendation:** Start with `160m.yaml` — it's the sweet spot.

---

## Phase 2: Tokenization

### Read: `src/tokenizer.py`

**What is tokenization?** Converting text into numbers the model can understand.

### Character-level vs BPE

**Old way (character-level, like learn-gpt-from-scratch):**
```
"hello" → [h, e, l, l, o] → 5 tokens
"understanding" → [u, n, d, e, r, s, t, a, n, d, i, n, g] → 13 tokens
```

**New way (BPE — Byte Pair Encoding):**
```
"hello" → [hello] → 1 token
"understanding" → [under, standing] → 2 tokens
```

**Why BPE is better:**
1. Shorter sequences → model sees more context
2. Captures subword meaning ("un-" = negation, "standing" = position)
3. Handles unknown words via subword fallback

### The GPT-2 Tokenizer

We use the pre-trained GPT-2 tokenizer from HuggingFace:
- **Vocabulary size:** 50,257 tokens
- **Already trained:** No need to train our own!
- **Works great:** Used by GPT-2 and many other models

### Chat Format Tokens

For fine-tuning, we add 4 special tokens:

| Token | ID | Purpose |
|-------|-----|---------|
| `<|system|>` | 50257 | System instructions |
| `<|user|>` | 50258 | User message start |
| `<|assistant|>` | 50259 | Assistant response start |
| `<|end_turn|>` | 50260 | End of turn marker |

### Loss Mask — The Key to Fine-tuning

**The problem:** During fine-tuning, we only want to learn from assistant responses, not user prompts.

**The solution:** A "loss mask" that tells the model which tokens to learn from:

```
Conversation:
<|user|>What is 2+2?<|end_turn|>
<|assistant|>2+2 equals 4.<|end_turn|>

Token IDs:    [role][user text][end][role][assistant text][end]
Loss Mask:    [  0  ][    0    ][ 0 ][  0  ][      1      ][ 1 ]
              ↑ Don't learn from these    ↑ Learn from these
```

**In code:**
```python
def encode_chat(messages):
    token_ids = []
    loss_mask = []
    
    for msg in messages:
        # Add role token (never trained)
        token_ids.extend(role_token_ids)
        loss_mask.extend([0] * len(role_token_ids))
        
        # Add content
        token_ids.extend(content_ids)
        if msg["role"] == "assistant":
            loss_mask.extend([1] * len(content_ids))  # Train on this!
        else:
            loss_mask.extend([0] * len(content_ids))  # Skip this
```

---

## Phase 3: Data Pipeline

### Read: `src/data.py`

### The Data Journey

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Download (01_download_data.sh)                          │
│   HuggingFace datasets → Raw text files on disk                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Prepare (src/data.py, CLI mode)                         │
│   Raw text → Tokenize with GPT-2 BPE → Save as .bin file       │
│   (uint16, memory-mapped for efficiency)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Train (PretrainDataset class)                           │
│   .bin file → Memory-mapped reads → Random chunks → DataLoader │
└─────────────────────────────────────────────────────────────────┘
```

### Why Memory-Mapped Files?

**The problem:** OpenWebText is ~18GB of text → ~9 billion tokens → ~18GB as uint16.

You can't fit 18GB in RAM, and you definitely can't fit it in GPU memory!

**The solution:** Memory mapping lets the OS load pages on demand:

```python
# Memory-map a file (OS loads pages as needed)
data = np.memmap("train.bin", dtype=np.uint16, mode="r", offset=8)

# Access any part of the file instantly
chunk = data[1000000:1001025]  # Reads just that page from disk
```

**Benefits:**
- Dataset can be larger than RAM
- Random access is fast (no sequential reading)
- Multiple workers can read simultaneously

### Random Offset Sampling

**The problem:** With 9 billion tokens, shuffling all indices would hang your computer.

**The solution:** Pick a random offset each time you need data:

```python
def __getitem__(self, idx):
    # Pick random starting point
    max_start = len(self.data) - self.seq_length - 1
    offset = np.random.randint(0, max_start)
    
    # Read contiguous chunk
    chunk = self.data[offset : offset + self.seq_length + 1]
    
    # Input = first seq_length tokens
    # Target = last seq_length tokens (shifted by 1)
    x = chunk[:-1]
    y = chunk[1:]
    return x, y
```

**Why it works:** Every token has equal probability of being sampled, but we never materialize a huge index list.

### The Binary File Format

Our `.bin` files have a simple format:

```
Header (8 bytes):
┌─────────────┬─────────────┬─────────────┐
│ "TLLM" (4B) │ version (2B)│ dtype (2B)  │
└─────────────┴─────────────┴─────────────┘
Data (N × 2 bytes for uint16):
┌─────────────┬─────────────┬─────────────┬───────┐
│  token_0    │  token_1    │  token_2    │  ...  │
└─────────────┴─────────────┴─────────────┴───────┘
```

---

## Phase 4: Distributed Training

### Read: `src/distributed.py`

### The Hardware Auto-Detection

The same code runs on different hardware:

```
┌─────────────────────────────────────────────────────────────────┐
│ Hardware Detection Hierarchy                                    │
├─────────────────────────────────────────────────────────────────┤
│  Multiple CUDA GPUs?  →  FSDP (shard across GPUs)               │
│  Single CUDA GPU?     →  Single GPU (no overhead)               │
│  Apple Silicon?       →  MPS backend                            │
│  None of the above?   →  CPU (slow but works)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Why FSDP Instead of DataParallel?

**DataParallel (old way):**
```
GPU 0: Full model + Gradient collection (bottleneck!)
GPU 1: Full model
GPU 2: Full model
```
- Each GPU holds a FULL copy of the model
- All gradients gathered on GPU 0 (memory bottleneck)
- Python GIL limits parallelism

**FSDP (new way):**
```
GPU 0: 1/3 of model + 1/3 of gradients
GPU 1: 1/3 of model + 1/3 of gradients
GPU 2: 1/3 of model + 1/3 of gradients
```
- Model is SHARDED across GPUs
- Weights gathered on-demand, then discarded
- Memory usage ~1/N (can train larger models!)

**Example (160M model, 3 GPUs):**
- DataParallel: 480M total GPU memory (160M × 3)
- FSDP: 160M total GPU memory (160M / 3 per GPU)

### How FSDP Works

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap the model
model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,  # Each block is a shard unit
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

**What happens internally:**
1. Each GPU holds 1/N of each layer's parameters
2. Before forward pass: gather needed parameters from other GPUs
3. Compute forward/backward
4. Reduce gradients
5. Discard gathered parameters (free memory)

### Mixed Precision Training

**The idea:** Use lower precision (16-bit) for faster computation, but keep master weights in 32-bit.

| Precision | Memory | Speed | Use Case |
|-----------|--------|-------|----------|
| FP32 | 4 bytes | Slow | CPU, old GPUs |
| BF16 | 2 bytes | Fast | Modern GPUs (A100, H100, B200) |
| FP16 | 2 bytes | Fast | Older GPUs (V100, RTX 3090) |

**Auto-detection:**
```python
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16  # Best for LLMs
elif device.type == "cuda":
    dtype = torch.float16   # Older GPUs
else:
    dtype = torch.float32   # CPU fallback
```

---

## Phase 5: Training Loop

### Read: `src/trainer.py`

### The Training Loop at a Glance

```python
for step in range(max_steps):
    # 1. Get learning rate from schedule
    lr = get_lr(step)
    
    # 2. Gradient accumulation (simulate larger batches)
    for micro_step in range(grad_accum_steps):
        x, y = get_batch()
        logits, loss = model(x, y)
        loss.backward()  # Accumulate gradients
    
    # 3. Clip gradients (prevent exploding gradients)
    clip_gradients()
    
    # 4. Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    # 5. Log metrics, save checkpoints, evaluate
```

### Gradient Accumulation

**The problem:** You want batch size 96, but GPU only fits 8 sequences.

**The solution:** Process 8 sequences 12 times, accumulate gradients, then update:

```
Micro-step 1: forward(8 seqs) → backward → accumulate grads
Micro-step 2: forward(8 seqs) → backward → accumulate grads
...
Micro-step 12: forward(8 seqs) → backward → accumulate grads
→ optimizer.step() → zero grads
```

**Result:** Same effect as batch_size=96!

```python
# Scale loss so accumulated gradients have correct magnitude
loss = loss / grad_accum_steps
```

### Cosine LR Schedule with Warmup

**Why not constant LR?**
- High LR at start: unstable training
- High LR at end: can't converge precisely

**The solution:** Warm up, then cosine decay:

```
LR
│     /\
│    /  \_____________
│   /                 \_____
│  /                        \___
│ +───────────────────────────── steps
│   warmup    cosine decay
```

**In code:**
```python
def get_lr(step):
    if step < warmup_steps:
        # Linear warmup: 0 → peak
        return learning_rate * (step + 1) / warmup_steps
    
    # Cosine decay: peak → min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + cos(π * progress))
    return min_lr + (learning_rate - min_lr) * cosine_decay
```

### Mixed Precision Training

```python
# Use automatic mixed precision (AMP)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(x, y)

# Scale loss for FP16 (no-op for BF16/FP32)
scaler.scale(loss).backward()

# Unscale, clip, step
scaler.unscale_(optimizer)
clip_gradients()
scaler.step(optimizer)
scaler.update()
```

---

## Phase 6: Pretraining vs Fine-tuning

### Read: `src/pretrain.py` and `src/finetune.py`

### Key Differences

| Aspect | Pretraining | Fine-tuning |
|--------|-------------|-------------|
| **Data** | Web text (~10B tokens) | Instructions (15K examples) |
| **Batch size** | 64 per GPU | 4 per GPU |
| **Learning rate** | 6e-4 (high) | 2e-5 (low) |
| **Steps** | 50,000 | 2,000 |
| **Loss** | All tokens | Only assistant responses |
| **Vocab** | 50,257 | 50,261 (+4 chat tokens) |
| **Result** | Text completer | Chatbot |

### Fine-tuning Special Handling

**1. Resize embeddings for chat tokens:**
```python
# Load pretrained model (vocab_size=50257)
model.load_state_dict(pretrained_checkpoint)

# Resize for chat tokens (vocab_size=50261)
model.resize_token_embeddings(50261)

# Initialize new embeddings with small random values
# Keep pretrained embeddings unchanged
```

**2. Deterministic train/val split:**
```python
# Ensure every GPU sees the same split
rng = torch.Generator().manual_seed(42)
perm = torch.randperm(n_total).tolist()
val_indices = perm[:n_val]
train_indices = perm[n_val:]
```

**3. Loss mask for assistant-only training:**
```python
# In FinetuneDataset.__getitem__:
x = token_ids[:-1]
y = token_ids[1:].clone()  # Clone to avoid view corruption!
y[mask == 0] = -1  # Ignore these tokens in loss
```

---

## Phase 7: Chat & Inference

### Read: `src/chat.py`

### How Inference Works

```python
# 1. Format conversation
prompt = "<|user|>Hello!<|end_turn|><|assistant|>"

# 2. Tokenize
token_ids = tokenizer.encode(prompt)

# 3. Generate until end_turn or max tokens
output = model.generate(token_ids, max_new_tokens=512)

# 4. Decode response
response = tokenizer.decode(output)
```

### Sampling Strategies

**Temperature:** Controls randomness
- Low (0.2): Focused, deterministic
- Medium (0.7): Balanced (default)
- High (1.5): Random, creative

**Top-k sampling:** Only sample from top k tokens
```python
# Keep only top 50 tokens, zero out rest
topk_probs, topk_indices = torch.topk(probs, 50)
next_token = multinomial(topk_probs)
```

**Top-p (nucleus) sampling:** Keep tokens until cumulative prob >= p
```python
# Keep smallest set of tokens with cumulative prob >= 0.9
sorted_probs = sort(probs, descending=True)
cumulative = cumsum(sorted_probs)
mask = cumulative >= 0.9
```

---

## Phase 8: Evaluation

### Read: `eval/evaluate.py`

### Metrics

**Perplexity:** How "surprised" the model is by the data
```
Perplexity = exp(cross_entropy_loss)
Lower = better (model is less surprised)
```

| Perplexity | Interpretation |
|------------|----------------|
| 10-20 | Excellent |
| 20-50 | Good |
| 50-100 | Okay |
| 100+ | Needs improvement |

### Running Evaluation

```bash
# Evaluate pretrained model
python -m eval.evaluate --checkpoint data/checkpoints/pretrain/best.pt

# Evaluate fine-tuned model
python -m eval.evaluate --checkpoint data/checkpoints/finetune/best.pt
```

---

## Quick Start

### Option 1: Interactive Mode (Recommended for First Time)

```bash
python run.py
```

This walks you through each step interactively.

### Option 2: Manual Steps

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

### Training Times

| Hardware | 50M Model | 160M Model | 400M Model |
|----------|-----------|------------|------------|
| 3x B200 | ~30 min | ~4 hours | ~12 hours |
| 1x A100 | ~1 hour | ~8 hours | ~24 hours |
| 1x RTX 4090 | ~2 hours | ~24 hours | ~48 hours |
| CPU | ~1 day | ~1 week | ~2 weeks |

---

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"

**Solutions:**
- Reduce batch size in config
- Enable gradient checkpointing: `gradient_checkpointing: true`
- Use FSDP for multi-GPU

#### 2. "Invalid file format: expected b'TLLM'"

**Cause:** Data preparation didn't complete.

**Solution:** Re-run `scripts/02_prepare_data.sh`

#### 3. Fine-tuning val loss is meaningless (shows 485165195.41)

**Cause:** Forgot to pass val loader or val split is 0.

**Solution:** Ensure `--val_split` is set (default 0.02).

#### 4. "vectorized gather kernel index out of bounds"

**Cause:** Bug in fine-tuning data where x and y share storage.

**Solution:** Clone y before mutating:
```python
y = token_ids[1:].clone()  # Not just token_ids[1:]
```

#### 5. Model repeats itself during generation

**Cause:** Small models tend to loop at this scale.

**Solutions:**
- Use chat template instead of raw prompts
- Lower temperature (0.5-0.7)
- Use top-k=40, top-p=0.9

---

## Key Concepts Cheat Sheet

| Concept | What It Means | Where in Code |
|---------|--------------|---------------|
| **BPE tokenization** | Split text into subwords, not characters | `src/tokenizer.py` |
| **RoPE** | Encode position by rotating Q/K vectors | `src/model.py:RotaryPositionalEmbedding` |
| **GQA** | Share KV heads across query heads | `src/model.py:GroupedQueryAttention` |
| **SwiGLU** | Gated FFN: `SiLU(gate) * up` | `src/model.py:SwiGLUFFN` |
| **RMSNorm** | Normalize by RMS only (no mean) | `src/model.py:RMSNorm` |
| **FSDP** | Shard model across GPUs | `src/distributed.py` |
| **Gradient accumulation** | Simulate large batches | `src/trainer.py:Trainer.train` |
| **Cosine LR schedule** | Warmup then smooth decay | `src/trainer.py:Trainer.get_lr` |
| **Loss masking** | Only train on assistant responses | `src/tokenizer.py:encode_chat` |
| **Weight tying** | Embedding and output share weights | `src/model.py:LLM.__init__` |
| **Memory mapping** | Load data pages on demand from disk | `src/data.py:read_tokenized_bin` |
| **Random offset sampling** | Pick random offsets into token stream | `src/data.py:PretrainDataset` |
| **Deterministic val holdout** | 2% of Dolly-15K, seed=42 | `src/finetune.py` |
| **Vocab resize on load** | Detect finetune state_dict vocab > model config vocab | `src/chat.py`, `eval/evaluate.py` |

---

## Next Steps

After understanding this codebase, you can:

1. **Experiment:** Change hyperparameters, try different data
2. **Scale up:** Train larger models with more GPUs
3. **Add features:** Implement beam search, multi-turn conversations
4. **Deploy:** Serve your model with FastAPI or Gradio
5. **Learn more:** Read the Llama, GPT-3, and Transformer papers

Happy training! 🚀