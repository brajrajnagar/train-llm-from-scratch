# Train Your Own LLM From Scratch

Build and train a 160M-parameter LLM chatbot from scratch using modern architecture (Llama-style). Educational code with detailed comments explaining every design decision.

## Getting Started

**New here?** Just run:

```bash
python run.py
```

This interactive tool walks you through every step -- no need to memorize commands.

**Want to understand the code?** Read [GUIDE.md](GUIDE.md) -- it tells you which files to read and in what order.

## What You'll Build

A complete LLM training pipeline in 5 steps:

1. **Download** web-scale text data (FineWeb-Edu)
2. **Tokenize** with BPE and pack into efficient binary files
3. **Pretrain** a Llama-style transformer (text completion)
4. **Fine-tune** on instruction data (chatbot behavior)
5. **Chat** with your model interactively

## Architecture

Our model uses the same architecture as Llama 3, scaled down for educational training:

| Component | GPT-2 (old) | **Ours / Llama (new)** | Why? |
|-----------|-------------|----------------------|------|
| Position encoding | Learned embeddings | **RoPE** | Extrapolates to any length |
| Normalization | LayerNorm | **RMSNorm** | 15% faster, equally effective |
| Activation | GELU | **SwiGLU** | 1-3% better training efficiency |
| Attention | Multi-Head | **Grouped Query (GQA)** | 30% less KV-cache memory |
| Bias terms | Yes | **No** | Fewer params, marginally better |
| FFN expansion | 4x | **8/3x (with 3 matrices)** | Same FLOPs, better gating |

## Prerequisites

- Python 3.10+
- PyTorch 2.1+
- 1+ NVIDIA GPU (or Apple Silicon, or CPU for testing)
- ~30GB disk space for data
- ~4-12 hours training time (on 3x B200)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download training data
bash scripts/01_download_data.sh

# 3. Tokenize data into .bin files
bash scripts/02_prepare_data.sh

# 4. Pretrain the model
bash scripts/03_pretrain.sh

# 5. Fine-tune for chat
bash scripts/04_finetune.sh

# 6. Chat with your model!
bash scripts/05_chat.sh
```

## Project Structure

```
train-llm-from-scratch/
├── configs/
│   ├── 50m.yaml              # Debug model (~30 min training)
│   ├── 160m.yaml             # Sweet spot (~4-12 hr)
│   └── 400m.yaml             # For bigger hardware (~24-48 hr)
├── scripts/
│   ├── 01_download_data.sh   # Download FineWeb-Edu from HuggingFace
│   ├── 02_prepare_data.sh    # Tokenize with GPT-2 BPE -> .bin files
│   ├── 03_pretrain.sh        # Auto-detect GPUs, launch pretraining
│   ├── 04_finetune.sh        # Fine-tune on Dolly-15K instructions
│   └── 05_chat.sh            # Interactive chat session
├── src/
│   ├── model.py              # LLM architecture (RoPE, RMSNorm, SwiGLU, GQA)
│   ├── tokenizer.py          # GPT-2 BPE tokenizer wrapper
│   ├── data.py               # Data pipeline (download, tokenize, dataloaders)
│   ├── distributed.py        # Hardware auto-detection, FSDP, fallbacks
│   ├── trainer.py            # Training loop, gradient accum, cosine LR
│   ├── pretrain.py           # Pretraining entry point
│   ├── finetune.py           # Fine-tuning entry point
│   └── chat.py               # Interactive chat inference
├── eval/
│   └── evaluate.py           # Perplexity + sample generation
└── data/                     # Downloaded data, .bin files, checkpoints
```

## Hardware Compatibility

The code auto-detects your hardware and adapts:

| Hardware | Strategy | Dtype | Est. Training Time (160M) |
|----------|----------|-------|---------------------------|
| 3x B200 | FSDP | BF16 | ~4-12 hours |
| 1x A100 | Single GPU | BF16 | ~24-48 hours |
| 1x RTX 4090 | Single GPU | FP16 | ~24-48 hours |
| Apple M3 Max | MPS | FP32 | ~96+ hours |
| CPU only | CPU | FP32 | Weeks (smoke test only) |

## Model Sizes

| Config | Parameters | Training Time (3x B200) | Quality |
|--------|-----------|-------------------------|---------|
| `50m.yaml` | ~50M | ~30 minutes | Smoke test / debug |
| `160m.yaml` | ~160M | ~4-12 hours | Basic chatbot |
| `400m.yaml` | ~400M | ~24-48 hours | Better quality |

Start with `50m.yaml` to verify your setup, then train `160m.yaml` for real results.

## Coming from learn-gpt-from-scratch?

This project picks up where `learn-gpt-from-scratch` left off. Here's what's new:

| learn-gpt-from-scratch | train-llm-from-scratch |
|----------------------|----------------------|
| Character tokenizer (65 chars) | BPE tokenizer (50K tokens) |
| Learned position embeddings | RoPE (rotary embeddings) |
| LayerNorm | RMSNorm |
| GELU activation | SwiGLU (gated) |
| Multi-Head Attention | Grouped Query Attention |
| DataParallel | FSDP (Fully Sharded) |
| Shakespeare (1MB) | FineWeb-Edu (20GB+) |
| Text completion only | Pretrain + fine-tune + chat |
| 10M parameters | 50M-400M parameters |

Every file in `src/` has comments explaining these changes and why they matter.

## Training Pipeline Explained

### Phase 1: Pretraining
The model reads billions of tokens of web text and learns to predict the next token. After pretraining, it can complete text ("The capital of France is..." -> "Paris") but doesn't know how to follow instructions.

### Phase 2: Fine-tuning
We teach the model to be a chatbot using ~15K instruction-response pairs (Dolly-15K). Key differences from pretraining:
- **Lower learning rate** (2e-5 vs 6e-4) to preserve pretrained knowledge
- **Loss masking** -- only train on assistant responses, not user prompts
- **Special tokens** -- `<|user|>`, `<|assistant|>`, `<|end_turn|>` format

### Phase 3: Chat
Interactive inference with conversation history, temperature sampling, and top-k/top-p filtering.

## Training Tips

- **Start with 50m.yaml** to verify the pipeline works before committing to hours of training
- **Monitor the loss** -- it should decrease steadily. If it spikes, something is wrong
- **Checkpoints are saved automatically** -- you can resume with `--resume`
- **Gradient checkpointing** -- enable in config if you run out of GPU memory
- **Loss around 3-4** after pretraining is typical for a 160M model

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) -- Vaswani et al., 2017
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) -- Su et al., 2021
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) -- Shazeer, 2020
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) -- Ainslie et al., 2023
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) -- Zhang & Sennrich, 2019
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) -- Touvron et al., 2023
- [nanoGPT](https://github.com/karpathy/nanoGPT) -- Karpathy

## License

MIT
