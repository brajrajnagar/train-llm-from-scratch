#!/bin/bash
# =============================================================================
# Step 4: Fine-tune for Chat
# =============================================================================
#
# Fine-tunes the pretrained model on instruction data (Dolly-15K).
# This teaches the model to follow instructions and have conversations.
#
# Time: ~30-60 minutes on 3x B200, ~2-4 hours on 1x GPU
#
# Usage:
#   bash scripts/04_finetune.sh                                        # Defaults
#   bash scripts/04_finetune.sh configs/160m.yaml path/to/checkpoint   # Custom
# =============================================================================

set -e

CONFIG=${1:-"configs/160m.yaml"}
CHECKPOINT=${2:-"data/checkpoints/pretrain/best.pt"}

echo "============================================"
echo "Step 4: Fine-tune for Chat"
echo "============================================"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Check if pretrained checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $CHECKPOINT"
    echo "Run pretraining first: bash scripts/03_pretrain.sh"
    exit 1
fi

# Prepare fine-tuning data if not already done
if [ ! -f "data/finetune.jsonl" ]; then
    echo "Preparing fine-tuning data (Dolly-15K)..."
    python3 -m src.data \
        --task finetune \
        --dataset "databricks/databricks-dolly-15k" \
        --output_dir data \
        --seq_length 1024
    echo ""
fi

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Found $GPU_COUNT GPUs -- launching distributed fine-tuning"
    echo ""
    torchrun --nproc_per_node="$GPU_COUNT" \
        -m src.finetune \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --finetune_data data/finetune.jsonl
else
    echo "Launching single-device fine-tuning"
    echo ""
    python3 -m src.finetune \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --finetune_data data/finetune.jsonl
fi

echo ""
echo "Fine-tuning complete!"
echo "Next step: bash scripts/05_chat.sh"
