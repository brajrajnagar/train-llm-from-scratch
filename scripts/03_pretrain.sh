#!/bin/bash
# =============================================================================
# Step 3: Pretrain the LLM
# =============================================================================
#
# Auto-detects GPU count and launches training accordingly:
#   - Multiple GPUs -> torchrun with FSDP
#   - Single GPU    -> python directly
#   - No GPU        -> python on CPU (very slow!)
#
# Time estimates for 160M model:
#   3x B200:    ~4 hours (~657K tokens/sec)
#   1x A100:    ~24-48 hours
#   1x RTX4090: ~24-48 hours
#   CPU:        Don't. (But it will work!)
#
# Usage:
#   bash scripts/03_pretrain.sh                          # 160M default
#   bash scripts/03_pretrain.sh configs/50m.yaml         # 50M smoke test
#   bash scripts/03_pretrain.sh configs/160m.yaml data/checkpoints/pretrain/latest.pt  # Resume
# =============================================================================

set -e

CONFIG=${1:-"configs/160m.yaml"}
RESUME=${2:-""}

# --- Logging setup ---
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/03_pretrain_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: $LOG_FILE"

echo "============================================"
echo "Step 3: Pretrain the LLM"
echo "============================================"
echo "Timestamp: $(date)"
echo "Config:    $CONFIG"

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
    echo "Resume: $RESUME"
fi

echo ""

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Found $GPU_COUNT GPUs -- launching distributed training with FSDP"
    echo ""
    torchrun --nproc_per_node="$GPU_COUNT" \
        -m src.pretrain \
        --config "$CONFIG" \
        $RESUME_FLAG
elif [ "$GPU_COUNT" -eq 1 ]; then
    echo "Found 1 GPU -- launching single-GPU training"
    echo ""
    python3 -m src.pretrain --config "$CONFIG" $RESUME_FLAG
else
    echo "No GPU found -- launching CPU training (this will be SLOW!)"
    echo ""
    python3 -m src.pretrain --config "$CONFIG" $RESUME_FLAG
fi
