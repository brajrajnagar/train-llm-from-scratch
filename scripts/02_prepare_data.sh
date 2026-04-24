#!/bin/bash
# =============================================================================
# Step 2: Prepare Data (Tokenize to .bin)
# =============================================================================
#
# Tokenizes the downloaded dataset using GPT-2 BPE tokenizer.
# Saves as memory-mapped .bin files for fast training.
#
# Output: data/train.bin, data/val.bin
# Time: 30-60 minutes for 10B tokens (parallelized across CPU cores)
# =============================================================================

set -e

DATASET=${1:-"HuggingFaceFW/fineweb-edu"}
SUBSET=${2:-"sample-10BT"}
OUTPUT_DIR=${3:-"data"}

echo "============================================"
echo "Step 2: Prepare Data (Tokenize)"
echo "============================================"
echo "Dataset: $DATASET"
echo "Output:  $OUTPUT_DIR"
echo ""

python3 -m src.data \
    --task pretrain \
    --dataset "$DATASET" \
    --dataset_subset "$SUBSET" \
    --output_dir "$OUTPUT_DIR" \
    --val_fraction 0.005

echo ""
echo "Data preparation complete!"
echo "  Train: $OUTPUT_DIR/train.bin"
echo "  Val:   $OUTPUT_DIR/val.bin"
echo ""
echo "Next step: bash scripts/03_pretrain.sh"
