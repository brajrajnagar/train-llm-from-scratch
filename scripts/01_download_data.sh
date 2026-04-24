#!/bin/bash
# =============================================================================
# Step 1: Download Training Data
# =============================================================================
#
# Downloads FineWeb-Edu (10B token sample) from HuggingFace.
# This is high-quality educational web text, perfect for pretraining.
#
# Alternative datasets:
#   bash scripts/01_download_data.sh openwebtext ""
#   bash scripts/01_download_data.sh wikitext wikitext-103-v1
#
# Size: ~20GB download, ~30GB on disk
# Time: 10-30 minutes depending on internet speed
# =============================================================================

set -e

DATASET=${1:-"HuggingFaceFW/fineweb-edu"}
SUBSET=${2:-"sample-10BT"}
OUTPUT_DIR=${3:-"data/raw"}

echo "============================================"
echo "Step 1: Download Training Data"
echo "============================================"
echo "Dataset: $DATASET"
echo "Subset:  $SUBSET"
echo "Output:  $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

python3 -c "
from datasets import load_dataset
import os

dataset_name = '$DATASET'
subset = '$SUBSET'
output_dir = '$OUTPUT_DIR'

print(f'Loading dataset from HuggingFace: {dataset_name}')
if subset:
    print(f'Subset: {subset}')
    dataset = load_dataset(dataset_name, name=subset, split='train')
else:
    dataset = load_dataset(dataset_name, split='train')

print(f'Downloaded {len(dataset):,} examples')

save_path = os.path.join(output_dir, subset if subset else 'data')
dataset.save_to_disk(save_path)
print(f'Saved to: {save_path}')
print('Done!')
"

echo ""
echo "Download complete!"
echo "Next step: bash scripts/02_prepare_data.sh"
