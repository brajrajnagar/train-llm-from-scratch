#!/bin/bash
# =============================================================================
# Step 5: Chat with Your LLM!
# =============================================================================
#
# Start an interactive chat session with your fine-tuned model.
# Type messages and see your LLM respond!
#
# Usage:
#   bash scripts/05_chat.sh                                        # Default checkpoint
#   bash scripts/05_chat.sh data/checkpoints/finetune/best.pt      # Custom checkpoint
# =============================================================================

CHECKPOINT=${1:-"data/checkpoints/finetune/best.pt"}

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Run fine-tuning first: bash scripts/04_finetune.sh"
    exit 1
fi

python3 -m src.chat \
    --checkpoint "$CHECKPOINT" \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 0.9
