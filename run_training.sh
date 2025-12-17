#!/bin/bash
# ==============================================================================
# Simple training launcher - Run from project root
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if from correct directory
if [[ ! -f "$PROJECT_ROOT/src/train_and_eval.py" ]]; then
    echo "Error: Must be run from project root directory"
    echo "Usage: bash run_training.sh [options]"
    exit 1
fi

# Forward to the actual training script
python3 "$PROJECT_ROOT/src/train_and_eval.py" "$@"
