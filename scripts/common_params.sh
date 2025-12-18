#!/bin/bash
# ==============================================================================
# Common parameters for all hyperparameter search scripts
# ==============================================================================

# Project root (relative to scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default training parameters
DEFAULT_EPOCHS=50
DEFAULT_BATCH_SIZE=512
DEFAULT_LR=0.001
DEFAULT_SEED=42
DEFAULT_BACKBONE="ViT-L/16"
DEFAULT_ROOT_PATH="/data/ICML2026/clip/FA/my_dataset"
DEFAULT_SHOTS=16
DEFAULT_CLASS_NEGATIVES_PATH="/data/ICML2026/clip/FA/my_dataset"

# Hyperparameter search grids
LEARNING_RATES=(0.0005 0.001 0.002)
BATCH_SIZES=(512 1024)
SEEDS=(42)

# Loss function coefficients grids
LAMBDA_LLM_NEGATIVES=(0.1 0.5 1.0)
LAMBDA_MIXUP=(0.1 0.5 1.0)
MARGIN_VALUES=(0.2 0.5 1.0)

# Selector parameters grid
NUM_SELECT_VALUES=(8 16 32)
