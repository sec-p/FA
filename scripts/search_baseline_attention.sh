#!/bin/bash
# Grid search for baseline_attention
# Usage: edit SEEDS LRS BSS EPOCHS below or export PYTHON_CMD to override python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_CMD=${PYTHON_CMD:-python3}

METHOD="baseline_attention"
EPOCHS=${EPOCHS:-50}
SEEDS=(1 2 3)
LRS=(0.001 0.002)
BSS=(32 64)

# Loss function hyperparameters
LAMBDA_LLM_NEGATIVES=(0.05)
LAMBDA_MIXUP=(0.1)
MARGINS=(0.1)

OUT_DIR="$PROJECT_ROOT/results/grid_search/${METHOD}"
mkdir -p "$OUT_DIR"

for s in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for bs in "${BSS[@]}"; do
      for llm_coeff in "${LAMBDA_LLM_NEGATIVES[@]}"; do
        for mixup_coeff in "${LAMBDA_MIXUP[@]}"; do
          for margin in "${MARGINS[@]}"; do
            run_id="${METHOD}_s${s}_lr${lr}_bs${bs}_llm${llm_coeff}_mix${mixup_coeff}_m${margin}"
            logfile="$OUT_DIR/${run_id}.log"
            echo "Running ${run_id} -> $logfile"
            "$PYTHON_CMD" "$PROJECT_ROOT/run_training.sh" --no-config --method "$METHOD" --epochs "$EPOCHS" --lr "$lr" --batch_size "$bs" --seed "$s" \
              --lambda_llm_negatives "$llm_coeff" --lambda_mixup "$mixup_coeff" --margin "$margin" >> "$logfile" 2>&1
          done
        done
      done
    done
  done
done

echo "Grid search for ${METHOD} completed. Logs in $OUT_DIR"
