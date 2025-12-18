#!/bin/bash
"""
Shell script to demonstrate grid search over loss function coefficients.
This script shows how to run experiments with different loss hyperparameters.
"""

# Define project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

# Hyperparameters to search
export SEEDS="42"
export LEARNING_RATES="0.001"
export BATCH_SIZES="32"

# Loss function coefficients (the new hyperparameters)
export LAMBDA_LLM_NEGATIVES="0.1 0.5"
export LAMBDA_MIXUP="0.1 0.5"
export MARGINS="0.2 0.5"

# Model components
export SELECTOR_TYPES="mlp slot"
export FUSER_TYPES="mean query_attn"

# Training settings
export EPOCHS="1"  # Just 1 epoch for testing
export SHOTS="1"    # Few shots for faster testing

echo "Starting loss coefficient search experiment..."
echo "Project root: $PROJECT_ROOT"
echo "="

# Iterate over all hyperparameter combinations
for seed in $SEEDS; do
  for lr in $LEARNING_RATES; do
    for batch_size in $BATCH_SIZES; do
      for lambda_llm_negatives in $LAMBDA_LLM_NEGATIVES; do
        for lambda_mixup in $LAMBDA_MIXUP; do
          for margin in $MARGINS; do
            for selector_type in $SELECTOR_TYPES; do
              for fuser_type in $FUSER_TYPES; do
                # Generate a descriptive method name
                method_name="${selector_type}_${fuser_type}_llmneg${lambda_llm_negatives}_mix${lambda_mixup}_margin${margin}_seed${seed}"
                
                echo "\nRunning experiment: $method_name"
                echo "Parameters:"
                echo "  seed: $seed"
                echo "  lr: $lr"
                echo "  batch_size: $batch_size"
                echo "  lambda_llm_negatives: $lambda_llm_negatives"
                echo "  lambda_mixup: $lambda_mixup"
                echo "  margin: $margin"
                echo "  selector_type: $selector_type"
                echo "  fuser_type: $fuser_type"
                
                # Run the experiment
                python "$PROJECT_ROOT/src/train_and_eval.py" \
                  --method "$method_name" \
                  --seed "$seed" \
                  --lr "$lr" \
                  --batch_size "$batch_size" \
                  --lambda_llm_negatives "$lambda_llm_negatives" \
                  --lambda_mixup "$lambda_mixup" \
                  --margin "$margin" \
                  --selector_type "$selector_type" \
                  --fuser_type "$fuser_type" \
                  --epochs "$EPOCHS" \
                  --shots "$SHOTS"
                
                if [ $? -eq 0 ]; then
                  echo "✓ Experiment completed successfully"
                else
                  echo "✗ Experiment failed"
                fi
              done
            done
          done
        done
      done
    done
  done
done

echo "\n"=""
echo "Loss coefficient search completed!"
echo "="""
