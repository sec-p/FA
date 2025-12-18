#!/bin/bash
# ==============================================================================
# Hyperparameter search for selector_mlp_all_fusers method
# ==============================================================================

# Load common parameters
source "$(dirname "${BASH_SOURCE[0]}")/common_params.sh"

# Method-specific configuration
METHOD="selector_mlp_all_fusers"
SELECTOR_TYPE="mlp"

# Define all fuser types to search
FUSER_TYPES=("mean" "query_attn" "self_attn")

# Log directory
LOG_DIR="$PROJECT_ROOT/logs/search_${METHOD}"
mkdir -p "$LOG_DIR"

# Start time
START_TIME=$(date +%s)
echo "Starting hyperparameter search for ${METHOD} at $(date)"
echo "Log directory: $LOG_DIR"
echo "Method: $METHOD"
echo "Selector: $SELECTOR_TYPE"
echo "Fusers: ${FUSER_TYPES[@]}"
echo "====================================="

# Hyperparameter combinations counter
COUNTER=0
total_combinations=$(( ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#SEEDS[@]} * ${#NUM_SELECT_VALUES[@]} * ${#FUSER_TYPES[@]} * ${#LAMBDA_LLM_NEGATIVES[@]} * ${#LAMBDA_MIXUP[@]} * ${#MARGIN_VALUES[@]} ))

# Grid search loop
for lr in "${LEARNING_RATES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for num_select in "${NUM_SELECT_VALUES[@]}"; do
                for fuser_type in "${FUSER_TYPES[@]}"; do
                    for llm_neg_lambda in "${LAMBDA_LLM_NEGATIVES[@]}"; do
                        for mixup_lambda in "${LAMBDA_MIXUP[@]}"; do
                            for margin_val in "${MARGIN_VALUES[@]}"; do
                                ((COUNTER++))
                                echo -e "\n[$COUNTER/$total_combinations] Running: $METHOD - lr=$lr, bs=$bs, seed=$seed, num_select=$num_select, fuser=$fuser_type, lambda_llm=$llm_neg_lambda, lambda_mixup=$mixup_lambda, margin=$margin_val"
                                
                                # Run training command
                                if python3 "$PROJECT_ROOT/src/train_and_eval.py" \
                                    --method "$METHOD" \
                                    --epochs "$DEFAULT_EPOCHS" \
                                    --lr "$lr" \
                                    --batch_size "$bs" \
                                    --seed "$seed" \
                                    --backbone "$DEFAULT_BACKBONE" \
                                    --root_path "$DEFAULT_ROOT_PATH" \
                                    --shots "$DEFAULT_SHOTS" \
                                    --class_negatives_path "$DEFAULT_CLASS_NEGATIVES_PATH" \
                                    --lambda_llm_negatives "$llm_neg_lambda" \
                                    --lambda_mixup "$mixup_lambda" \
                                    --margin "$margin_val" \
                                    --selector_type "$SELECTOR_TYPE" \
                                    --fuser_type "$fuser_type" \
                                    --num_select "$num_select" 2>&1 | tee -a "$LOG_DIR/grid_search.log"; then
                                    echo "  ✓ Success"
                                else
                                    echo "  ✗ Failed"
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "====================================="
echo "Hyperparameter search completed at $(date)"
echo "Total duration: $((DURATION/3600))h $(((DURATION%3600)/60))m $((DURATION%60))s"
echo "Results saved to: $LOG_DIR"
