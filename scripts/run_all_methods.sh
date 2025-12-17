#!/bin/bash
# Grid Search Training Scripts for Method Combinations
# Auto-generated for parallel execution
# Run: bash run_all_methods.sh

set -e

# ============================================================================
# Method Combinations
# ============================================================================
# selector: mlp, slot
# fuser: mean, query_attn, self_attn
# Total combinations: 2 * 3 = 6

METHODS=(
    # Format: selector_type:fuser_type
    "mlp:mean"
    "mlp:query_attn"
    "mlp:self_attn"
    "slot:mean"
    "slot:query_attn"
    "slot:self_attn"
)

# ============================================================================
# Hyperparameter Variations per Method
# ============================================================================

SEEDS=(1 2 3)
LAMBDA_LLM=(0.01 0.05 0.1)
LAMBDA_MIXUP=(0.05 0.1 0.2)
LEARNING_RATES=(0.0005 0.001 0.002)

# ============================================================================
# Configuration Template
# ============================================================================

BASE_CONFIG="configs/my_config.yaml"
CACHE_ROOT="./my_caches"

# ============================================================================
# Functions
# ============================================================================

run_training() {
    local selector=$1
    local fuser=$2
    local seed=$3
    local lambda_llm=$4
    local lambda_mixup=$5
    local lr=$6
    
    echo "========================================================================"
    echo "Running: selector=$selector, fuser=$fuser"
    echo "         seed=$seed, lambda_llm=$lambda_llm, lambda_mixup=$lambda_mixup, lr=$lr"
    echo "========================================================================"
    
    # Load base config
    cp "$BASE_CONFIG" "configs/temp_config.yaml"
    
    # Create Python script to update YAML
    python3 << EOF
import yaml

with open('configs/temp_config.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

cfg['selector_type'] = '$selector'
cfg['fuser_type'] = '$fuser'
cfg['seed'] = $seed
cfg['lambda_llm_negatives'] = $lambda_llm
cfg['lambda_mixup'] = $lambda_mixup
cfg['lr'] = $lr

with open('configs/temp_config.yaml', 'w') as f:
    yaml.dump(cfg, f)
EOF
    
    # Run training
    python train_modular.py --config configs/temp_config.yaml --is_train 1
    
    # Cleanup
    rm -f configs/temp_config.yaml
    
    echo "✓ Completed: selector=$selector, fuser=$fuser, seed=$seed"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║          Grid Search: Method Combinations                          ║"
    echo "║          Total: ${#METHODS[@]} methods × ${#SEEDS[@]} seeds × ${#LAMBDA_LLM[@]} lambdas × ${#LEARNING_RATES[@]} lrs"
    echo "║                = $((${#METHODS[@]} * ${#SEEDS[@]} * ${#LAMBDA_LLM[@]} * ${#LEARNING_RATES[@]})) experiments"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    total_experiments=0
    completed_experiments=0
    
    # Loop through all combinations
    for method in "${METHODS[@]}"; do
        IFS=':' read -r selector fuser <<< "$method"
        
        for seed in "${SEEDS[@]}"; do
            for lambda_llm in "${LAMBDA_LLM[@]}"; do
                for lambda_mixup in "${LAMBDA_MIXUP[@]}"; do
                    for lr in "${LEARNING_RATES[@]}"; do
                        total_experiments=$((total_experiments + 1))
                    done
                done
            done
        done
    done
    
    # Execute training
    for method in "${METHODS[@]}"; do
        IFS=':' read -r selector fuser <<< "$method"
        
        for seed in "${SEEDS[@]}"; do
            for lambda_llm in "${LAMBDA_LLM[@]}"; do
                for lambda_mixup in "${LAMBDA_MIXUP[@]}"; do
                    for lr in "${LEARNING_RATES[@]}"; do
                        completed_experiments=$((completed_experiments + 1))
                        
                        echo "[$completed_experiments/$total_experiments]"
                        
                        run_training "$selector" "$fuser" "$seed" "$lambda_llm" "$lambda_mixup" "$lr"
                        
                    done
                done
            done
        done
    done
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║ ✓ All grid search experiments completed!                           ║"
    echo "║   Total: $total_experiments experiments                             ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# Run main function
main "$@"
