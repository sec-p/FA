#!/bin/bash
# ==============================================================================
# Grid Search Training Script for Method and Hyperparameter Combinations
# Supports comprehensive grid search over methods, seeds, learning rates, etc.
# Reference: scripts/run_all_methods.sh (adapted and enhanced)
# ==============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ==============================================================================
# Grid Search Configuration
# ==============================================================================

# Method combinations (selector_type:fuser_type, or baseline variations)
declare -a METHODS=(
    "baseline_mean"
    "baseline_attention"
    "selector_mlp"
    "selector_slot"
    "fuser_attention"
    "full_model"
)

# Hyperparameter grids
declare -a SEEDS=(1 2 3)
declare -a LEARNING_RATES=(0.0005 0.001 0.002)
declare -a BATCH_SIZES=(32 64)

# Default values
DEFAULT_EPOCHS=50
DEFAULT_CONFIG="configs/my_config.yaml"

# ==============================================================================
# Functions
# ==============================================================================

print_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  bash grid_search.sh [options]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -m, --methods METHOD1 METHOD2 ...  Methods to train (or 'all'). Default: all"
    echo "  -e, --epochs EPOCHS                Epochs per experiment. Default: $DEFAULT_EPOCHS"
    echo "  -s, --seeds SEED1 SEED2 ...        Random seeds to try. Default: ${SEEDS[@]}"
    echo "  -lr, --learning-rates LR1 LR2 ...  Learning rates to try. Default: ${LEARNING_RATES[@]}"
    echo "  -bs, --batch-sizes BS1 BS2 ...     Batch sizes to try. Default: ${BATCH_SIZES[@]}"
    echo "  -c, --config CONFIG                Config file. Default: $DEFAULT_CONFIG"
    echo "  -p, --parallel N                   Run N experiments in parallel. Default: 1"
    echo "  -h, --help                         Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  # Grid search all methods with all hyperparameters"
    echo "  bash grid_search.sh"
    echo ""
    echo "  # Grid search specific methods"
    echo "  bash grid_search.sh -m baseline_mean selector_mlp full_model -e 20"
    echo ""
    echo "  # Grid search with limited seeds and learning rates"
    echo "  bash grid_search.sh -s 1 2 -lr 0.001 -bs 32"
    echo ""
    echo "  # Run 4 experiments in parallel"
    echo "  bash grid_search.sh -p 4"
}

parse_arguments() {
    local methods_specified=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--methods)
                shift
                methods_specified=true
                METHODS=()
                while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                    if [[ "$1" == "all" ]]; then
                        METHODS=("baseline_mean" "baseline_attention" "selector_mlp" "selector_slot" "fuser_attention" "full_model")
                        shift
                        break
                    else
                        METHODS+=("$1")
                        shift
                    fi
                done
                ;;
            -e|--epochs)
                DEFAULT_EPOCHS=$2
                shift 2
                ;;
            -s|--seeds)
                shift
                SEEDS=()
                while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                    SEEDS+=("$1")
                    shift
                done
                ;;
            -lr|--learning-rates)
                shift
                LEARNING_RATES=()
                while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                    LEARNING_RATES+=("$1")
                    shift
                done
                ;;
            -bs|--batch-sizes)
                shift
                BATCH_SIZES=()
                while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                    BATCH_SIZES+=("$1")
                    shift
                done
                ;;
            -c|--config)
                DEFAULT_CONFIG=$2
                shift 2
                ;;
            -p|--parallel)
                PARALLEL_JOBS=$2
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
}

calculate_statistics() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                 Grid Search Statistics                        ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    local total_methods=${#METHODS[@]}
    local total_seeds=${#SEEDS[@]}
    local total_lrs=${#LEARNING_RATES[@]}
    local total_bss=${#BATCH_SIZES[@]}
    
    local experiments_per_method=$((total_seeds * total_lrs * total_bss))
    local total_experiments=$((total_methods * experiments_per_method))
    
    echo "Method Combinations:"
    echo "  Methods: $total_methods (${METHODS[@]})"
    echo "  - ${METHODS[@]}"
    echo ""
    
    echo "Hyperparameter Grid:"
    echo "  Seeds:          ${SEEDS[@]} ($total_seeds values)"
    echo "  Learning Rates: ${LEARNING_RATES[@]} ($total_lrs values)"
    echo "  Batch Sizes:    ${BATCH_SIZES[@]} ($total_bss values)"
    echo ""
    
    echo "Total Experiments:"
    echo "  Per method:  $experiments_per_method = $total_seeds × $total_lrs × $total_bss"
    echo "  Total:       $total_experiments = $total_methods × $experiments_per_method"
    echo ""
    
    echo "Estimated Runtime (per experiment ~1-2 hours):"
    if [[ $total_experiments -le 10 ]]; then
        echo "  Sequential:  $((total_experiments * 1))-$((total_experiments * 2)) hours"
    else
        echo "  Sequential:  $((total_experiments * 1))-$((total_experiments * 2)) hours"
        echo "  With 4 parallel: $((total_experiments / 4))-$((total_experiments / 2)) hours"
    fi
    echo ""
}

run_training() {
    local method=$1
    local seed=$2
    local lr=$3
    local bs=$4
    
    local experiment_name="${method}_s${seed}_lr${lr}_bs${bs}"
    
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} Running: $experiment_name"
    
    if python3 train_and_eval.py \
        --config "$DEFAULT_CONFIG" \
        --method "$method" \
        --epochs "$DEFAULT_EPOCHS" \
        --lr "$lr" \
        --batch_size "$bs" \
        --seed "$seed" 2>&1 | tee -a "logs/grid_search.log"; then
        echo -e "${GREEN}✓ Completed: $experiment_name${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed: $experiment_name${NC}"
        return 1
    fi
}

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         Grid Search: Methods and Hyperparameters              ║"
    echo "║         (Reference: scripts/run_all_methods.sh)               ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Parse arguments
    PARALLEL_JOBS=1
    parse_arguments "$@"
    
    # Show statistics
    calculate_statistics
    
    # Ask for confirmation
    read -p "Start grid search with above configuration? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Grid search cancelled.${NC}"
        exit 0
    fi
    
    # Setup logging
    mkdir -p logs results
    > logs/grid_search.log
    
    echo -e "${BLUE}Starting grid search at $(date)${NC}"
    echo "Grid Search Started: $(date)" >> logs/grid_search.log
    
    # Generate experiment queue
    local experiment_count=0
    local success_count=0
    local failed_count=0
    declare -a failed_experiments
    
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for bs in "${BATCH_SIZES[@]}"; do
                    ((experiment_count++))
                    
                    # Run experiment
                    if run_training "$method" "$seed" "$lr" "$bs"; then
                        ((success_count++))
                    else
                        ((failed_count++))
                        failed_experiments+=("${method}_s${seed}_lr${lr}_bs${bs}")
                    fi
                    
                    # Show progress
                    echo -e "${BLUE}Progress: $experiment_count experiments completed${NC}"
                done
            done
        done
    done
    
    # Print summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              Grid Search Completed                            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Total Experiments: $experiment_count"
    echo -e "  ${GREEN}✓ Succeeded: $success_count${NC}"
    echo -e "  ${RED}✗ Failed: $failed_count${NC}"
    
    if [[ $failed_count -gt 0 ]]; then
        echo ""
        echo -e "${RED}Failed experiments:${NC}"
        for exp in "${failed_experiments[@]}"; do
            echo "  - $exp"
        done
    fi
    
    echo ""
    echo -e "${BLUE}Results saved to: results/${NC}"
    echo -e "${BLUE}Logs saved to: logs/grid_search.log${NC}"
    echo ""
    
    # Analyze results
    echo -e "${BLUE}Running result analysis...${NC}"
    python3 analyze_results.py --summary
    echo ""
}

# Run main function
main "$@"
