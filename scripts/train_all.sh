#!/bin/bash
# ==============================================================================
# Unified Training and Evaluation Script
# Supports training all methods with per-epoch ID and OOD evaluation
# Optimized for Linux servers with checkpoint optimization
#
# Usage:
#   bash train_all.sh                                    # Default: baseline_mean, 5 epochs
#   bash train_all.sh -m baseline_mean -e 50            # Single method, 50 epochs
#   bash train_all.sh -m all -e 50                       # All 6 methods
#   bash train_all.sh -m all -e 50 -s 2 -lr 0.002 -bs 32  # Custom params
#   nohup bash train_all.sh -m all -e 50 > training.log 2>&1 &  # Background
# ==============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Available methods
AVAILABLE_METHODS=(
    "baseline_mean"
    "baseline_attention"
    "selector_mlp"
    "selector_slot"
    "fuser_attention"
    "full_model"
)

# Default parameters
DEFAULT_METHOD="baseline_mean"
DEFAULT_EPOCHS=5
DEFAULT_LR=0.001
DEFAULT_BATCH_SIZE=32
DEFAULT_SEED=42
DEFAULT_CONFIG="configs/my_config.yaml"

# Parse arguments
METHODS=()
EPOCHS=$DEFAULT_EPOCHS
LR=$DEFAULT_LR
BATCH_SIZE=$DEFAULT_BATCH_SIZE
SEED=$DEFAULT_SEED
CONFIG=$DEFAULT_CONFIG

print_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  bash train_all.sh [options]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -m, --method METHOD         Method name (or 'all'). Default: $DEFAULT_METHOD"
    echo "  -e, --epochs EPOCHS         Number of epochs. Default: $DEFAULT_EPOCHS"
    echo "  -lr, --learning-rate LR     Learning rate. Default: $DEFAULT_LR"
    echo "  -bs, --batch-size SIZE      Batch size. Default: $DEFAULT_BATCH_SIZE"
    echo "  -s, --seed SEED             Random seed. Default: $DEFAULT_SEED"
    echo "  -c, --config CONFIG         Config file. Default: $DEFAULT_CONFIG"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo -e "${BLUE}Available Methods:${NC}"
    for method in "${AVAILABLE_METHODS[@]}"; do
        echo "  - $method"
    done
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  # Quick test with single method"
    echo "  bash train_all.sh -m baseline_mean -e 5"
    echo ""
    echo "  # Full training with all methods"
    echo "  bash train_all.sh -m all -e 50"
    echo ""
    echo "  # Custom hyperparameters"
    echo "  bash train_all.sh -m all -e 50 -s 2 -lr 0.002 -bs 32"
    echo ""
    echo "  # Background execution"
    echo "  nohup bash train_all.sh -m all -e 50 > training.log 2>&1 &"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--method)
                shift
                if [[ "$1" == "all" ]]; then
                    METHODS=("${AVAILABLE_METHODS[@]}")
                else
                    METHODS=("$1")
                fi
                shift
                ;;
            -e|--epochs)
                EPOCHS=$2
                shift 2
                ;;
            -lr|--learning-rate)
                LR=$2
                shift 2
                ;;
            -bs|--batch-size)
                BATCH_SIZE=$2
                shift 2
                ;;
            -s|--seed)
                SEED=$2
                shift 2
                ;;
            -c|--config)
                CONFIG=$2
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
    
    # Set default method if none specified
    if [[ ${#METHODS[@]} -eq 0 ]]; then
        METHODS=("$DEFAULT_METHOD")
    fi
}

validate_config() {
    echo -e "${BLUE}Validating configuration...${NC}"
    
    # Check config file
    if [[ ! -f "$CONFIG" ]]; then
        echo -e "${RED}✗ Config file not found: $CONFIG${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Config file found${NC}"
    
    # Check if train_and_eval.py exists
    if [[ ! -f "train_and_eval.py" ]]; then
        echo -e "${RED}✗ train_and_eval.py not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Training script found${NC}"
    
    # Check Python and dependencies
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python3 not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python3 available${NC}"
    
    # Check PyTorch
    if ! python3 -c "import torch" 2>/dev/null; then
        echo -e "${RED}✗ PyTorch not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ PyTorch available${NC}"
}

print_config() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         Training Configuration                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "  Methods:        ${YELLOW}${METHODS[@]}${NC}"
    echo -e "  Epochs:         ${YELLOW}$EPOCHS${NC}"
    echo -e "  Learning Rate:  ${YELLOW}$LR${NC}"
    echo -e "  Batch Size:     ${YELLOW}$BATCH_SIZE${NC}"
    echo -e "  Seed:           ${YELLOW}$SEED${NC}"
    echo -e "  Config:         ${YELLOW}$CONFIG${NC}"
    echo ""
    
    TOTAL_JOBS=${#METHODS[@]}
    echo "  Total Jobs: $TOTAL_JOBS method(s)"
    echo ""
}

run_training() {
    local method=$1
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║ Training: $method"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    python3 train_and_eval.py \
        --config "$CONFIG" \
        --method "$method" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Completed: $method${NC}"
    else
        echo -e "${RED}✗ Failed: $method${NC}"
        return 1
    fi
}

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║    Modular OOD Detection - Training & Evaluation Script        ║"
    echo "║              Linux Server Optimized Version                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Parse arguments
    parse_arguments "$@"
    
    # Validate environment
    validate_config
    
    # Print configuration
    print_config
    
    # Ask for confirmation
    read -p "Continue with training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Training cancelled.${NC}"
        exit 0
    fi
    
    # Create results directory
    mkdir -p results
    
    # Run training for each method
    FAILED_METHODS=()
    for method in "${METHODS[@]}"; do
        if ! run_training "$method"; then
            FAILED_METHODS+=("$method")
        fi
    done
    
    # Summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                      Training Summary                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    SUCCEEDED=$((${#METHODS[@]} - ${#FAILED_METHODS[@]}))
    echo "  Total Methods:   ${#METHODS[@]}"
    echo -e "  ${GREEN}✓ Succeeded:${NC}     $SUCCEEDED"
    
    if [[ ${#FAILED_METHODS[@]} -gt 0 ]]; then
        echo -e "  ${RED}✗ Failed:${NC}        ${#FAILED_METHODS[@]}"
        echo ""
        echo -e "  ${RED}Failed methods:${NC}"
        for method in "${FAILED_METHODS[@]}"; do
            echo "    - $method"
        done
    fi
    
    echo ""
    echo -e "  ${BLUE}Results saved to:${NC} results/"
    echo ""
    
    # Exit with appropriate code
    if [[ ${#FAILED_METHODS[@]} -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

# Run main function
main "$@"
