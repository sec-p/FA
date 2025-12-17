#!/bin/bash
# Parallel Grid Search Script 2/3
# Runs second set of method combinations

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Grid Search Part 2/3: Methods 3-4 with all hyperparameter combos ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

BASE_CONFIG="configs/my_config.yaml"

# Part 2: mlp:self_attn, slot:mean
METHODS=("mlp:self_attn" "slot:mean")
SEEDS=(1 2 3)
LAMBDA_LLM=(0.01 0.05 0.1)
LAMBDA_MIXUP=(0.05 0.1 0.2)
LRS=(0.0005 0.001 0.002)

run_training() {
    local selector=$1
    local fuser=$2
    local seed=$3
    local lambda_llm=$4
    local lambda_mixup=$5
    local lr=$6
    
    cp "$BASE_CONFIG" "configs/temp_grid.yaml"
    
    python3 << EOF
import yaml
with open('configs/temp_grid.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)
cfg['selector_type'] = '$selector'
cfg['fuser_type'] = '$fuser'
cfg['seed'] = $seed
cfg['lambda_llm_negatives'] = $lambda_llm
cfg['lambda_mixup'] = $lambda_mixup
cfg['lr'] = $lr
with open('configs/temp_grid.yaml', 'w') as f:
    yaml.dump(cfg, f)
EOF
    
    echo "[Part2] $selector + $fuser | seed=$seed, llm=$lambda_llm, mixup=$lambda_mixup, lr=$lr"
    python train_modular.py --config configs/temp_grid.yaml --is_train 1 || true
    rm -f configs/temp_grid.yaml
}

total=0
for m in "${METHODS[@]}"; do
    IFS=':' read -r sel fus <<< "$m"
    for s in "${SEEDS[@]}"; do
        for ll in "${LAMBDA_LLM[@]}"; do
            for lm in "${LAMBDA_MIXUP[@]}"; do
                for lr in "${LRS[@]}"; do
                    total=$((total + 1))
                    run_training "$sel" "$fus" "$s" "$ll" "$lm" "$lr"
                done
            done
        done
    done
done

echo ""
echo "✓ Part 2 completed ($total experiments)"
