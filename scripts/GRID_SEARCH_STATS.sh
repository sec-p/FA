#!/bin/bash
# Quick Reference: Grid Search Statistics

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    Grid Search Overview                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Method Combinations (Selector × Fuser):"
echo "  1. mlp × mean"
echo "  2. mlp × query_attn"
echo "  3. mlp × self_attn"
echo "  4. slot × mean"
echo "  5. slot × query_attn"
echo "  6. slot × self_attn"
echo ""

echo "Hyperparameter Grid:"
echo "  Seeds (--seed):              1, 2, 3              (3 values)"
echo "  Lambda LLM (--llm):          0.01, 0.05, 0.1      (3 values)"
echo "  Lambda Mixup (--mixup):      0.05, 0.1, 0.2       (3 values)"
echo "  Learning Rate (--lr):        0.0005, 0.001, 0.002 (3 values)"
echo ""

echo "Total Experiments per Method:"
EXPS_PER_METHOD=$((3 * 3 * 3 * 3))
echo "  $EXPS_PER_METHOD = 3 × 3 × 3 × 3"
echo ""

echo "Total Experiments:"
TOTAL_EXPS=$((6 * EXPS_PER_METHOD))
echo "  $TOTAL_EXPS = 6 methods × $EXPS_PER_METHOD per method"
echo ""

echo "Distribution (3 parallel parts):"
EXPS_PER_PART=$((TOTAL_EXPS / 3))
echo "  Part 1 (2 methods): $((2 * EXPS_PER_METHOD)) = $((2 * 3 * 3 * 3 * 3)) experiments"
echo "  Part 2 (2 methods): $((2 * EXPS_PER_METHOD)) = $((2 * 3 * 3 * 3 * 3)) experiments"
echo "  Part 3 (2 methods): $((2 * EXPS_PER_METHOD)) = $((2 * 3 * 3 * 3 * 3)) experiments"
echo "  ─────────────────────────────────────"
echo "  Total:             $TOTAL_EXPS experiments"
echo ""

echo "Estimated Runtime (per experiment):"
echo "  ~1-2 hours (depends on GPU/CPU)"
echo ""

echo "Estimated Total Time:"
HOURS_PER_EXP=1.5
SEQUENTIAL_HOURS=$(echo "$TOTAL_EXPS * $HOURS_PER_EXP / 60" | bc)
PARALLEL_HOURS=$(echo "$EXPS_PER_PART * $HOURS_PER_EXP / 60" | bc)
echo "  Sequential: ~${SEQUENTIAL_HOURS}h (not recommended)"
echo "  Parallel:   ~${PARALLEL_HOURS}h (recommended with 3 GPUs/processes)"
echo ""

echo "Output Structure:"
echo "  ./my_caches/imagenet/ViT-B-16/"
echo "  ├── selector_mlp_fuser_mean/seed1/      (log, model, eval_history.json)"
echo "  ├── selector_mlp_fuser_mean/seed2/"
echo "  ├── selector_mlp_fuser_query_attn/seed1/"
echo "  └── ... (more combinations)"
echo ""

echo "Commands:"
echo "  # Run all sequentially"
echo "  bash scripts/run_all_methods.sh"
echo ""
echo "  # Run 3 parts in parallel (recommended)"
echo "  bash scripts/run_parallel.sh"
echo ""
echo "  # Or run individually in separate terminals:"
echo "  bash scripts/run_methods_part1.sh &"
echo "  bash scripts/run_methods_part2.sh &"
echo "  bash scripts/run_methods_part3.sh &"
echo ""

echo "Monitoring:"
echo "  # Watch logs in real-time"
echo "  tail -f logs/part1.log"
echo "  tail -f logs/part2.log"
echo "  tail -f logs/part3.log"
echo ""

echo "Analysis:"
echo "  # After all experiments complete"
echo "  python scripts/analyze_results.py --cache_root ./my_caches --top_n 20"
echo ""

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║ Ready to start grid search? Run: bash scripts/run_parallel.sh     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
