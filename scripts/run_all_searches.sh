#!/bin/bash
# Run all per-method grid search scripts sequentially (edit parallelization as needed)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPTS=(
  "$SCRIPT_DIR/search_baseline_mean.sh"
  "$SCRIPT_DIR/search_baseline_attention.sh"
  "$SCRIPT_DIR/search_selector_mlp.sh"
  "$SCRIPT_DIR/search_selector_slot.sh"
  "$SCRIPT_DIR/search_fuser_attention.sh"
  "$SCRIPT_DIR/search_full_model.sh"
)

for s in "${SCRIPTS[@]}"; do
  echo "Executing: $s"
  bash "$s"
done

echo "All searches submitted."