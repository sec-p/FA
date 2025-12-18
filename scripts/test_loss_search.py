#!/usr/bin/env python3
"""
Test script to verify loss function coefficient search functionality.
This script demonstrates how to run grid search over loss coefficients.
"""

import os
import sys
import subprocess

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define hyperparameter search grid
seeds = [42]
learning_rates = [0.001]
batch_sizes = [32]

# Loss function coefficients (the new hyperparameters)
lambda_llm_negatives_values = [0.1, 0.5]
lambda_mixup_values = [0.1, 0.5]
margin_values = [0.2, 0.5]

# Model components to test
selector_types = ['mlp', 'slot']
fuser_types = ['mean', 'query_attn']

def run_experiment(method_name, seed, lr, batch_size, lambda_llm_negatives, lambda_mixup, margin, selector_type, fuser_type):
    """Run a single experiment with given hyperparameters."""
    cmd = [
        'python3', os.path.join(project_root, 'src', 'train_and_eval.py'),
        '--method', method_name,
        '--seed', str(seed),
        '--lr', str(lr),
        '--batch_size', str(batch_size),
        '--lambda_llm_negatives', str(lambda_llm_negatives),
        '--lambda_mixup', str(lambda_mixup),
        '--margin', str(margin),
        '--selector_type', selector_type,
        '--fuser_type', fuser_type,
        '--epochs', '1',  # Just 1 epoch for testing
        '--shots', '1',    # Few shots for faster testing
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Success: {method_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {method_name}")
        print(f"  Error: {e.stderr}")
        return False

def main():
    """Run all experiments in the grid search."""
    print("Testing loss function coefficient search functionality...")
    print(f"Project root: {project_root}")
    print("=" * 80)
    
    total_runs = 0
    successful_runs = 0
    
    for seed in seeds:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for lambda_llm_negatives in lambda_llm_negatives_values:
                    for lambda_mixup in lambda_mixup_values:
                        for margin in margin_values:
                            for selector_type in selector_types:
                                for fuser_type in fuser_types:
                                    # Generate a descriptive method name
                                    method_name = f"test_{selector_type}_{fuser_type}_llmneg{lambda_llm_negatives}_mix{lambda_mixup}_margin{margin}_seed{seed}"
                                    
                                    total_runs += 1
                                    if run_experiment(method_name, seed, lr, batch_size, lambda_llm_negatives, lambda_mixup, margin, selector_type, fuser_type):
                                        successful_runs += 1
                                    
                                    print("=" * 80)
    
    print(f"\nTesting completed!")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Success rate: {successful_runs / total_runs * 100:.1f}%")

if __name__ == '__main__':
    main()
