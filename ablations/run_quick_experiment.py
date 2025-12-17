"""
Quick Experiment Runner - Execute specific ablation groups
Simple interface for running experiments with flexible configuration.
"""

import os
import yaml
import argparse
from pathlib import Path
from run_ablation_study import AblationExperiment


def run_quick_experiment(group_key: str, base_config: str = 'configs/my_config.yaml', 
                        gpu_id: int = 0, output_dir: str = './ablation_results'):
    """Run a single experiment group quickly."""
    
    ablation = AblationExperiment(base_config, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Quick Run: {ablation.GROUPS[group_key]['name']}")
    print(f"{'='*70}\n")
    
    # Create and run experiment
    result = ablation.run_experiment(group_key, seed=42, gpu_id=gpu_id)
    
    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    if result['status'] == 'completed':
        print(f"  Test Accuracy: {result['metrics']['best_test_acc']:.4f}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    return result


def list_groups():
    """List all available experiment groups."""
    ablation = AblationExperiment('configs/my_config.yaml')
    
    print("\nAvailable Experiment Groups:\n")
    for key, group in ablation.GROUPS.items():
        print(f"  {key:15} - {group['name']}")
        for cfg_key, cfg_val in group['configs'].items():
            print(f"                    {cfg_key}: {cfg_val}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Quick Ablation Study Runner')
    parser.add_argument('--group', type=str, default=None,
                       help='Experiment group to run (baseline, structure, fusion, interaction, full)')
    parser.add_argument('--list', action='store_true',
                       help='List all available groups')
    parser.add_argument('--base_config', type=str, default='configs/my_config.yaml',
                       help='Base configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.list:
        list_groups()
        return
    
    if args.group is None:
        print("Error: Please specify a group with --group or use --list to see options")
        return
    
    # Run quick experiment
    result = run_quick_experiment(
        args.group, 
        args.base_config, 
        args.gpu,
        args.output_dir
    )


if __name__ == '__main__':
    main()
