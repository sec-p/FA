"""
Ablation Study Script - Modular OOD Detection Framework
Runs comprehensive experiments across selector/fuser/loss combinations.
"""

import os
import sys
import subprocess
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple


class AblationExperiment:
    """Manages ablation study experiments."""
    
    # Configuration combinations for ablation study
    GROUPS = {
        'baseline': {
            'name': 'Baseline (MLP Selector + Mean Fuser)',
            'configs': {
                'selector_type': 'mlp',
                'fuser_type': 'mean',
                'num_heads_selector': 4,
                'num_select': 49,
                'lambda_diversity': 0.0,  # Disabled
                'lambda_semantic_exclusion': 0.0,
            }
        },
        'structure': {
            'name': 'Structure (Slot Selector + Mean Fuser)',
            'configs': {
                'selector_type': 'slot',
                'fuser_type': 'mean',
                'num_slots': 4,
                'num_select': 49,
                'lambda_orthogonality': 0.01,
            }
        },
        'fusion': {
            'name': 'Fusion (MLP Selector + Query Attention Fuser)',
            'configs': {
                'selector_type': 'mlp',
                'fuser_type': 'query_attn',
                'num_heads_selector': 4,
                'num_heads_fuser': 4,
                'num_select': 49,
                'lambda_diversity': 0.01,
            }
        },
        'interaction': {
            'name': 'Interaction (MLP Selector + Self-Attention Fuser)',
            'configs': {
                'selector_type': 'mlp',
                'fuser_type': 'self_attn',
                'num_heads_selector': 4,
                'num_heads_fuser': 4,
                'num_select': 49,
                'lambda_diversity': 0.01,
            }
        },
        'full': {
            'name': 'Full (Slot Selector + Query Attention Fuser + All Losses)',
            'configs': {
                'selector_type': 'slot',
                'fuser_type': 'query_attn',
                'num_slots': 4,
                'num_heads_fuser': 4,
                'num_select': 49,
                'lambda_orthogonality': 0.01,
                'lambda_diversity': 0.01,
                'lambda_semantic_exclusion': 0.01,
                'use_semantic_exclusion': True,
            }
        },
    }
    
    def __init__(self, base_config_path: str, output_dir: str = './ablation_results'):
        self.base_config_path = base_config_path
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_cfg = yaml.load(f, Loader=yaml.Loader)
    
    def create_experiment_config(self, group_key: str, seed: int) -> Dict:
        """Create config for specific experiment group."""
        group_cfg = self.base_cfg.copy()
        
        # Update with group-specific settings
        for key, value in self.GROUPS[group_key]['configs'].items():
            group_cfg[key] = value
        
        # Set seed
        group_cfg['seed'] = seed
        
        return group_cfg
    
    def save_config(self, config: Dict, filepath: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config, f)
    
    def run_experiment(self, group_key: str, seed: int, gpu_id: int = 0) -> Dict:
        """Run single experiment."""
        print(f"\n{'='*70}")
        print(f"Running: {self.GROUPS[group_key]['name']} (Seed: {seed})")
        print(f"{'='*70}")
        
        # Create config
        exp_cfg = self.create_experiment_config(group_key, seed)
        
        # Save config
        exp_dir = os.path.join(
            self.output_dir, 
            group_key, 
            f"seed{seed}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        cfg_path = os.path.join(exp_dir, 'config.yaml')
        self.save_config(exp_cfg, cfg_path)
        
        # Run training
        cmd = [
            'python', 'train_modular.py',
            '--config', cfg_path,
            '--is_train', '1'
        ]
        
        # Set GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        try:
            result = subprocess.run(
                cmd, 
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"✓ Experiment completed successfully")
                # Parse results from log file
                log_path = os.path.join(cfg_path.replace('config.yaml', ''), 'log_train.txt')
                metrics = self._parse_log(log_path)
                return {
                    'status': 'completed',
                    'metrics': metrics,
                    'exp_dir': exp_dir,
                }
            else:
                print(f"✗ Experiment failed with error:")
                print(result.stderr)
                return {
                    'status': 'failed',
                    'error': result.stderr,
                }
        except subprocess.TimeoutExpired:
            print(f"✗ Experiment timed out")
            return {
                'status': 'timeout',
                'error': 'Experiment exceeded 1 hour time limit'
            }
        except Exception as e:
            print(f"✗ Unexpected error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _parse_log(self, log_path: str) -> Dict:
        """Parse metrics from log file."""
        metrics = {
            'best_test_acc': 0.0,
            'final_train_loss': 0.0,
        }
        
        if not os.path.exists(log_path):
            return metrics
        
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Extract best accuracy
                    if 'Best test accuracy' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            acc_str = parts[-1].strip()
                            metrics['best_test_acc'] = float(acc_str)
        except Exception as e:
            print(f"Warning: Could not parse log file: {e}")
        
        return metrics
    
    def run_all_groups(self, seeds: List[int] = [1, 2, 3], gpu_ids: List[int] = None):
        """Run all experiment groups with multiple seeds."""
        if gpu_ids is None:
            gpu_ids = [0]
        
        results_summary = {}
        
        for group_key, group_info in self.GROUPS.items():
            print(f"\n{'#'*70}")
            print(f"# GROUP: {group_key.upper()} - {group_info['name']}")
            print(f"{'#'*70}")
            
            results_summary[group_key] = {
                'name': group_info['name'],
                'seeds': {},
            }
            
            for seed in seeds:
                gpu_id = gpu_ids[seeds.index(seed) % len(gpu_ids)]
                
                result = self.run_experiment(group_key, seed, gpu_id)
                results_summary[group_key]['seeds'][seed] = result
                
                if result['status'] == 'completed':
                    print(f"  Test Acc: {result['metrics']['best_test_acc']:.4f}")
        
        # Compute group statistics
        self._compute_statistics(results_summary)
        
        # Save results
        self._save_results(results_summary)
        
        return results_summary
    
    def _compute_statistics(self, results_summary: Dict):
        """Compute mean and std for each group."""
        for group_key, group_results in results_summary.items():
            accs = []
            for seed, result in group_results['seeds'].items():
                if result['status'] == 'completed':
                    accs.append(result['metrics']['best_test_acc'])
            
            if accs:
                mean_acc = sum(accs) / len(accs)
                std_acc = (sum((x - mean_acc) ** 2 for x in accs) / len(accs)) ** 0.5
                
                group_results['mean_test_acc'] = mean_acc
                group_results['std_test_acc'] = std_acc
                group_results['num_completed'] = len(accs)
                
                print(f"\n{group_key.upper()}:")
                print(f"  Mean Test Acc: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"  Completed: {len(accs)}/{len(group_results['seeds'])}")
    
    def _save_results(self, results_summary: Dict):
        """Save results to JSON."""
        results_path = os.path.join(self.output_dir, 'ablation_results.json')
        
        # Convert to JSON-serializable format
        results_json = {}
        for group_key, group_results in results_summary.items():
            results_json[group_key] = {
                'name': group_results['name'],
                'mean_test_acc': group_results.get('mean_test_acc', 0.0),
                'std_test_acc': group_results.get('std_test_acc', 0.0),
                'num_completed': group_results.get('num_completed', 0),
                'seeds': {
                    str(seed): {
                        'status': result['status'],
                        'metrics': result.get('metrics', {}),
                    }
                    for seed, result in group_results.get('seeds', {}).items()
                }
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
    
    def generate_report(self, results_summary: Dict):
        """Generate markdown report."""
        report_path = os.path.join(self.output_dir, 'ABLATION_REPORT.md')
        
        with open(report_path, 'w') as f:
            f.write("# Modular OOD Detection - Ablation Study Report\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Group | Architecture | Mean Acc (%) | Std (%) | Status |\n")
            f.write("|-------|--------------|--------------|---------|--------|\n")
            
            for group_key, group_results in results_summary.items():
                mean = group_results.get('mean_test_acc', 0.0) * 100
                std = group_results.get('std_test_acc', 0.0) * 100
                completed = group_results.get('num_completed', 0)
                total = len(group_results.get('seeds', {}))
                status = f"{completed}/{total} completed"
                
                f.write(f"| {group_key} | {group_results['name']} | "
                       f"{mean:.2f} | {std:.2f} | {status} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for group_key, group_results in results_summary.items():
                f.write(f"### {group_key.upper()} - {group_results['name']}\n\n")
                
                for seed, result in group_results.get('seeds', {}).items():
                    status = result['status']
                    if status == 'completed':
                        acc = result['metrics']['best_test_acc'] * 100
                        f.write(f"- **Seed {seed}**: {acc:.2f}%\n")
                    else:
                        f.write(f"- **Seed {seed}**: {status}\n")
                
                if group_results.get('mean_test_acc'):
                    mean = group_results['mean_test_acc'] * 100
                    std = group_results['std_test_acc'] * 100
                    f.write(f"\n**Mean**: {mean:.2f}% ± {std:.2f}%\n\n")
        
        print(f"✓ Report generated: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation Study for Modular OOD Detection')
    parser.add_argument('--base_config', type=str, default='configs/my_config.yaml',
                       help='Base configuration file')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                       help='Output directory for results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                       help='Random seeds to run')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                       help='GPU IDs to use')
    parser.add_argument('--groups', type=str, nargs='+', default=None,
                       help='Specific groups to run (default: all)')
    
    args = parser.parse_args()
    
    # Initialize experiment manager
    ablation = AblationExperiment(args.base_config, args.output_dir)
    
    # Filter groups if specified
    if args.groups:
        ablation.GROUPS = {k: v for k, v in ablation.GROUPS.items() if k in args.groups}
    
    print("\n" + "="*70)
    print("Modular OOD Detection - Ablation Study")
    print("="*70)
    print(f"\nConfiguration Summary:")
    print(f"  Base Config: {args.base_config}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Seeds: {args.seeds}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Groups: {list(ablation.GROUPS.keys())}")
    print(f"\n{'='*70}\n")
    
    # Run all experiments
    results_summary = ablation.run_all_groups(seeds=args.seeds, gpu_ids=args.gpus)
    
    # Generate report
    ablation.generate_report(results_summary)
    
    print(f"\n{'='*70}")
    print("✓ Ablation study completed!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
