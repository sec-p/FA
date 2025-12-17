#!/usr/bin/env python3
"""
Grid Search Orchestrator - Run multiple method/hyperparameter combinations
Supports parallel execution with job queue management
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp


@dataclass
class ExperimentConfig:
    """Single experiment configuration"""
    method: str
    seed: int
    lr: float
    batch_size: int
    epochs: int
    config_file: str
    experiment_id: str = ""
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = f"{self.method}_s{self.seed}_lr{self.lr}_bs{self.batch_size}"
    
    def to_command(self) -> List[str]:
        """Convert to train_and_eval.py command"""
        return [
            "python3", "train_and_eval.py",
            "--config", self.config_file,
            "--method", self.method,
            "--epochs", str(self.epochs),
            "--lr", str(self.lr),
            "--batch_size", str(self.batch_size),
            "--seed", str(self.seed),
        ]


class GridSearchOrchestrator:
    """Orchestrate grid search experiments"""
    
    def __init__(self, num_workers: int = 1, log_dir: str = "logs"):
        self.num_workers = num_workers
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.results_file = self.log_dir / "grid_search_results.json"
        self.results = {
            "start_time": datetime.now().isoformat(),
            "experiments": {},
            "summary": {}
        }
    
    @staticmethod
    def run_experiment(config: ExperimentConfig) -> Tuple[str, bool, str]:
        """Run a single experiment"""
        exp_id = config.experiment_id
        
        try:
            cmd = config.to_command()
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour timeout
            )
            
            if result.returncode == 0:
                return exp_id, True, "Completed successfully"
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                return exp_id, False, error_msg
        
        except subprocess.TimeoutExpired:
            return exp_id, False, "Timeout (24 hours)"
        except Exception as e:
            return exp_id, False, str(e)
    
    def run_grid_search(self, experiments: List[ExperimentConfig]):
        """Run all experiments with parallel workers"""
        total = len(experiments)
        completed = 0
        succeeded = 0
        failed = 0
        
        print(f"\n{'='*70}")
        print(f"Starting Grid Search: {total} experiments with {self.num_workers} workers")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Run with multiprocessing pool
        with mp.Pool(processes=self.num_workers) as pool:
            for exp_id, success, message in pool.imap_unordered(
                self.run_experiment, experiments
            ):
                completed += 1
                if success:
                    succeeded += 1
                    status = "✓"
                else:
                    failed += 1
                    status = "✗"
                
                # Log result
                self.results["experiments"][exp_id] = {
                    "status": "success" if success else "failed",
                    "message": message,
                    "completed_at": datetime.now().isoformat()
                }
                
                # Print progress
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = (total - completed) * avg_time / self.num_workers
                
                print(f"[{completed}/{total}] {status} {exp_id}")
                print(f"  Time: {message}")
                print(f"  ETA: {remaining/3600:.1f} hours\n")
        
        # Save results
        self.results["end_time"] = datetime.now().isoformat()
        self.results["summary"] = {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "duration_seconds": time.time() - start_time
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("Grid Search Summary")
        print("="*70)
        print(f"Total Experiments: {total}")
        print(f"  ✓ Succeeded: {succeeded}")
        print(f"  ✗ Failed: {failed}")
        print(f"Duration: {(time.time() - start_time)/3600:.1f} hours")
        print(f"Results saved to: {self.results_file}\n")


def generate_experiments(
    methods: List[str],
    seeds: List[int],
    learning_rates: List[float],
    batch_sizes: List[int],
    epochs: int,
    config_file: str,
) -> List[ExperimentConfig]:
    """Generate all experiment combinations"""
    experiments = []
    
    for method in methods:
        for seed in seeds:
            for lr in learning_rates:
                for bs in batch_sizes:
                    exp = ExperimentConfig(
                        method=method,
                        seed=seed,
                        lr=lr,
                        batch_size=bs,
                        epochs=epochs,
                        config_file=config_file,
                    )
                    experiments.append(exp)
    
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for methods and hyperparameters"
    )
    
    # Method options
    parser.add_argument(
        '-m', '--methods',
        nargs='+',
        default=['baseline_mean', 'baseline_attention', 'selector_mlp', 
                'selector_slot', 'fuser_attention', 'full_model'],
        help='Methods to search'
    )
    
    # Hyperparameter grids
    parser.add_argument(
        '-s', '--seeds',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='Random seeds'
    )
    parser.add_argument(
        '-lr', '--learning-rates',
        type=float,
        nargs='+',
        default=[0.0005, 0.001, 0.002],
        help='Learning rates'
    )
    parser.add_argument(
        '-bs', '--batch-sizes',
        type=int,
        nargs='+',
        default=[32, 64],
        help='Batch sizes'
    )
    
    # Training options
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '-c', '--config',
        default='configs/my_config.yaml',
        help='Config file'
    )
    
    # Parallel options
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=1,
        help='Number of parallel jobs'
    )
    parser.add_argument(
        '-l', '--log-dir',
        default='logs',
        help='Logging directory'
    )
    
    # Analysis
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis after grid search'
    )
    
    args = parser.parse_args()
    
    # Generate experiments
    experiments = generate_experiments(
        methods=args.methods,
        seeds=args.seeds,
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        epochs=args.epochs,
        config_file=args.config,
    )
    
    # Show statistics
    print(f"\n{'='*70}")
    print("Grid Search Configuration")
    print(f"{'='*70}")
    print(f"Methods:        {len(args.methods)} ({', '.join(args.methods)})")
    print(f"Seeds:          {args.seeds}")
    print(f"Learning Rates: {args.learning_rates}")
    print(f"Batch Sizes:    {args.batch_sizes}")
    print(f"Epochs:         {args.epochs}")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Parallel Jobs:  {args.jobs}")
    print(f"{'='*70}\n")
    
    # Confirm
    response = input("Start grid search? (y/n) ")
    if response.lower() != 'y':
        print("Cancelled")
        sys.exit(0)
    
    # Run grid search
    orchestrator = GridSearchOrchestrator(
        num_workers=args.jobs,
        log_dir=args.log_dir
    )
    
    orchestrator.run_grid_search(experiments)
    
    # Optional: run analysis
    if args.analyze:
        print("\nRunning analysis...")
        subprocess.run(["python3", "analyze_results.py", "--summary"])


if __name__ == '__main__':
    main()
