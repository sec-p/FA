"""
Training Results Analysis Tool
Analyze and compare results from different methods and seeds.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class ResultsAnalyzer:
    """Analyze training results."""
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        self.all_results = self._load_all_results()
    
    def _load_all_results(self) -> Dict:
        """Load all results from result directories."""
        all_results = defaultdict(lambda: defaultdict(list))
        
        if not os.path.exists(self.results_dir):
            print(f"Results directory not found: {self.results_dir}")
            return all_results
        
        # Find all results.json files
        for root, dirs, files in os.walk(self.results_dir):
            if 'results.json' in files:
                results_file = os.path.join(root, 'results.json')
                
                # Extract method and seed from path
                # Path format: results/method_seed_timestamp/results.json
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2:
                    dir_name = path_parts[-1]
                    parts = dir_name.rsplit('_', 2)  # Split from right to separate timestamp
                    
                    if len(parts) >= 2:
                        method = parts[0]
                        seed = parts[1]
                        
                        try:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            all_results[method][seed] = results
                        except Exception as e:
                            print(f"Error loading {results_file}: {e}")
        
        return all_results
    
    def show_latest(self):
        """Show latest result."""
        if not self.all_results:
            print("No results found.")
            return
        
        # Find latest results
        latest_method = max(self.all_results.keys(), 
                           key=lambda m: max([max([len(r) for r in self.all_results[m].values()]) 
                                            for seed, results in self.all_results[m].items()]))
        
        latest_seed = list(self.all_results[latest_method].keys())[-1]
        results = self.all_results[latest_method][latest_seed]
        
        # Show final epoch
        final_epoch = results[-1]
        
        print("\n" + "="*80)
        print(f"Latest Results: {latest_method} (seed={latest_seed})")
        print("="*80)
        print(f"\nFinal Epoch (Epoch {final_epoch['epoch']+1}):")
        print(f"  Train Loss: {final_epoch['train_loss']:.4f}")
        print(f"  Train Acc:  {final_epoch['train_acc']:.2f}%")
        print(f"  ID Accuracy: {final_epoch['id_accuracy']:.2f}%")
        print(f"\n  OOD Results:")
        print(f"    sun397:       AUROC={final_epoch['sun397_auroc']:.2f}%, FPR95={final_epoch['sun397_fpr95']:.2f}%")
        print(f"    dtd:          AUROC={final_epoch['dtd_auroc']:.2f}%, FPR95={final_epoch['dtd_fpr95']:.2f}%")
        print(f"    eurosat:      AUROC={final_epoch['eurosat_auroc']:.2f}%, FPR95={final_epoch['eurosat_fpr95']:.2f}%")
        print(f"    oxford_pets:  AUROC={final_epoch['oxford_pets_auroc']:.2f}%, FPR95={final_epoch['oxford_pets_fpr95']:.2f}%")
        print(f"  Avg OOD AUROC: {final_epoch['avg_ood_auroc']:.2f}%")
        print("="*80 + "\n")
    
    def compare_methods(self):
        """Compare performance across methods."""
        if not self.all_results:
            print("No results found.")
            return
        
        print("\n" + "="*80)
        print("Method Comparison (Latest Epoch)")
        print("="*80)
        print(f"\n{'Method':<20} {'Seed':<6} {'ID Acc':<12} {'Avg AUROC':<12} {'Avg FPR95':<12}")
        print("-"*80)
        
        for method in sorted(self.all_results.keys()):
            for seed in sorted(self.all_results[method].keys()):
                results = self.all_results[method][seed]
                final = results[-1]
                
                id_acc = final['id_accuracy']
                avg_auroc = final['avg_ood_auroc']
                avg_fpr95 = final['avg_ood_fpr95']
                
                print(f"{method:<20} {seed:<6} {id_acc:>10.2f}% {avg_auroc:>10.2f}% {avg_fpr95:>10.2f}%")
        
        print("="*80 + "\n")
    
    def compare_seeds(self):
        """Compare stability across seeds for each method."""
        if not self.all_results:
            print("No results found.")
            return
        
        print("\n" + "="*80)
        print("Seed Stability Analysis")
        print("="*80)
        
        for method in sorted(self.all_results.keys()):
            seeds_data = self.all_results[method]
            
            id_accs = []
            aurocs = []
            fpr95s = []
            
            for seed in sorted(seeds_data.keys()):
                results = seeds_data[seed]
                final = results[-1]
                
                id_accs.append(final['id_accuracy'])
                aurocs.append(final['avg_ood_auroc'])
                fpr95s.append(final['avg_ood_fpr95'])
            
            id_accs = np.array(id_accs)
            aurocs = np.array(aurocs)
            fpr95s = np.array(fpr95s)
            
            print(f"\n{method}:")
            print(f"  ID Accuracy:    {id_accs.mean():.2f}% ± {id_accs.std():.2f}%")
            print(f"  Avg OOD AUROC:  {aurocs.mean():.2f}% ± {aurocs.std():.2f}%")
            print(f"  Avg OOD FPR95:  {fpr95s.mean():.2f}% ± {fpr95s.std():.2f}%")
        
        print("\n" + "="*80 + "\n")
    
    def show_summary(self):
        """Generate comprehensive summary."""
        if not self.all_results:
            print("No results found.")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY")
        print("="*80)
        
        # Aggregate results by method
        method_stats = {}
        
        for method in sorted(self.all_results.keys()):
            seeds_data = self.all_results[method]
            
            id_accs = []
            aurocs = []
            
            for seed in sorted(seeds_data.keys()):
                results = seeds_data[seed]
                final = results[-1]
                id_accs.append(final['id_accuracy'])
                aurocs.append(final['avg_ood_auroc'])
            
            id_accs = np.array(id_accs)
            aurocs = np.array(aurocs)
            
            method_stats[method] = {
                'id_acc_mean': id_accs.mean(),
                'id_acc_std': id_accs.std(),
                'auroc_mean': aurocs.mean(),
                'auroc_std': aurocs.std(),
                'num_seeds': len(seeds_data),
            }
        
        # Sort by average AUROC
        sorted_methods = sorted(method_stats.items(), 
                               key=lambda x: x[1]['auroc_mean'], 
                               reverse=True)
        
        print("\n" + "Ranking by Average OOD AUROC:")
        print(f"{'Rank':<6} {'Method':<25} {'ID Acc':<20} {'OOD AUROC':<20}")
        print("-"*80)
        
        for rank, (method, stats) in enumerate(sorted_methods, 1):
            id_acc_str = f"{stats['id_acc_mean']:.2f}% ± {stats['id_acc_std']:.2f}%"
            auroc_str = f"{stats['auroc_mean']:.2f}% ± {stats['auroc_std']:.2f}%"
            print(f"{rank:<6} {method:<25} {id_acc_str:<20} {auroc_str:<20}")
        
        print("\n" + "="*80)
        print("Summary Statistics:")
        print("-"*80)
        
        best_method, best_stats = sorted_methods[0]
        print(f"\n✓ Best Method: {best_method}")
        print(f"  ID Accuracy:   {best_stats['id_acc_mean']:.2f}% ± {best_stats['id_acc_std']:.2f}%")
        print(f"  Avg OOD AUROC: {best_stats['auroc_mean']:.2f}% ± {best_stats['auroc_std']:.2f}%")
        print(f"  Seeds tested:  {best_stats['num_seeds']}")
        
        print("\n" + "="*80 + "\n")
    
    def export_csv(self, output_file: str = 'results_summary.csv'):
        """Export results to CSV."""
        import csv
        
        if not self.all_results:
            print("No results found.")
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Seed', 'ID_Accuracy', 'Avg_OOD_AUROC', 'Avg_OOD_FPR95'])
            
            for method in sorted(self.all_results.keys()):
                for seed in sorted(self.all_results[method].keys()):
                    results = self.all_results[method][seed]
                    final = results[-1]
                    
                    writer.writerow([
                        method,
                        seed,
                        f"{final['id_accuracy']:.4f}",
                        f"{final['avg_ood_auroc']:.4f}",
                        f"{final['avg_ood_fpr95']:.4f}",
                    ])
        
        print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--latest', action='store_true',
                       help='Show latest result')
    parser.add_argument('--compare-methods', action='store_true',
                       help='Compare methods')
    parser.add_argument('--compare-seeds', action='store_true',
                       help='Compare seeds (stability analysis)')
    parser.add_argument('--summary', action='store_true',
                       help='Show comprehensive summary')
    parser.add_argument('--export-csv', type=str,
                       help='Export results to CSV file')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir)
    
    # If no action specified, show summary
    if not any([args.latest, args.compare_methods, args.compare_seeds, args.summary, args.export_csv]):
        analyzer.show_summary()
    else:
        if args.latest:
            analyzer.show_latest()
        if args.compare_methods:
            analyzer.compare_methods()
        if args.compare_seeds:
            analyzer.compare_seeds()
        if args.summary:
            analyzer.show_summary()
        if args.export_csv:
            analyzer.export_csv(args.export_csv)


if __name__ == '__main__':
    main()
