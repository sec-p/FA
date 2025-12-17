"""
Evaluation and Comparison Script for Ablation Study Results
Analyzes and visualizes results from different experiment groups.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class ResultsAnalyzer:
    """Analyze and compare ablation study results."""
    
    def __init__(self, results_dir: str = './ablation_results'):
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, 'ablation_results.json')
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        """Load results from JSON file."""
        if not os.path.exists(self.results_file):
            print(f"Error: Results file not found at {self.results_file}")
            return {}
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def print_summary(self):
        """Print summary of all experiments."""
        if not self.results:
            print("No results found.")
            return
        
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        print(f"\n{'Group':<15} {'Name':<40} {'Accuracy':<12} {'Std Dev':<10}")
        print("-"*80)
        
        for group_key, group_data in sorted(self.results.items()):
            name = group_data.get('name', 'Unknown')[:38]
            mean_acc = group_data.get('mean_test_acc', 0.0) * 100
            std_acc = group_data.get('std_test_acc', 0.0) * 100
            
            print(f"{group_key:<15} {name:<40} {mean_acc:>6.2f}%    ±{std_acc:<7.2f}%")
        
        print("-"*80)
    
    def print_detailed_results(self):
        """Print detailed results for each group."""
        if not self.results:
            print("No results found.")
            return
        
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for group_key, group_data in self.results.items():
            print(f"\n{group_key.upper()}: {group_data['name']}")
            print("-"*80)
            
            seeds_data = group_data.get('seeds', {})
            for seed, seed_result in sorted(seeds_data.items(), key=lambda x: int(x[0])):
                status = seed_result.get('status', 'unknown')
                
                if status == 'completed':
                    acc = seed_result['metrics'].get('best_test_acc', 0.0) * 100
                    print(f"  Seed {seed:<3}: {acc:>6.2f}% ✓")
                else:
                    print(f"  Seed {seed:<3}: {status}")
            
            if 'mean_test_acc' in group_data:
                mean = group_data['mean_test_acc'] * 100
                std = group_data['std_test_acc'] * 100
                completed = group_data['num_completed']
                total = len(seeds_data)
                print(f"  {'─'*40}")
                print(f"  Mean: {mean:.2f}% ± {std:.2f}% ({completed}/{total})")
    
    def get_best_group(self) -> Tuple[str, float]:
        """Get best performing group."""
        if not self.results:
            return None, 0.0
        
        best_group = None
        best_acc = -1
        
        for group_key, group_data in self.results.items():
            mean_acc = group_data.get('mean_test_acc', 0.0)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_group = group_key
        
        return best_group, best_acc * 100
    
    def get_improvement_over_baseline(self) -> Dict[str, float]:
        """Calculate improvement over baseline."""
        if 'baseline' not in self.results:
            print("Warning: Baseline group not found")
            return {}
        
        baseline_acc = self.results['baseline'].get('mean_test_acc', 0.0)
        improvements = {}
        
        for group_key, group_data in self.results.items():
            if group_key != 'baseline':
                mean_acc = group_data.get('mean_test_acc', 0.0)
                improvement = (mean_acc - baseline_acc) * 100
                improvements[group_key] = improvement
        
        return improvements
    
    def print_comparison(self):
        """Print comparison with baseline."""
        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE")
        print("="*80)
        
        if 'baseline' not in self.results:
            print("Baseline group not found.")
            return
        
        baseline_acc = self.results['baseline'].get('mean_test_acc', 0.0) * 100
        baseline_std = self.results['baseline'].get('std_test_acc', 0.0) * 100
        
        print(f"\nBaseline: {baseline_acc:.2f}% ± {baseline_std:.2f}%\n")
        print(f"{'Group':<15} {'Accuracy':<15} {'Improvement':<15} {'Status'}")
        print("-"*80)
        
        improvements = self.get_improvement_over_baseline()
        
        for group_key, group_data in sorted(self.results.items()):
            if group_key == 'baseline':
                continue
            
            name = group_key
            acc = group_data.get('mean_test_acc', 0.0) * 100
            std = group_data.get('std_test_acc', 0.0) * 100
            
            if group_key in improvements:
                improvement = improvements[group_key]
                symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                print(f"{name:<15} {acc:>6.2f}% ±{std:>5.2f}%  "
                     f"{symbol} {abs(improvement):>6.2f}%")
            else:
                print(f"{name:<15} {acc:>6.2f}% ±{std:>5.2f}%")
        
        print("-"*80)
    
    def generate_markdown_table(self) -> str:
        """Generate markdown table for publication."""
        if not self.results:
            return "No results available."
        
        lines = [
            "## Ablation Study Results\n",
            "| Group | Architecture | Accuracy (%) | Std (%) | Improvement |\n",
            "|-------|--------------|--------------|---------|-------------|\n"
        ]
        
        improvements = self.get_improvement_over_baseline()
        
        for group_key, group_data in sorted(self.results.items()):
            name = group_data.get('name', 'Unknown')
            acc = group_data.get('mean_test_acc', 0.0) * 100
            std = group_data.get('std_test_acc', 0.0) * 100
            
            if group_key == 'baseline':
                improvement = "Baseline"
            elif group_key in improvements:
                improvement = f"+{improvements[group_key]:.2f}%"
            else:
                improvement = "N/A"
            
            lines.append(f"| {group_key} | {name} | {acc:.2f} | {std:.2f} | {improvement} |\n")
        
        return "".join(lines)
    
    def save_markdown_report(self, output_path: str = None):
        """Save detailed markdown report."""
        if output_path is None:
            output_path = os.path.join(self.results_dir, 'ANALYSIS_REPORT.md')
        
        with open(output_path, 'w') as f:
            f.write("# Ablation Study - Detailed Analysis Report\n\n")
            
            # Best group
            best_group, best_acc = self.get_best_group()
            f.write(f"## Best Performing Group\n\n")
            f.write(f"**{best_group}**: {best_acc:.2f}%\n\n")
            
            # Summary table
            f.write(self.generate_markdown_table())
            f.write("\n")
            
            # Detailed results
            f.write("## Detailed Results by Group\n\n")
            
            for group_key, group_data in sorted(self.results.items()):
                f.write(f"### {group_key.upper()}\n\n")
                f.write(f"**Name**: {group_data.get('name', 'Unknown')}\n\n")
                
                seeds_data = group_data.get('seeds', {})
                f.write("| Seed | Status | Accuracy |\n")
                f.write("|------|--------|----------|\n")
                
                for seed in sorted(seeds_data.keys(), key=lambda x: int(x)):
                    seed_result = seeds_data[seed]
                    status = seed_result.get('status', 'unknown')
                    
                    if status == 'completed':
                        acc = seed_result['metrics'].get('best_test_acc', 0.0) * 100
                        f.write(f"| {seed} | ✓ | {acc:.2f}% |\n")
                    else:
                        f.write(f"| {seed} | ✗ {status} | N/A |\n")
                
                if 'mean_test_acc' in group_data:
                    mean = group_data['mean_test_acc'] * 100
                    std = group_data['std_test_acc'] * 100
                    f.write(f"\n**Mean**: {mean:.2f}% ± {std:.2f}%\n\n")
        
        print(f"✓ Report saved to {output_path}")
    
    def export_csv(self, output_path: str = None) -> str:
        """Export results as CSV."""
        if output_path is None:
            output_path = os.path.join(self.results_dir, 'ablation_results.csv')
        
        lines = ["Group,Name,Mean_Accuracy,Std_Dev,Completed_Seeds,Total_Seeds\n"]
        
        for group_key, group_data in sorted(self.results.items()):
            name = group_data.get('name', 'Unknown').replace(',', ';')
            mean = group_data.get('mean_test_acc', 0.0) * 100
            std = group_data.get('std_test_acc', 0.0) * 100
            completed = group_data.get('num_completed', 0)
            total = len(group_data.get('seeds', {}))
            
            lines.append(f"{group_key},{name},{mean:.2f},{std:.2f},{completed},{total}\n")
        
        with open(output_path, 'w') as f:
            f.writelines(lines)
        
        print(f"✓ CSV exported to {output_path}")
        return output_path
    
    def calculate_statistics(self) -> Dict:
        """Calculate aggregate statistics."""
        stats = {
            'total_groups': len(self.results),
            'completed_groups': 0,
            'total_experiments': 0,
            'completed_experiments': 0,
            'mean_accuracy': 0.0,
            'best_accuracy': 0.0,
            'worst_accuracy': 1.0,
        }
        
        accuracies = []
        
        for group_data in self.results.values():
            if group_data.get('mean_test_acc'):
                stats['completed_groups'] += 1
                accuracies.append(group_data['mean_test_acc'])
            
            stats['total_experiments'] += len(group_data.get('seeds', {}))
            stats['completed_experiments'] += group_data.get('num_completed', 0)
        
        if accuracies:
            stats['mean_accuracy'] = np.mean(accuracies) * 100
            stats['best_accuracy'] = max(accuracies) * 100
            stats['worst_accuracy'] = min(accuracies) * 100
        
        return stats
    
    def print_statistics(self):
        """Print aggregate statistics."""
        stats = self.calculate_statistics()
        
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        print(f"\nTotal Groups: {stats['total_groups']}")
        print(f"Completed Groups: {stats['completed_groups']}/{stats['total_groups']}")
        print(f"Total Experiments: {stats['total_experiments']}")
        print(f"Completed: {stats['completed_experiments']}/{stats['total_experiments']}")
        print(f"\nAccuracy Range: {stats['worst_accuracy']:.2f}% - {stats['best_accuracy']:.2f}%")
        print(f"Mean Accuracy: {stats['mean_accuracy']:.2f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Ablation Study Results')
    parser.add_argument('--results_dir', type=str, default='./ablation_results',
                       help='Results directory')
    parser.add_argument('--summary', action='store_true', default=True,
                       help='Print summary')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed results')
    parser.add_argument('--comparison', action='store_true',
                       help='Print comparison with baseline')
    parser.add_argument('--statistics', action='store_true',
                       help='Print statistics')
    parser.add_argument('--report', action='store_true',
                       help='Generate markdown report')
    parser.add_argument('--csv', action='store_true',
                       help='Export as CSV')
    parser.add_argument('--all', action='store_true',
                       help='Print all analyses')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir)
    
    if not analyzer.results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"\nLoading results from: {args.results_dir}")
    
    # Default: print summary
    if args.summary or (not any([args.detailed, args.comparison, args.statistics, 
                                args.report, args.csv, args.all])):
        analyzer.print_summary()
    
    if args.detailed or args.all:
        analyzer.print_detailed_results()
    
    if args.comparison or args.all:
        analyzer.print_comparison()
    
    if args.statistics or args.all:
        analyzer.print_statistics()
    
    if args.report or args.all:
        analyzer.save_markdown_report()
    
    if args.csv or args.all:
        analyzer.export_csv()
    
    print()


if __name__ == '__main__':
    main()
