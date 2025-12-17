"""
Quick test to demonstrate the new per-epoch evaluation feature.
This script shows how the enhanced evaluate() method records detailed metrics.
"""

import json
import argparse
import os

def load_eval_history(history_path: str):
    """Load and display evaluation history."""
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"Evaluation History ({len(history)} epochs)")
    print(f"{'='*80}\n")
    
    # Print header
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    print(f"{'-'*60}")
    
    # Print data
    best_acc = 0
    best_epoch = 0
    for record in history:
        epoch = record['epoch']
        train_loss = record['train_loss']
        train_acc = record['train_acc']
        test_loss = record['test_loss']
        test_acc = record['test_acc']
        
        print(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
    
    print(f"{'-'*60}")
    print(f"\nBest Test Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    
    # Print per-class accuracies for best epoch
    print(f"\nPer-class accuracies at Epoch {best_epoch}:")
    best_record = history[best_epoch - 1]
    if best_record['per_class_acc']:
        for class_name, acc in best_record['per_class_acc'].items():
            print(f"  {class_name:<30} {acc:.4f}")
    
    print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='my_caches/*/eval_history.json',
                       help='Path to eval_history.json file')
    args = parser.parse_args()
    
    # If using wildcard, find the actual file
    import glob
    if '*' in args.history:
        files = glob.glob(args.history)
        if files:
            history_path = files[0]
        else:
            print(f"No history files found matching pattern: {args.history}")
            return
    else:
        history_path = args.history
    
    load_eval_history(history_path)

if __name__ == '__main__':
    main()
