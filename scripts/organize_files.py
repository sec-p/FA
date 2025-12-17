#!/usr/bin/env python3
"""
File Organization Script
Reorganizes project files into logical folders
WARNING: This will move files. Make sure to backup first!
"""

import os
import shutil
from pathlib import Path

def safe_move(src, dst_dir, dry_run=True):
    """
    Safely move file to destination directory.
    
    Args:
        src: Source file path
        dst_dir: Destination directory
        dry_run: If True, only print what would be done
    """
    if not os.path.exists(src):
        return False
    
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    
    if dry_run:
        print(f"  Would move: {src} -> {dst}")
    else:
        shutil.move(src, dst)
        print(f"  ✓ Moved: {src} -> {dst}")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize project files')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually move files (default is dry-run)')
    args = parser.parse_args()
    
    root = Path('.')
    
    print("\n" + "="*80)
    print("File Organization Script")
    print("="*80 + "\n")
    
    if not args.execute:
        print("DRY RUN MODE (use --execute to actually move files)\n")
    
    # Define file movements
    moves = {
        'docs': [
            'README.md',
            'QUICK_START.md',
            'QUICK_START_ICML.md',
            'PROJECT_OVERVIEW.md',
            'VERSION_GUIDE.md',
            'ABLATION_GUIDE.md',
            'EXPERIMENTS_README.md',
            'COMPLETION_SUMMARY.md',
            'DELIVERY_CHECKLIST.md',
            'DELIVER_SUMMARY.md',
            'FINAL_DELIVERY_REPORT.md',
            'README_ICML_COMPLETION.md',
            'IMPLEMENTATION_SUMMARY.md',
            'MODIFICATION_CHECKLIST.md',
            'START_HERE.md',
        ],
        'ablations': [
            'run_ablation_study.py',
            'run_quick_experiment.py',
            'verify_implementation.py',
            'test_eval_feature.py',
            'examples_icml_features.py',
        ],
        'scripts': [
            'analyze_results.py',
        ]
    }
    
    # Define files to delete (optional)
    delete_files = [
        'main.py',
        'main_v2.py',
        'model.py',
        'model_v2.py',
    ]
    
    # Move files
    print("Moving files to organized folders:\n")
    for dst_dir, files in moves.items():
        print(f"Moving to {dst_dir}/:")
        for file in files:
            if os.path.exists(file):
                safe_move(file, dst_dir, dry_run=not args.execute)
        print()
    
    # Show files to delete
    print("Files recommended for deletion (old versions):\n")
    print("  These files have been superseded by model_modular.py and train_modular.py:")
    for file in delete_files:
        if os.path.exists(file):
            print(f"  - {file}")
    
    if args.execute:
        response = input("\nDelete old files? (y/n): ")
        if response.lower() == 'y':
            for file in delete_files:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"  ✓ Deleted: {file}")
    else:
        print("\n  (use --execute to actually delete)")
    
    print("\n" + "="*80)
    if args.execute:
        print("✓ File organization complete!")
    else:
        print("Dry run complete. Use --execute to actually move files.")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
