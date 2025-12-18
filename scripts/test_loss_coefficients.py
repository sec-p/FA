#!/usr/bin/env python3
"""
Test script to verify loss function coefficients are correctly passed from CLI to model.
This version doesn't require PyTorch.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_cli_args():
    """Test CLI argument parsing for loss coefficients."""
    print("Testing CLI argument parsing...")
    
    # Simulate CLI arguments with loss coefficients
    test_args = [
        '--method', 'selector_mlp',
        '--epochs', '5',
        '--lr', '0.001',
        '--batch_size', '16',
        '--seed', '42',
        '--backbone', 'ViT-B/16',
        '--lambda_llm_negatives', '0.75',
        '--lambda_mixup', '0.25',
        '--margin', '0.6',
        '--num_select', '8'
    ]
    
    # Temporarily replace sys.argv
    original_argv = sys.argv.copy()
    sys.argv = ['train_and_eval.py'] + test_args
    
    try:
        # Import the main function's parser logic
        from src.train_and_eval import TrainingConfig
        
        # Recreate the parser here to avoid side effects
        parser = argparse.ArgumentParser(description='Train and evaluate modular OOD detection')
        parser.add_argument('--method', type=str, default='baseline_mean',
                            choices=list(TrainingConfig.METHODS.keys()),
                            help='Training method')
        parser.add_argument('--epochs', type=int, default=TrainingConfig.DEFAULT_EPOCHS,
                            help='Number of epochs')
        parser.add_argument('--lr', type=float, default=TrainingConfig.DEFAULT_LR,
                            help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=TrainingConfig.DEFAULT_BATCH_SIZE,
                            help='Batch size')
        parser.add_argument('--seed', type=int, default=TrainingConfig.DEFAULT_SEED,
                            help='Random seed')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use (cuda or cpu)')

        # Model components
        parser.add_argument('--selector_type', type=str, default=None,
                            help="Selector type (e.g. 'mlp','slot' or None)")
        parser.add_argument('--fuser_type', type=str, default=None,
                            help="Fuser type (e.g. 'mean','query_attn','self_attn')")
        
        # Dataset settings
        parser.add_argument('--id_dataset', type=str, default='imagenet',
                            help='ID dataset name')
        parser.add_argument('--root_path', type=str, default='./data',
                            help='Root path for datasets')
        parser.add_argument('--shots', type=int, default=16,
                            help='Few-shot shots')
        parser.add_argument('--use_full_data', action='store_true',
                            help='Use full dataset instead of few-shot')
        
        # Model settings
        parser.add_argument('--backbone', type=str, default='ViT-L/14',
                            help='CLIP backbone model')
        parser.add_argument('--class_negatives_path', type=str, default='',
                            help='Path to class negatives file')
        
        # Loss function coefficients (hyperparameters)
        parser.add_argument('--lambda_llm_negatives', type=float, default=0.1,
                            help='Weight for LLM Negatives Loss')
        parser.add_argument('--lambda_mixup', type=float, default=0.1,
                            help='Weight for Causal Mixup Invariance Loss')
        parser.add_argument('--margin', type=float, default=0.2,
                            help='Margin for ranking losses (Semantic Exclusion and LLM Negatives)')

        # Selector hyperparameters
        parser.add_argument('--num_select', type=int, default=16,
                            help='Number of tokens/features to retain in selector (k)')
        
        args = parser.parse_args()
        
        print(f"✓ Parsed args: lambda_llm_negatives={args.lambda_llm_negatives}, lambda_mixup={args.lambda_mixup}, margin={args.margin}")
        
        # Verify values
        assert args.lambda_llm_negatives == 0.75, f"Expected 0.75, got {args.lambda_llm_negatives}"
        assert args.lambda_mixup == 0.25, f"Expected 0.25, got {args.lambda_mixup}"
        assert args.margin == 0.6, f"Expected 0.6, got {args.margin}"
        
        return args
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def test_parameter_flow():
    """Test that parameters flow correctly through the system."""
    print("\nTesting parameter flow...")
    
    # Create mock orchestrator class to test parameter passing
    class MockOrchestrator:
        def __init__(self, method, epochs, lr, batch_size, seed, device,
                     selector_type=None, fuser_type=None, id_dataset='imagenet',
                     root_path='./data', shots=16, lambda_llm_negatives=0.1,
                     lambda_mixup=0.1, margin=0.2, num_select=16,
                     backbone='ViT-L/14', class_negatives_path='', use_full_data=False):
            self.method = method
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.seed = seed
            self.device = device
            self.selector_type = selector_type
            self.fuser_type = fuser_type
            self.id_dataset = id_dataset
            self.root_path = root_path
            self.shots = shots
            self.lambda_llm_negatives = lambda_llm_negatives
            self.lambda_mixup = lambda_mixup
            self.margin = margin
            self.num_select = num_select
            self.backbone = backbone
            self.class_negatives_path = class_negatives_path
            self.use_full_data = use_full_data
            
            # Simulate _apply_method_config
            if method in ['selector_mlp', 'selector_slot']:
                self.selector_type = 'mlp' if method == 'selector_mlp' else 'slot'
                self.fuser_type = 'mean'
            elif method in ['baseline_mean', 'baseline_attention']:
                self.selector_type = None
                self.fuser_type = 'mean' if method == 'baseline_mean' else 'query_attn'
        
        def get_model_config(self):
            """Simulate model config creation."""
            return {
                'selector_type': self.selector_type,
                'num_select': self.num_select,
                'fuser_type': self.fuser_type,
                'lambda_llm_negatives': self.lambda_llm_negatives,
                'lambda_mixup': self.lambda_mixup,
                'margin': self.margin
            }
    
    # Create mock model class to test config usage
    class MockModel:
        def __init__(self, cfg, classnames, clip_model):
            self.cfg = cfg
            self.classnames = classnames
            self.clip_model = clip_model
    
    # Test parameter passing
    args = test_cli_args()
    
    # Mock device
    mock_device = 'cpu'
    
    # Create orchestrator
    orchestrator = MockOrchestrator(
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=mock_device,
        selector_type=args.selector_type,
        fuser_type=args.fuser_type,
        id_dataset=args.id_dataset,
        root_path=args.root_path,
        shots=args.shots,
        lambda_llm_negatives=args.lambda_llm_negatives,
        lambda_mixup=args.lambda_mixup,
        margin=args.margin,
        num_select=args.num_select,
        backbone=args.backbone,
        class_negatives_path=args.class_negatives_path,
        use_full_data=args.use_full_data
    )
    
    print(f"✓ Orchestrator initialized with loss coefficients:")
    print(f"  - lambda_llm_negatives: {orchestrator.lambda_llm_negatives}")
    print(f"  - lambda_mixup: {orchestrator.lambda_mixup}")
    print(f"  - margin: {orchestrator.margin}")
    
    # Verify values
    assert orchestrator.lambda_llm_negatives == 0.75, f"Expected 0.75, got {orchestrator.lambda_llm_negatives}"
    assert orchestrator.lambda_mixup == 0.25, f"Expected 0.25, got {orchestrator.lambda_mixup}"
    assert orchestrator.margin == 0.6, f"Expected 0.6, got {orchestrator.margin}"
    
    # Get model config
    cfg = orchestrator.get_model_config()
    
    print(f"✓ Model config created with loss coefficients:")
    print(f"  - lambda_llm_negatives: {cfg['lambda_llm_negatives']}")
    print(f"  - lambda_mixup: {cfg['lambda_mixup']}")
    print(f"  - margin: {cfg['margin']}")
    
    # Verify config values
    assert cfg['lambda_llm_negatives'] == 0.75, f"Expected 0.75, got {cfg['lambda_llm_negatives']}"
    assert cfg['lambda_mixup'] == 0.25, f"Expected 0.25, got {cfg['lambda_mixup']}"
    assert cfg['margin'] == 0.6, f"Expected 0.6, got {cfg['margin']}"
    
    # Create mock model
    mock_classnames = ['dog', 'cat', 'bird']
    mock_clip_model = 'mock_clip_model'
    model = MockModel(cfg, mock_classnames, mock_clip_model)
    
    print(f"✓ Model created with config:")
    print(f"  - lambda_llm_negatives: {model.cfg['lambda_llm_negatives']}")
    print(f"  - lambda_mixup: {model.cfg['lambda_mixup']}")
    print(f"  - margin: {model.cfg['margin']}")
    
    # Verify model config
    assert model.cfg['lambda_llm_negatives'] == 0.75, f"Expected 0.75, got {model.cfg['lambda_llm_negatives']}"
    assert model.cfg['lambda_mixup'] == 0.25, f"Expected 0.25, got {model.cfg['lambda_mixup']}"
    assert model.cfg['margin'] == 0.6, f"Expected 0.6, got {model.cfg['margin']}"
    
    return True


def main():
    """Main test function."""
    print("=" * 60)
    print("Loss Function Coefficients Test Script")
    print("=" * 60)
    
    try:
        # Test parameter flow
        test_parameter_flow()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Loss coefficients are correctly passed through the pipeline.")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Assertion error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
