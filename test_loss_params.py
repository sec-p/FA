#!/usr/bin/env python3
"""
Test script to verify loss function parameter handling in the modular OOD detection model.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_modular import build_modular_model
from src.train_and_eval import TrainEvalOrchestrator

def test_loss_parameters():
    """Test loss function parameters are correctly passed through the system."""
    print("Testing loss function parameter handling...")
    
    # Test parameter extraction from args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_llm_negatives', type=float, default=0.1)
    parser.add_argument('--lambda_mixup', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.2)
    
    # Test different parameter values
    test_cases = [
        (0.5, 0.3, 0.4),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.1),
    ]
    
    for llm_neg, mixup, margin in test_cases:
        args = parser.parse_args([
            f'--lambda_llm_negatives={llm_neg}',
            f'--lambda_mixup={mixup}',
            f'--margin={margin}'
        ])
        
        print(f"\nTest case: llm_neg={llm_neg}, mixup={mixup}, margin={margin}")
        print(f"  Args parsed correctly: {args.lambda_llm_negatives == llm_neg}, {args.lambda_mixup == mixup}, {args.margin == margin}")
    
    # Test model configuration generation
    print("\nTesting model configuration generation...")
    
    # Mock classnames
    mock_classnames = ["class1", "class2", "class3"]
    
    # Create mock clip model
    class MockCLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Identity()
            self.logit_scale = torch.tensor(100.0)
            self.dtype = torch.float32
    
    mock_clip = MockCLIP()
    
    # Test different loss parameter configurations
    test_configs = [
        {"lambda_llm_negatives": 0.5, "lambda_mixup": 0.3, "margin": 0.4},
        {"lambda_llm_negatives": 1.0, "lambda_mixup": 0.0, "margin": 0.1},
    ]
    
    for config in test_configs:
        cfg = {
            'device': torch.device('cpu'),
            'selector_type': 'mlp',
            'num_select': 16,
            'fuser_type': 'mean',
            **config,
            'templates': ["a photo of a"],
            'use_redundancy_loss': False,
            'use_llm_negatives': True,
            'use_semantic_exclusion': True,
            'use_mixup_invariance': True
        }
        
        print(f"\nModel config: {cfg}")
        
        # This will test if the config is valid
        try:
            model = build_modular_model(cfg, mock_classnames, mock_clip)
            print("  ✓ Model built successfully with config")
            
            # Test if parameters are accessible
            print(f"    - Model cfg has llm_negatives: {'lambda_llm_negatives' in model.cfg}")
            print(f"    - Model cfg has mixup: {'lambda_mixup' in model.cfg}")
            print(f"    - Model cfg has margin: {'margin' in model.cfg}")
        except Exception as e:
            print(f"  ✗ Model build failed: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_loss_parameters()
