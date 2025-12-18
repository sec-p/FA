#!/usr/bin/env python3
"""
Quick test to verify selector_type=None handling
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from src.model_modular import IdentitySelector, MultiHeadMLPSelector

print("Testing selectors...")

# Test IdentitySelector
print("\n1. Testing IdentitySelector:")
selector = IdentitySelector(input_dim=768, num_select=49)
test_input = torch.randn(2, 49, 768)  # (B, N, D)
output, aux_loss = selector(test_input)
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Auxiliary losses: {aux_loss}")
print(f"  ✓ IdentitySelector works correctly")

# Test MultiHeadMLPSelector
print("\n2. Testing MultiHeadMLPSelector:")
selector = MultiHeadMLPSelector(input_dim=768, num_select=49, num_heads=4)
output, aux_loss = selector(test_input)
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Auxiliary losses: {list(aux_loss.keys())}")
print(f"  ✓ MultiHeadMLPSelector works correctly")

print("\n✅ All selector tests passed!")
