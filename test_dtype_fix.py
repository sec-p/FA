import torch
import clip
import sys
sys.path.append('./src')
from model_modular import build_modular_model

cfg = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'selector_type': 'mlp',
    'fuser_type': 'query_attn',
    'num_select': 49,
    'num_heads_selector': 4,
    'num_heads_fuser': 4,
    'templates': ['a photo of a {}'],
    'use_semantic_exclusion': True,
    'use_redundancy_loss': True,
    'use_llm_negatives': False,
    'use_mixup_invariance': False,
    'lambda_llm_negatives': 0.05,
    'lambda_mixup': 0.1,
    'margin': 0.1
}

classnames = ['dog', 'cat', 'bird']

# Test with different dtypes
for dtype in [torch.float32]:  # Start with float32 first
    print(f'\nTesting with dtype: {dtype}')
    clip_model, _ = clip.load('ViT-B/16', device=cfg['device'], dtype=dtype)
    clip_model = clip_model.to(cfg['device'])
    clip_model.eval()
    
    try:
        model = build_modular_model(cfg, classnames, clip_model)
        print(f'✓ Model built successfully')
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224).to(cfg['device'], dtype=dtype)
        labels = torch.tensor([0, 1]).to(cfg['device'])
        
        output = model(x, labels)
        print(f'✓ Forward pass successful')
        print(f'  logits shape: {output["logits"].shape}')
        print(f'  logits dtype: {output["logits"].dtype}')
        print(f'  aux_losses: {output["aux_losses"].keys()}')
        
        # Test loss computation
        total_loss = sum(output['aux_losses'].values())
        print(f'✓ Loss computation successful: {total_loss.item()}')
        
        # Test backward pass
        total_loss.backward()
        print(f'✓ Backward pass successful')
        
        print(f'✓ All tests passed for dtype: {dtype}')
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
