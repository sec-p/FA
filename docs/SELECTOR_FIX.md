# selector_type=None 问题修复

## 问题描述

运行 `baseline_mean` 或 `baseline_attention` 方法时出现错误：
```
ValueError: Unknown selector_type: None
```

## 根本原因

在 `train_and_eval.py` 中，基线方法定义为：
```python
'baseline_mean': {'selector_type': None, 'fuser_type': 'mean'},
'baseline_attention': {'selector_type': None, 'fuser_type': 'query_attn'},
```

但在 `model_modular.py` 中的 `ModularCustomCLIP.__init__()` 方法只处理了具体的选择器类型（'mlp'、'slot'），没有处理 `selector_type=None` 的情况。

## 修复方案

### 1. 创建 IdentitySelector 类

在 `src/model_modular.py` 的 PART 2 中添加一个"无操作"选择器：

```python
class IdentitySelector(BaseSelector):
    """No-op selector that returns all features as-is (for baseline methods)."""
    
    def __init__(self, input_dim: int, num_select: int, cfg: Dict = None):
        super().__init__(input_dim, num_select, cfg)
    
    def forward(self, local_feats: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Returns all features without any selection.
        
        Args:
            local_feats: (B, N, D)
        
        Returns:
            selected_feats: (B, N, D) - all features unchanged
            aux_loss: empty dict
        """
        return local_feats, {}
```

### 2. 更新 ModularCustomCLIP 初始化

在 `ModularCustomCLIP.__init__()` 中处理 `selector_type=None`：

```python
# Initialize selector
selector_type = cfg.get('selector_type', 'mlp')
num_select = cfg.get('num_select', 49)

if selector_type is None:
    # Baseline methods: no feature selection, use identity selector
    self.selector = IdentitySelector(self.feat_dim, num_select, cfg)
elif selector_type == 'mlp':
    num_heads = cfg.get('num_heads_selector', 4)
    self.selector = MultiHeadMLPSelector(self.feat_dim, num_select, num_heads, cfg)
elif selector_type == 'slot':
    num_slots = cfg.get('num_slots', 4)
    self.selector = SparseSlotAttentionSelector(self.feat_dim, num_select, num_slots, cfg)
else:
    raise ValueError(f"Unknown selector_type: {selector_type}")
```

## 工作原理

### 基线方法流程

**baseline_mean**:
```
Image → CLIP Image Encoder → Local Features (B, N, D)
                           ↓
                    IdentitySelector (返回所有特征)
                           ↓
                    MeanPoolFuser (平均所有特征)
                           ↓
                         Final Features (B, D)
                           ↓
                        分类和OOD检测
```

**baseline_attention**:
```
Image → CLIP Image Encoder → Local Features (B, N, D)
                           ↓
                    IdentitySelector (返回所有特征)
                           ↓
              QueryGuidedAttentionFuser (使用文本特征引导注意力融合)
                           ↓
                         Final Features (B, D)
                           ↓
                        分类和OOD检测
```

### 高级方法流程

**selector_mlp + fuser_mean** (selector_mlp):
```
Image → CLIP Image Encoder → Local Features (B, N, D)
                           ↓
              MultiHeadMLPSelector (选择K个最重要特征)
                           ↓
                    MeanPoolFuser (平均K个特征)
                           ↓
                         Final Features (B, D)
```

## 修复内容

✅ **已修复**:
- 添加 `IdentitySelector` 类处理 `selector_type=None`
- 更新 `ModularCustomCLIP.__init__()` 初始化逻辑
- 支持所有6种方法的完整流程

## 验证修复

```bash
# 测试基线方法
python run_training.py --method baseline_mean --epochs 5

# 测试其他方法
python run_training.py --method selector_mlp --epochs 5
python run_training.py --method full_model --epochs 5
```

## 6种方法的选择器配置

| 方法 | selector_type | fuser_type | 描述 |
|------|---------------|-----------|------|
| baseline_mean | None | mean | 所有特征 + 平均融合 |
| baseline_attention | None | query_attn | 所有特征 + 注意力融合 |
| selector_mlp | mlp | mean | MLP选择 + 平均融合 |
| selector_slot | slot | mean | Slot注意力选择 + 平均融合 |
| fuser_attention | mlp | query_attn | MLP选择 + 注意力融合 |
| full_model | slot | self_attn | Slot选择 + 自注意力融合 |

---

**状态**: ✅ 已修复  
**修复时间**: 2025-12-18
