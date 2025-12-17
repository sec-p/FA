"""
ICML 论文新特性使用示例
展示如何集成 LLM Negatives, Causal Mixup, Visualization, 和 Orthogonal Init
"""

import torch
import torch.nn as nn
import json
# import yaml  # 如需要请取消注释
# from model_modular import build_modular_model  # 在实际使用中导入
# from train_modular import ModularTrainer
# import clip

# ============================================================================
# 示例 1: 加载配置并启用所有新特性
# ============================================================================

def load_config_with_new_features():
    """加载配置并启用所有新特性"""
    cfg = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # 基础模型配置
        'selector_type': 'slot',
        'fuser_type': 'query_attn',
        'num_select': 49,
        'num_slots': 4,
        'num_heads_selector': 4,
        'num_heads_fuser': 4,
        
        # ✨ 新特性 1: LLM Negatives
        'use_llm_negatives': True,
        'lambda_llm_negatives': 0.05,
        'class_negatives_path': 'configs/class_negatives.json',
        
        # ✨ 新特性 2: Causal Mixup
        'use_mixup_invariance': True,
        'lambda_mixup': 0.1,
        
        # ✨ 新特性 3: Visualization (自动在 training 中调用)
        'save_vis_interval': 5,
        
        # ✨ 新特性 4: Orthogonal Slot Init (在 SparseSlotAttentionSelector 中自动使用)
        
        # 其他损失
        'lambda_diversity': 0.01,
        'lambda_orthogonality': 0.01,
        'lambda_semantic_exclusion': 0.01,
        'use_semantic_exclusion': False,
        
        # 训练参数
        'templates': ['a photo of a'],
        'lr': 0.002,
        'fine_tune_train_epoch': 50,
        'fine_tune_batch_size': 160,
    }
    return cfg


# ============================================================================
# 示例 2: 创建支持新特性的模型
# ============================================================================

def create_model_with_new_features(classnames, cfg):
    """创建启用了所有新特性的模型"""
    
    # 加载 CLIP
    # from model_modular import build_modular_model  # 导入
    # import clip  # 导入
    # clip_model, _ = clip.load('ViT-B/32', device=cfg['device'])
    
    # 构建模型（自动启用所有新特性）
    # model = build_modular_model(cfg, classnames, clip_model)
    
    print("✓ 模型已启用以下新特性:")
    print(f"  - LLM Negatives: {cfg.get('use_llm_negatives', False)}")
    print(f"  - Causal Mixup: {cfg.get('use_mixup_invariance', False)}")
    print(f"  - Visualization: 每 {cfg.get('save_vis_interval', 10)} epoch 保存")
    print(f"  - Slot Orthogonal Init: ✓ (自动)")
    
    return model


# ============================================================================
# 示例 3: 前向传递中使用负面 tokens
# ============================================================================

def forward_pass_with_negatives(model, images, labels, neg_tokens, cfg):
    """演示如何在前向传递中使用负面文本 tokens"""
    
    # 标准前向传递（支持负面 tokens）
    output = model(images, labels, negative_text_tokens=neg_tokens)
    
    print("\n✓ 前向传递完成:")
    print(f"  - 输出 logits shape: {output['logits'].shape}")
    print(f"  - 辅助损失项:")
    for loss_name, loss_val in output['aux_losses'].items():
        print(f"    • {loss_name}: {loss_val.item():.6f}")
    
    return output


# ============================================================================
# 示例 4: 计算所有损失（包括新的）
# ============================================================================

def compute_all_losses(output, labels, cfg):
    """计算包含所有新特性的综合损失"""
    
    losses = {}
    criterion = nn.CrossEntropyLoss()
    
    # 分类损失
    ce_loss = criterion(output['logits'], labels)
    losses['ce'] = ce_loss
    
    # ✨ 新增: LLM Negatives 损失
    if 'llm_negatives' in output['aux_losses']:
        losses['llm_negatives'] = output['aux_losses']['llm_negatives']
        print(f"  LLM Negatives Loss: {losses['llm_negatives'].item():.6f}")
    
    # ✨ 新增: Causal Mixup 损失
    if 'mixup_invariance' in output['aux_losses']:
        losses['mixup_invariance'] = output['aux_losses']['mixup_invariance']
        print(f"  Mixup Invariance Loss: {losses['mixup_invariance'].item():.6f}")
    
    # 其他损失
    if 'orthogonality' in output['aux_losses']:
        lambda_ortho = cfg.get('lambda_orthogonality', 0.01)
        losses['orthogonality'] = lambda_ortho * output['aux_losses']['orthogonality']
        print(f"  Orthogonality Loss: {losses['orthogonality'].item():.6f}")
    
    total_loss = sum(losses.values())
    print(f"\nTotal Loss: {total_loss.item():.6f}")
    
    return losses, total_loss


# ============================================================================
# 示例 5: 准备 LLM 负面词数据
# ============================================================================

def prepare_class_negatives(class_names):
    """为您的类别列表生成负面词映射"""
    
    # 这是示例，实际应该由 LLM 或人工精选
    class_negatives = {
        "dog": ["cat", "wolf", "animal", "pet", "mammal"],
        "cat": ["dog", "bird", "rodent", "feline", "beast"],
        "bird": ["airplane", "insect", "butterfly", "flying_object"],
        # ... 为每个类添加相关概念
    }
    
    # 保存为 JSON
    import json
    with open('configs/class_negatives.json', 'w') as f:
        json.dump(class_negatives, f, indent=2)
    
    print(f"✓ 已保存 {len(class_negatives)} 个类别的负面词映射")
    return class_negatives


# ============================================================================
# 示例 6: 训练循环中的完整集成
# ============================================================================

def training_loop_with_all_features():
    """展示完整的训练循环，集成所有新特性"""
    
    print("=" * 70)
    print("ICML 论文新特性集成示例 - 完整训练循环")
    print("=" * 70)
    
    # Step 1: 加载配置
    cfg = load_config_with_new_features()
    print("\n[Step 1] 配置已加载")
    print(f"  - 使用 LLM 负面词: {cfg['use_llm_negatives']}")
    print(f"  - 使用 Causal Mixup: {cfg['use_mixup_invariance']}")
    
    # Step 2: 准备数据
    classnames = ["dog", "cat", "bird"]
    neg_tokens = torch.randn(4, 2, 512)  # (B, num_neg, D)
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 0])
    
    # Step 3: 创建模型 (在实际使用中取消注释)
    # print("\n[Step 2] 创建模型...")
    # from model_modular import build_modular_model
    # import clip
    # clip_model, _ = clip.load('ViT-B/32', device=cfg['device'])
    # model = build_modular_model(cfg, classnames, clip_model)
    
    print("\n[Step 3] 执行前向传递...")
    # with torch.no_grad():
    #     output = forward_pass_with_negatives(...)
    
    print("\n✓ 示例展示完成！")


# ============================================================================
# 示例 7: 消融研究 - 逐个禁用特性
# ============================================================================

def ablation_study_example(cfg):
    """展示如何进行消融研究"""
    
    print("\n" + "=" * 70)
    print("消融研究示例 - 测试各特性的贡献")
    print("=" * 70)
    
    # 基础配置
    base_cfg = cfg.copy()
    
    # 配置组合
    configurations = [
        ("Baseline (仅分类损失)", {
            'use_llm_negatives': False,
            'use_mixup_invariance': False,
            'use_semantic_exclusion': False,
        }),
        ("+ LLM Negatives", {
            'use_llm_negatives': True,
            'use_mixup_invariance': False,
            'use_semantic_exclusion': False,
        }),
        ("+ Causal Mixup", {
            'use_llm_negatives': False,
            'use_mixup_invariance': True,
            'use_semantic_exclusion': False,
        }),
        ("+ LLM + Mixup", {
            'use_llm_negatives': True,
            'use_mixup_invariance': True,
            'use_semantic_exclusion': False,
        }),
        ("完整方法 (+ Orthogonal Init)", {
            'use_llm_negatives': True,
            'use_mixup_invariance': True,
            'use_semantic_exclusion': False,
            # Orthogonal init 已自动使用
        }),
    ]
    
    print("\n建议的消融研究配置:\n")
    for desc, config_override in configurations:
        print(f"  ▶ {desc}")
        for key, val in config_override.items():
            if val is not None:
                print(f"    - {key}: {val}")
        print()


# ============================================================================
# 示例 8: 可视化生成（在 trainer 中自动调用）
# ============================================================================

def visualization_example():
    """展示可视化的工作原理"""
    
    print("\n" + "=" * 70)
    print("注意力可视化示例")
    print("=" * 70)
    
    print("""
在训练中，每 N 个 epoch 会自动生成可视化：

trainer.save_attention_maps(epoch=5)

输出目录结构：
  my_caches/{dataset}/{model}/attention_maps/epoch_5/
  ├── batch00_sample0_class_dog.png
  ├── batch00_sample1_class_cat.png
  └── ...

每个 PNG 文件显示：
  [原始图像] | [注意力掩码叠加]
  
  左侧：原始输入图像
  右侧：模型学到的"最小特征集"可视化（Top-K 掩码）
  
这直观展示了：
  ✓ 模型关注的核心特征区域
  ✓ 最小特征集的大小和分布
  ✓ 是否学到了有意义的模式
""")


# ============================================================================
# 主函数：运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("\\n" * 2)
    print("🎯 ICML 论文新特性使用示例")
    print("=" * 70)
    
    # 运行示例
    cfg = load_config_with_new_features()
    
    try:
        # 示例 1-6: 完整训练循环
        training_loop_with_all_features()
        
        # 示例 7: 消融研究
        ablation_study_example(cfg)
        
        # 示例 8: 可视化说明
        visualization_example()
        
        print("\n" + "=" * 70)
        print("✅ 所有示例已完成！")
        print("=" * 70)
        print("""
下一步：
  1. 根据需求修改 configs/class_negatives.json
  2. 设置合适的 lambda 参数
  3. 运行: python train_modular.py --config configs/my_config.yaml
  4. 在 my_caches/ 中查看训练结果和可视化
  5. 在论文中使用生成的图表和数据
""")
        
    except Exception as e:
        print(f"\n❌ 示例执行出错: {e}")
        import traceback
        traceback.print_exc()
