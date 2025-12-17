#!/usr/bin/env python3
"""
ICML 论文新特性 - 最终验证检查清单
运行此脚本以验证所有功能是否正确配置
"""

import os
import json
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"  ✅ {description}: {filepath}")
        return True
    else:
        print(f"  ❌ {description}: {filepath} (文件未找到)")
        return False

def check_json_valid(filepath):
    """检查 JSON 文件是否有效"""
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        return True
    except:
        return False

def check_config_yaml(filepath):
    """检查 YAML 配置文件"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # 检查关键配置项是否存在
            checks = [
                'use_llm_negatives',
                'use_mixup_invariance',
                'lambda_llm_negatives',
                'lambda_mixup',
                'class_negatives_path',
                'save_vis_interval',
                'selector_type',
                'fuser_type'
            ]
            for check in checks:
                if check not in content:
                    return False, f"缺少配置项: {check}"
        return True, "所有配置项完整"
    except Exception as e:
        return False, str(e)

def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "ICML 论文新特性 - 最终验证检查".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_passed = True
    
    # ==================== 检查新增文件 ====================
    print("📂 [1/5] 检查新增文件...")
    print("─" * 80)
    
    new_files = [
        ('configs/class_negatives.json', 'LLM 负面词映射文件'),
        ('IMPLEMENTATION_SUMMARY.md', '实现总结文档'),
        ('QUICK_START_ICML.md', '快速启动指南'),
        ('MODIFICATION_CHECKLIST.md', '修改清单'),
        ('FINAL_DELIVERY_REPORT.md', '最终交付报告'),
        ('examples_icml_features.py', '代码示例'),
    ]
    
    for filename, desc in new_files:
        filepath = os.path.join(base_dir, filename)
        if not check_file_exists(filepath, desc):
            all_passed = False
    
    print()
    
    # ==================== 检查修改的文件 ====================
    print("📝 [2/5] 检查主要代码文件...")
    print("─" * 80)
    
    core_files = [
        ('model_modular.py', '模型核心文件'),
        ('train_modular.py', '训练脚本'),
        ('my_dataset/utils.py', '数据加载工具'),
        ('configs/my_config.yaml', '配置文件'),
    ]
    
    for filename, desc in core_files:
        filepath = os.path.join(base_dir, filename)
        if not check_file_exists(filepath, desc):
            all_passed = False
    
    print()
    
    # ==================== 检查 JSON 格式 ====================
    print("🔍 [3/5] 验证文件格式...")
    print("─" * 80)
    
    json_file = os.path.join(base_dir, 'configs/class_negatives.json')
    if os.path.exists(json_file):
        if check_json_valid(json_file):
            print("  ✅ class_negatives.json: JSON 格式正确")
        else:
            print("  ❌ class_negatives.json: JSON 格式错误")
            all_passed = False
    
    print()
    
    # ==================== 检查配置项 ====================
    print("⚙️  [4/5] 检查配置文件...")
    print("─" * 80)
    
    yaml_file = os.path.join(base_dir, 'configs/my_config.yaml')
    if os.path.exists(yaml_file):
        valid, msg = check_config_yaml(yaml_file)
        if valid:
            print(f"  ✅ my_config.yaml: {msg}")
        else:
            print(f"  ⚠️  my_config.yaml: {msg}")
            all_passed = False
    
    print()
    
    # ==================== 检查关键功能 ====================
    print("🎯 [5/5] 验证关键功能实现...")
    print("─" * 80)
    
    # 检查 model_modular.py 中的关键方法
    model_file = os.path.join(base_dir, 'model_modular.py')
    if os.path.exists(model_file):
        with open(model_file, 'r') as f:
            content = f.read()
            
        checks = [
            ('_compute_llm_negatives_loss', 'LLM 负面词损失计算'),
            ('compute_mixup_invariance_loss', 'Causal Mixup 损失计算'),
            ('negative_text_tokens', 'forward 中负面 tokens 支持'),
            ('orthogonal_', 'Slot 正交初始化'),
        ]
        
        for method, desc in checks:
            if method in content:
                print(f"  ✅ {desc}: 已实现")
            else:
                print(f"  ❌ {desc}: 未找到")
                all_passed = False
    
    # 检查 train_modular.py 中的关键方法
    train_file = os.path.join(base_dir, 'train_modular.py')
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            content = f.read()
        
        if 'save_attention_maps' in content:
            print(f"  ✅ 可视化函数: 已实现")
        else:
            print(f"  ❌ 可视化函数: 未找到")
            all_passed = False
    
    print()
    
    # ==================== 最终报告 ====================
    print("=" * 80)
    if all_passed:
        print("🎉 [完成] 所有检查通过！可以开始训练了。")
        print()
        print("建议的下一步:")
        print("  1. 根据需要修改 configs/class_negatives.json")
        print("  2. 调整 configs/my_config.yaml 中的超参数")
        print("  3. 运行: python train_modular.py --config configs/my_config.yaml")
        print()
        return 0
    else:
        print("⚠️  [警告] 某些检查未通过，请解决上述问题。")
        print()
        print("常见问题:")
        print("  - 文件缺失: 检查是否在正确的目录下运行")
        print("  - JSON 格式错误: 使用在线 JSON 验证工具检查")
        print("  - 配置项缺失: 参考 QUICK_START_ICML.md 添加缺失项")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
