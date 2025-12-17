# 模块化OOD检测 - 消融研究使用指南

## 概述

该项目实现了一个灵活的模块化架构，支持以下可切换的组件：

- **特征选择器 (Feature Selectors)**
  - MLP多头选择器 (MultiHeadMLPSelector)：4个独立评分头，多样性损失
  - 稀疏槽注意力选择器 (SparseSlotAttentionSelector)：可学习的槽查询，正交性正则化

- **特征融合器 (Feature Fusers)**
  - 均值融合 (MeanPoolFuser)：简单的特征均值
  - 查询引导注意力融合 (QueryGuidedAttentionFuser)：文本引导的多头注意力
  - 自注意力融合 (SelfAttentionFuser)：TransformerEncoderLayer融合

- **损失函数 (Loss Functions)**
  - 多样性损失 (Diversity Loss)：鼓励选择多样化的补丁
  - 语义排斥损失 (Semantic Exclusion Loss)：确保不同类别的不同表示
  - Mixup不变性损失 (Mixup Invariance Loss)：增强鲁棒性

## 实验分组 (Ablation Study Groups)

### Group 1: Baseline (基准)
```
Selector: MLP多头选择器
Fuser: 均值融合
Losses: 无辅助损失
配置键: baseline
```

### Group 2: Structure (结构)
```
Selector: 槽注意力选择器 (引入结构)
Fuser: 均值融合
Losses: 正交性正则化
配置键: structure
```

### Group 3: Fusion (融合)
```
Selector: MLP多头选择器
Fuser: 查询引导注意力融合 (增强融合)
Losses: 多样性损失
配置键: fusion
```

### Group 4: Interaction (交互)
```
Selector: MLP多头选择器
Fuser: 自注意力融合 (补丁间交互)
Losses: 多样性损失
配置键: interaction
```

### Group 5: Full (完整)
```
Selector: 槽注意力选择器
Fuser: 查询引导注意力融合
Losses: 所有损失函数 (正交性 + 多样性 + 语义排斥)
配置键: full
```

## 使用方法

### 1. 快速运行单个实验

```bash
# 运行Baseline组
python run_quick_experiment.py --group baseline --gpu 0

# 运行Full组
python run_quick_experiment.py --group full --gpu 0

# 列出所有可用组
python run_quick_experiment.py --list
```

### 2. 运行完整消融研究

```bash
# 默认配置：3个随机种子，所有GPU
python run_ablation_study.py

# 自定义设置
python run_ablation_study.py \
    --base_config configs/my_config.yaml \
    --output_dir ./ablation_results \
    --seeds 1 2 3 \
    --gpus 0 1 2 \
    --groups baseline structure fusion interaction full
```

### 3. 运行特定组合

```bash
# 只运行Structure和Full两个组
python run_ablation_study.py --groups structure full

# 使用2个GPU和5个种子
python run_ablation_study.py --gpus 0 1 --seeds 1 2 3 4 5
```

## 配置系统

### 基础配置文件 (configs/my_config.yaml)

```yaml
# 数据集
dataset_name: 'caltech101'
use_full_data: 1  # 1=完整数据集，0=少样本

# 模型架构
selector_type: 'mlp'  # 或 'slot'
fuser_type: 'mean'     # 或 'query_attn', 'self_attn'
num_select: 49         # 选择的补丁数量

# 选择器参数
num_heads_selector: 4  # MLP选择器的头数
num_slots: 4           # 槽选择器的槽数

# 融合器参数
num_heads_fuser: 4     # 注意力融合的头数

# 损失权重
lambda_diversity: 0.01
lambda_orthogonality: 0.01
lambda_semantic_exclusion: 0.01

# 训练参数
epochs: 30
batch_size: 128
learning_rate: 0.001
seed: 42

# 模板
templates: ['a photo of a', 'a bad photo of a']
```

## 输出结构

```
ablation_results/
├── baseline/
│   ├── seed1/
│   │   ├── config.yaml           # 该实验的配置
│   │   ├── checkpoint_best.pth   # 最佳模型检查点
│   │   └── log_train.txt         # 训练日志
│   ├── seed2/
│   └── seed3/
├── structure/
├── fusion/
├── interaction/
├── full/
├── ablation_results.json         # 汇总结果 (JSON格式)
└── ABLATION_REPORT.md            # 消融研究报告 (Markdown格式)
```

## 结果分析

### 查看汇总结果

```bash
cat ablation_results/ablation_results.json
```

### 查看完整报告

```bash
cat ablation_results/ABLATION_REPORT.md
```

### 提取特定组的最佳模型

```bash
# Full组中seed1的最佳模型
ls ablation_results/full/seed1/checkpoint_best.pth
```

## 关键指标

### 每个组计算的指标

- **Test Accuracy**: 测试集准确率 (%)
- **Mean Accuracy**: 跨种子的平均准确率 (%)
- **Std Accuracy**: 准确率的标准差 (%)
- **Training Loss**: 最终训练损失
- **Loss Breakdown**: 各部分损失的贡献

### 消融研究目标

1. **Baseline vs Structure**: 验证槽注意力选择器的有效性
2. **Baseline vs Fusion**: 验证查询引导注意力融合的效果
3. **Baseline vs Interaction**: 验证自注意力融合的效果
4. **Full vs Others**: 验证完整模块化设计的优势

## 高级用法

### 自定义实验

编辑 `run_ablation_study.py` 中的 `GROUPS` 字典，添加新的实验配置：

```python
GROUPS = {
    'custom': {
        'name': '自定义实验描述',
        'configs': {
            'selector_type': 'mlp',
            'fuser_type': 'self_attn',
            'num_heads_selector': 8,
            'lambda_diversity': 0.02,
            # 其他配置...
        }
    }
}
```

然后运行：

```bash
python run_ablation_study.py --groups custom
```

### 并行运行多GPU

```bash
# 使用4个GPU运行所有实验
python run_ablation_study.py --gpus 0 1 2 3 --seeds 1 2 3 4 5
```

系统会自动分配GPU，轮循使用。

### 继续未完成的实验

```bash
# 只运行那些没有完成的种子
python run_ablation_study.py --groups baseline
```

## 故障排除

### 问题：内存不足 (OOM)

**解决方案**：
1. 减小 `batch_size` (在config.yaml中)
2. 减少 `num_select` (选择更少的补丁)
3. 减小 `num_heads_fuser` 或 `num_heads_selector`

### 问题：实验超时

**解决方案**：
1. 检查GPU是否正常工作
2. 减少 `epochs`
3. 增加超时限制 (在 `run_ablation_study.py` 中)

### 问题：导入错误

**解决方案**：
```bash
# 确保在项目根目录运行
cd f:\shixi\ailab\ood\FA

# 检查依赖
pip install -r requirements.txt

# 验证模块
python -c "import torch; import clip; print('OK')"
```

## 比较分析

### 生成对比表

结果会自动汇总到 `ABLATION_REPORT.md`，包含：

| 组名 | 架构 | 平均准确率 (%) | 标准差 (%) | 状态 |
|------|------|--------------|----------|------|
| baseline | MLP + Mean | 78.45 | 1.23 | 3/3 completed |
| structure | Slot + Mean | 79.12 | 0.98 | 3/3 completed |
| ... | ... | ... | ... | ... |

## 最佳实践

1. **种子设置**: 至少使用3个不同的随机种子以获得统计显著性
2. **GPU选择**: 对于不同的实验组，使用不同的GPU以加快速度
3. **监控**: 使用 `nvidia-smi` 监控GPU使用情况
4. **检查点**: 所有最佳模型自动保存，可用于后续推理
5. **日志分析**: 检查 `log_train.txt` 了解训练动态

## 参考文献

架构设计参考了以下论文的技术：
- Vision Transformer (ViT)
- CLIP (Contrastive Language-Image Pre-training)
- Slot Attention 机制
- 多头自注意力融合

详见 `model_modular.py` 和 `train_modular.py` 的代码注释。
