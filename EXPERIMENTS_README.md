# 模块化OOD检测框架 - 消融研究

## 项目概述

这是一个为ICML会议提交而优化的**模块化架构**，用于深度OOD（Out-of-Distribution）检测。框架支持灵活的组件组合，以进行全面的消融研究。

## 核心特性

### 1. 灵活的特征选择器 (Feature Selectors)

选择视觉特征中最相关的补丁（patch）：

| 选择器 | 机制 | 优势 |
|--------|------|------|
| **MultiHeadMLPSelector** | 4个独立MLP头评分补丁，多样性损失 | 快速、轻量、易于集成 |
| **SparseSlotAttentionSelector** | 可学习的槽查询交叉注意，稀疏掩模 | 结构化学习、正交正则 |

### 2. 多种特征融合器 (Feature Fusers)

将选中的特征融合为紧凑表示：

| 融合器 | 机制 | 应用场景 |
|--------|------|---------|
| **MeanPoolFuser** | 简单平均池化 | 基准、计算高效 |
| **QueryGuidedAttentionFuser** | 文本引导的多头注意力 | 语义对齐融合 |
| **SelfAttentionFuser** | Transformer编码层 | 补丁间相互作用 |

### 3. 完整的损失函数套件 (Loss Functions)

```
总损失 = CE损失 + λ₁·多样性损失 + λ₂·正交性损失 + λ₃·语义排斥损失
```

- **多样性损失**: 鼓励选择不同的补丁
- **正交性损失**: 槽表示间的正交约束
- **语义排斥损失**: 不同类别的不同表示

### 4. 配置驱动实验

完全通过YAML配置切换架构：

```yaml
selector_type: 'mlp'          # 选择器类型
fuser_type: 'query_attn'      # 融合器类型
num_select: 49                # 选择的补丁数
lambda_diversity: 0.01        # 多样性权重
lambda_semantic_exclusion: 0.01  # 语义排斥权重
```

## 文件结构

```
FA/
├── model_modular.py              # 核心模块化模型
├── train_modular.py              # 训练脚本
├── run_ablation_study.py         # 消融研究自动化
├── run_quick_experiment.py       # 快速单个实验
├── analyze_results.py            # 结果分析工具
├── configs/
│   └── my_config.yaml            # 基础配置模板
├── ABLATION_GUIDE.md             # 详细使用指南
└── ablation_results/
    ├── baseline/
    │   ├── seed1/, seed2/, seed3/
    ├── structure/
    ├── fusion/
    ├── interaction/
    ├── full/
    ├── ablation_results.json     # 结果汇总
    └── ABLATION_REPORT.md        # 自动生成的报告
```

## 快速开始

### 1. 环境设置

```bash
pip install -r requirements.txt
```

### 2. 运行单个实验

```bash
# 查看所有可用组
python run_quick_experiment.py --list

# 运行Baseline
python run_quick_experiment.py --group baseline --gpu 0

# 运行Full（完整模型）
python run_quick_experiment.py --group full --gpu 0
```

### 3. 运行完整消融研究

```bash
# 默认配置（3个种子，单GPU）
python run_ablation_study.py

# 多GPU加速（使用3个GPU）
python run_ablation_study.py --gpus 0 1 2 --seeds 1 2 3 4 5

# 只运行特定组合
python run_ablation_study.py --groups baseline full
```

### 4. 分析结果

```bash
# 查看摘要
python analyze_results.py --summary

# 详细对比
python analyze_results.py --detailed --comparison

# 生成完整报告
python analyze_results.py --all

# 导出CSV
python analyze_results.py --csv
```

## 消融研究设计

### 实验分组矩阵

| Group | 选择器 | 融合器 | 损失函数 | 目标 |
|-------|--------|--------|---------|------|
| **Baseline** | MLP | Mean | CE | 建立基准 |
| **Structure** | Slot | Mean | CE + Ortho | 验证结构化选择 |
| **Fusion** | MLP | Query Attn | CE + Diversity | 验证融合机制 |
| **Interaction** | MLP | Self Attn | CE + Diversity | 验证补丁交互 |
| **Full** | Slot | Query Attn | CE + All | 完整架构效果 |

### 预期改进

```
Baseline: 78.0%
├─ +Structure: 79.2% (+1.2%)    [槽选择的价值]
├─ +Fusion: 79.5% (+1.5%)       [注意力融合的价值]
├─ +Interaction: 79.8% (+1.8%)  [自注意力的价值]
└─ Full: 80.5% (+2.5%)          [完整模型的最大效果]
```

## 核心代码示例

### 初始化模块化模型

```python
from model_modular import ModularCustomCLIP

model = ModularCustomCLIP(
    visual_model='vit_b16',
    selector_type='slot',
    fuser_type='query_attn',
    num_slots=4,
    num_select=49,
    num_heads_fuser=4,
    templates=['a photo of a'],
    lambda_diversity=0.01,
    lambda_orthogonality=0.01,
)
```

### 训练流程

```python
from train_modular import ModularTrainer

trainer = ModularTrainer(config_path='configs/my_config.yaml')
trainer.setup_data()
trainer.setup_model()
trainer.setup_optimizer()

# 训练
trainer.train(
    num_epochs=30,
    validate_every=5,
    save_best=True
)
```

### 分析结果

```python
from analyze_results import ResultsAnalyzer

analyzer = ResultsAnalyzer('./ablation_results')
analyzer.print_summary()
analyzer.print_comparison()
analyzer.save_markdown_report()
```

## 性能指标

### 评估指标

- **Test Accuracy**: 测试集准确率
- **Mean Accuracy**: 跨种子平均准确率
- **Std Dev**: 准确率标准差（稳定性）
- **Training Loss**: 最终训练损失

### 统计显著性

所有实验使用3个不同的随机种子运行，确保结果的稳健性：

```
Group Performance = Mean ± Std (N=3)
```

## 发表准备

### 生成论文图表

```bash
# 生成所有分析
python analyze_results.py --all

# 输出文件：
# - ablation_results.csv (表格数据)
# - ANALYSIS_REPORT.md (详细报告)
# - ablation_results.json (原始数据)
```

### 关键数据点

1. **架构对比**: 所有选择器/融合器组合的性能
2. **损失函数贡献**: 每个损失项的单独影响
3. **计算效率**: 不同配置的速度对比
4. **鲁棒性**: 跨种子的性能稳定性

## 高级配置

### 自定义实验组

编辑 `run_ablation_study.py` 的 `GROUPS` 字典：

```python
GROUPS = {
    'custom_exp': {
        'name': 'Custom Experiment',
        'configs': {
            'selector_type': 'mlp',
            'fuser_type': 'self_attn',
            'num_heads_selector': 8,
            'lambda_diversity': 0.02,
        }
    }
}
```

### 调整超参数

在 `configs/my_config.yaml` 中修改：

```yaml
# 模型架构
num_heads_selector: 4
num_heads_fuser: 4
num_select: 49

# 损失权重
lambda_diversity: 0.01
lambda_orthogonality: 0.01
lambda_semantic_exclusion: 0.01

# 训练参数
batch_size: 128
learning_rate: 0.001
epochs: 30
```

## 故障排除

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| CUDA内存不足 | 减小 `batch_size` 或 `num_select` |
| 实验超时 | 增加超时限制或使用更小的数据集 |
| 导入错误 | 检查 `requirements.txt` 依赖 |
| GPU不识别 | 验证 CUDA 安装，检查 `nvidia-smi` |

### 调试模式

```bash
# 运行单个小批次测试
python train_modular.py --config configs/my_config.yaml --is_train 1
```

## 参考文献

### 相关工作

- CLIP: Contrastive Language-Image Pre-training (Radford et al., 2021)
- Vision Transformer: An Image is Worth 16x16 Words (Dosovitskiy et al., 2021)
- Slot Attention: Object-Centric Representation (Locatello et al., 2020)
- OOD Detection: A Baseline for Detecting Out-of-Distribution Examples (Hendrycks et al., 2019)

### 代码参考

所有实现参考了：
- PyTorch官方文档
- CLIP官方实现
- timm（PyTorch Image Models）

## 联系与反馈

有任何问题或建议，请提交问题或拉取请求。

---

**Last Updated**: 2024
**Framework Version**: 1.0
**Status**: Ready for ICML Submission
