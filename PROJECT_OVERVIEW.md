# 完整项目说明文档

本文档整合了所有模块化OOD检测框架的关键信息。

## 📁 项目结构总览

```
FA/
│
├── [核心模型文件]
│   ├── model.py                      # 原始模型 (PromptLearner + TextEncoder)
│   ├── model_v2.py                   # 简化版本 (无PromptLearner)
│   ├── model_modular.py              # 模块化版本 (消融研究专用) ⭐
│   ├── main.py                       # 原始训练脚本
│   ├── main_v2.py                    # 简化版训练脚本
│   └── train_modular.py              # 模块化训练脚本 ⭐
│
├── [实验和分析工具] ⭐ 消融研究专用
│   ├── run_ablation_study.py         # 自动化消融研究
│   ├── run_quick_experiment.py       # 快速单个实验
│   └── analyze_results.py            # 结果分析工具
│
├── [数据集模块]
│   └── my_dataset/
│       ├── __init__.py
│       ├── caltech101.py
│       ├── dtd.py
│       ├── eurosat.py
│       ├── fgvc.py
│       ├── food101.py
│       ├── imagenet.py
│       ├── oxford_flowers.py
│       ├── oxford_pets.py
│       ├── stanford_cars.py
│       ├── sun397.py
│       ├── ucf101.py
│       └── utils.py                  # 数据加载工具
│
├── [CLIP编码器模块]
│   └── clip/
│       ├── __init__.py
│       ├── clip.py                   # CLIP模型接口
│       ├── model_origin.py           # 原始CLIP模型
│       ├── model.py                  # 自定义CLIP变体
│       └── simple_tokenizer.py       # 文本分词器
│
├── [工具和配置]
│   ├── utils.py                      # 通用工具函数
│   ├── ood_utils/
│   │   └── ood_tool.py              # OOD评估工具
│   ├── configs/
│   │   └── my_config.yaml            # YAML配置文件
│   └── requirements.txt              # 依赖包列表
│
└── [文档]
    ├── README.md                     # 项目概述
    ├── VERSION_GUIDE.md              # 版本说明 (v1 vs v2)
    ├── ABLATION_GUIDE.md             # 消融研究指南 ⭐
    ├── EXPERIMENTS_README.md         # 完整实验框架 ⭐
    └── PROJECT_OVERVIEW.md           # 本文件
```

## 🔄 项目演进路径

### Phase 1: 原始实现 (model.py + main.py)
```
✓ PromptLearner: 可学习的提示文本
✓ TextEncoder: 冻结的CLIP文本编码器
✓ CustomCLIP: 门控机制
✗ 不灵活，难以修改
```

### Phase 2: 简化版本 (model_v2.py + main_v2.py)
```
✗ 移除PromptLearner
✓ 固定文本模板
✓ 支持少样本和全数据集模式
✓ 简洁但仍不够灵活
```

### Phase 3: 模块化设计 (model_modular.py + train_modular.py) ⭐ 当前
```
✓ 抽象基类设计 (BaseSelector, BaseFuser)
✓ 多个实现选项可组合
✓ 配置驱动的架构切换
✓ 完整的损失函数套件
✓ 自动化消融研究工具
✓ 论文发表级代码质量
```

## 🎯 核心组件详解

### 1. 特征选择器 (Feature Selectors)

**MultiHeadMLPSelector** (model_modular.py, lines 86-173)
```python
def __init__(self, dim, num_heads=4, num_select=49):
    # 4个独立MLP评分头
    # 输出: (B, K, D) 形状的特征子集
    # 损失: 多样性约束
```

**SparseSlotAttentionSelector** (model_modular.py, lines 176-250)
```python
def __init__(self, dim, num_slots=4, num_select=49):
    # 可学习的槽查询
    # 输出: (B, K, D) 的稀疏特征
    # 损失: 正交性约束
```

### 2. 特征融合器 (Feature Fusers)

**MeanPoolFuser** (model_modular.py, lines 276-288)
```python
# 简单平均: mean(selected_features, dim=1)
# 计算量: 最小
# 效果: 基准
```

**QueryGuidedAttentionFuser** (model_modular.py, lines 291-320)
```python
# 文本向量引导的多头注意力
# 输出: (B, D) 融合特征
# 语义对齐: 文本-图像配对
```

**SelfAttentionFuser** (model_modular.py, lines 323-345)
```python
# TransformerEncoderLayer
# 补丁间的相互作用建模
# 表示力: 最强
```

### 3. 损失函数

**多样性损失** (model_modular.py, lines 250-260)
```python
def compute_diversity_loss(indices, K):
    # 最大化选择的补丁多样性
    # 避免重复选择相邻补丁
```

**正交性损失** (model_modular.py, lines 261-268)
```python
def compute_orthogonality_loss(slots):
    # 槽表示间的正交约束
    # 促进不同角色的学习
```

**语义排斥损失** (model_modular.py, lines 269-280)
```python
def compute_semantic_exclusion_loss(logits, labels):
    # 不同类别的不同表示
    # 提升类别区分度
```

### 4. 主模型 (ModularCustomCLIP)

**前向传播** (model_modular.py, lines 348-485)
```python
def forward(self, images, labels=None):
    # 1. 提取ViT特征 (B, N, D)
    # 2. 选择子集 (B, K, D)
    # 3. 融合特征 (B, D)
    # 4. 计算相似度
    # 5. 返回 {logits, aux_losses, selected_feats, ...}
```

### 5. 训练管道 (ModularTrainer)

**关键方法** (train_modular.py)
```python
trainer.setup_data()        # 数据加载 (少样本/全数据)
trainer.setup_model()       # 配置驱动的模型实例化
trainer.setup_optimizer()   # 优化器设置
trainer.compute_losses()    # 多损失聚合
trainer.train_epoch()       # 单个epoch训练
trainer.train()            # 完整训练循环
trainer.evaluate()         # 验证集评估
```

## 📊 消融研究设计

### Group A: Baseline
```yaml
selector_type: mlp
fuser_type: mean
lambda_diversity: 0.0
lambda_orthogonality: 0.0
lambda_semantic_exclusion: 0.0
```
**目标**: 建立性能基准

### Group B: Structure
```yaml
selector_type: slot
fuser_type: mean
lambda_orthogonality: 0.01
```
**目标**: 验证结构化选择机制的价值

### Group C: Fusion
```yaml
selector_type: mlp
fuser_type: query_attn
lambda_diversity: 0.01
```
**目标**: 验证注意力融合的效果

### Group D: Interaction
```yaml
selector_type: mlp
fuser_type: self_attn
lambda_diversity: 0.01
```
**目标**: 验证补丁间相互作用的作用

### Group E: Full
```yaml
selector_type: slot
fuser_type: query_attn
lambda_orthogonality: 0.01
lambda_diversity: 0.01
lambda_semantic_exclusion: 0.01
```
**目标**: 完整模块化设计的综合效果

## 🚀 快速使用指南

### 1. 运行单个实验

```bash
# 查看可用的实验组
python run_quick_experiment.py --list

# 运行Baseline
python run_quick_experiment.py --group baseline --gpu 0

# 运行Full (最强配置)
python run_quick_experiment.py --group full --gpu 0
```

### 2. 完整消融研究

```bash
# 基础运行 (3个种子，单GPU)
python run_ablation_study.py

# 多GPU加速
python run_ablation_study.py --gpus 0 1 2 --seeds 1 2 3 4 5

# 特定组合
python run_ablation_study.py --groups baseline full
```

### 3. 结果分析

```bash
# 摘要对比
python analyze_results.py

# 详细分析
python analyze_results.py --detailed --comparison

# 完整报告 + CSV
python analyze_results.py --all
```

## 📈 预期结果

### 性能递进

```
Baseline (A)      78.0%
  ↓
Structure (B)     79.2%  (+1.2%)
Fusion (C)        79.5%  (+1.5%)
Interaction (D)   79.8%  (+1.8%)
  ↓
Full (E)          80.5%  (+2.5%)
```

### 输出文件

```
ablation_results/
├── baseline/seed{1,2,3}/
│   ├── config.yaml               # 配置备份
│   ├── checkpoint_best.pth       # 最佳模型
│   └── log_train.txt             # 训练日志
├── structure/seed{1,2,3}/
├── fusion/seed{1,2,3}/
├── interaction/seed{1,2,3}/
├── full/seed{1,2,3}/
│
├── ablation_results.json         # 汇总数据 (JSON)
├── ABLATION_REPORT.md            # 自动报告 (Markdown)
└── ANALYSIS_REPORT.md            # 分析报告 (Markdown)
```

## 🔧 配置系统

### 基础模板 (configs/my_config.yaml)

```yaml
# 数据
dataset_name: 'caltech101'
use_full_data: 1                 # 1=完整，0=少样本

# 模型
visual_model: 'vit_b16'          # 或 'vit_b32'
selector_type: 'mlp'             # 或 'slot'
fuser_type: 'mean'               # 或 'query_attn', 'self_attn'
num_select: 49                   # 选择的补丁数

# 架构参数
num_heads_selector: 4
num_heads_fuser: 4
num_slots: 4

# 损失权重
lambda_diversity: 0.01
lambda_orthogonality: 0.01
lambda_semantic_exclusion: 0.01

# 训练
epochs: 30
batch_size: 128
learning_rate: 0.001
seed: 42

# 文本模板
templates: ['a photo of a', 'a bad photo of a']
```

## 📚 关键文件速查

| 文件 | 用途 | 关键类/函数 |
|------|------|-----------|
| model_modular.py | 模块化模型定义 | ModularCustomCLIP, BaseSelector, BaseFuser |
| train_modular.py | 训练管道 | ModularTrainer |
| run_ablation_study.py | 自动消融研究 | AblationExperiment |
| run_quick_experiment.py | 快速单实验 | run_quick_experiment() |
| analyze_results.py | 结果分析 | ResultsAnalyzer |
| configs/my_config.yaml | 配置模板 | YAML格式 |

## 🔍 代码架构模式

### 抽象基类模式

```python
class BaseSelector(ABC):
    @abstractmethod
    def forward(self, features):
        pass

class MultiHeadMLPSelector(BaseSelector):
    def forward(self, features):
        # 具体实现
        pass

class SparseSlotAttentionSelector(BaseSelector):
    def forward(self, features):
        # 具体实现
        pass
```

### 工厂模式

```python
def get_selector(selector_type, **kwargs):
    if selector_type == 'mlp':
        return MultiHeadMLPSelector(**kwargs)
    elif selector_type == 'slot':
        return SparseSlotAttentionSelector(**kwargs)
    else:
        raise ValueError(f"Unknown selector: {selector_type}")
```

### 配置驱动模式

```python
# 所有架构切换只需修改配置
config = load_yaml('config.yaml')
model = ModularCustomCLIP(
    selector_type=config['selector_type'],
    fuser_type=config['fuser_type'],
    # ... 其他参数
)
```

## 💡 论文提交要点

### 消融研究贡献

1. **架构灵活性**: 演示了5种不同的组件组合
2. **性能对比**: 量化了每个组件的贡献
3. **稳健性分析**: 3个种子的统计显著性
4. **计算效率**: 不同配置的训练时间对比

### 实验可重现性

- ✓ 固定随机种子
- ✓ YAML配置版本控制
- ✓ 自动检查点保存
- ✓ 完整日志记录
- ✓ 可重现的数据集加载

## ⚡ 性能优化建议

### 计算效率

```python
# 使用MLP选择器 + Mean融合 (快)
num_heads_selector = 2
num_heads_fuser = 2
num_select = 25

# vs 

# 使用Slot选择器 + Self-Attn融合 (慢但准)
num_heads_selector = 8
num_heads_fuser = 8
num_select = 49
```

### 内存优化

```python
# 减小这些参数以节省内存
batch_size = 64  # 默认128
num_select = 25  # 默认49
num_heads_fuser = 2  # 默认4
```

## 📝 发表检查清单

- [ ] 所有5个实验组完成训练
- [ ] 3个随机种子均完成
- [ ] 生成了 `ablation_results.json`
- [ ] 生成了 `ABLATION_REPORT.md` 
- [ ] 生成了 `ANALYSIS_REPORT.md`
- [ ] CSV导出用于论文表格
- [ ] 模型检查点已保存
- [ ] 训练日志已备份
- [ ] 配置文件已版本控制
- [ ] 代码注释完整

## 🔗 关键命令速查

```bash
# 列出实验
python run_quick_experiment.py --list

# 单个实验
python run_quick_experiment.py --group baseline

# 全部实验
python run_ablation_study.py

# 分析结果
python analyze_results.py --all

# 生成报告
python analyze_results.py --report --csv
```

## 📞 问题排查

| 错误 | 原因 | 解决 |
|------|------|------|
| ModuleNotFoundError | 缺少依赖 | `pip install -r requirements.txt` |
| CUDA Out of Memory | 显存不足 | 减小 batch_size 或 num_select |
| Config not found | 配置文件路径错误 | 检查 configs/my_config.yaml 路径 |

---

**文档版本**: 1.0  
**最后更新**: 2024  
**框架状态**: 生产就绪 (Production Ready)  
**发表目标**: ICML 2024/2025
