# 📖 快速导航索引

## 🎯 按任务选择文档

### "我想快速了解项目"
👉 **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - 10分钟快速入门

### "我想运行一个实验"
👉 **[EXPERIMENTS_README.md](EXPERIMENTS_README.md)** - 实验框架完整说明

### "我想进行消融研究"
👉 **[ABLATION_GUIDE.md](ABLATION_GUIDE.md)** - 详细的消融研究指南

### "我想分析结果"
👉 详见 **[EXPERIMENTS_README.md](EXPERIMENTS_README.md)** 的"分析结果"部分

### "我想了解项目版本差异"
👉 **[VERSION_GUIDE.md](VERSION_GUIDE.md)** - v1 vs v2 的对比

### "我想了解原始项目"
👉 **[README.md](README.md)** - 项目基础说明

---

## 🚀 快速命令速查

### 查看可用实验组
```bash
python run_quick_experiment.py --list
```

### 运行单个实验
```bash
# 运行Baseline
python run_quick_experiment.py --group baseline --gpu 0

# 运行Full（最强配置）
python run_quick_experiment.py --group full --gpu 0
```

### 完整消融研究
```bash
# 基础配置
python run_ablation_study.py

# 多GPU加速
python run_ablation_study.py --gpus 0 1 2 --seeds 1 2 3 4 5

# 特定组合
python run_ablation_study.py --groups baseline full
```

### 分析结果
```bash
# 摘要
python analyze_results.py

# 详细分析
python analyze_results.py --all

# 生成报告和CSV
python analyze_results.py --report --csv
```

---

## 📁 文件结构导航

### 核心实验工具
| 文件 | 作用 |
|------|------|
| `run_ablation_study.py` | 自动化消融研究 (5个实验组×3个种子) |
| `run_quick_experiment.py` | 快速单个实验 |
| `analyze_results.py` | 结果分析和可视化 |

### 模块化模型
| 文件 | 作用 |
|------|------|
| `model_modular.py` | 核心模块化模型定义 |
| `train_modular.py` | 训练管道 |

### 简化版本 (参考)
| 文件 | 作用 |
|------|------|
| `model_v2.py` | 无PromptLearner的简化模型 |
| `main_v2.py` | 简化版训练脚本 |

### 原始版本 (参考)
| 文件 | 作用 |
|------|------|
| `model.py` | 原始模型 |
| `main.py` | 原始训练脚本 |

### 配置和数据
| 文件/文件夹 | 作用 |
|----------|------|
| `configs/my_config.yaml` | 模型配置文件 |
| `my_dataset/` | 11个数据集适配器 |
| `clip/` | CLIP编码器 |

---

## 📊 消融研究五大分组

### Group A: Baseline 🟦
```
Selector: MLP多头选择器
Fuser: 均值融合
Losses: 无辅助损失
目标: 建立性能基准
```

### Group B: Structure 🟩
```
Selector: 槽注意力选择器
Fuser: 均值融合
Losses: 正交性正则化
目标: 验证结构化选择机制
```

### Group C: Fusion 🟨
```
Selector: MLP多头选择器
Fuser: 查询引导注意力融合
Losses: 多样性损失
目标: 验证注意力融合效果
```

### Group D: Interaction 🟧
```
Selector: MLP多头选择器
Fuser: 自注意力融合
Losses: 多样性损失
目标: 验证补丁交互作用
```

### Group E: Full 🟪
```
Selector: 槽注意力选择器
Fuser: 查询引导注意力融合
Losses: 所有损失函数
目标: 完整模块化设计效果
```

---

## 🔍 关键代码位置速查

### 特征选择器
| 类 | 文件 | 行数 |
|----|------|------|
| `MultiHeadMLPSelector` | model_modular.py | 86-173 |
| `SparseSlotAttentionSelector` | model_modular.py | 176-250 |

### 特征融合器
| 类 | 文件 | 行数 |
|----|------|------|
| `MeanPoolFuser` | model_modular.py | 276-288 |
| `QueryGuidedAttentionFuser` | model_modular.py | 291-320 |
| `SelfAttentionFuser` | model_modular.py | 323-345 |

### 主模型
| 类 | 文件 | 行数 |
|----|------|------|
| `ModularCustomCLIP` | model_modular.py | 348-485 |

### 损失函数
| 函数 | 文件 | 行数 |
|----|------|------|
| `compute_diversity_loss` | model_modular.py | 250-260 |
| `compute_orthogonality_loss` | model_modular.py | 261-268 |
| `compute_semantic_exclusion_loss` | model_modular.py | 269-280 |

### 训练管道
| 类 | 文件 | 行数 |
|----|------|------|
| `ModularTrainer` | train_modular.py | 全文 |

---

## 📈 实验输出结构

```
ablation_results/
├── baseline/
│   ├── seed1/
│   │   ├── config.yaml              ← 该实验的配置
│   │   ├── checkpoint_best.pth      ← 最佳模型权重
│   │   └── log_train.txt            ← 训练日志
│   ├── seed2/
│   └── seed3/
├── structure/
├── fusion/
├── interaction/
├── full/
│
├── ablation_results.json            ← 汇总数据（JSON格式）
├── ABLATION_REPORT.md               ← 自动生成报告（Markdown）
└── ANALYSIS_REPORT.md               ← 分析报告（Markdown）
```

---

## 🎓 学习路径

### 初级: 快速开始 (15分钟)
1. 阅读 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
2. 运行 `python run_quick_experiment.py --list`
3. 运行单个实验: `python run_quick_experiment.py --group baseline`

### 中级: 完整消融研究 (2小时)
1. 详细阅读 [ABLATION_GUIDE.md](ABLATION_GUIDE.md)
2. 运行 `python run_ablation_study.py --gpus 0 1`
3. 分析结果: `python analyze_results.py --all`

### 高级: 自定义实验
1. 编辑 `run_ablation_study.py` 的 `GROUPS` 字典
2. 修改 `configs/my_config.yaml`
3. 运行自定义实验: `python run_ablation_study.py --groups custom_exp`

---

## 🔧 常见问题速解

### Q: 如何快速测试设置?
```bash
python run_quick_experiment.py --group baseline --gpu 0
```

### Q: 如何进行完整消融研究?
```bash
python run_ablation_study.py --gpus 0 1 2 --seeds 1 2 3 4 5
```

### Q: 如何查看结果?
```bash
python analyze_results.py --summary        # 摘要
python analyze_results.py --detailed       # 详细
python analyze_results.py --comparison     # 对比基准
python analyze_results.py --report         # 生成报告
```

### Q: 内存不足怎么办?
编辑 `configs/my_config.yaml`:
```yaml
batch_size: 64           # 减小批大小
num_select: 25           # 减小补丁数
num_heads_fuser: 2       # 减小注意力头数
```

### Q: 如何使用多GPU?
```bash
python run_ablation_study.py --gpus 0 1 2 3
```

---

## 📚 文档导航树

```
📄 本文件 (导航索引)
├── 🟦 PROJECT_OVERVIEW.md (项目总览)
│   ├── 项目结构
│   ├── 组件详解
│   ├── 消融设计
│   └── 快速使用
│
├── 🟩 EXPERIMENTS_README.md (实验框架)
│   ├── 核心特性
│   ├── 快速开始
│   ├── 消融矩阵
│   └── 性能指标
│
├── 🟨 ABLATION_GUIDE.md (消融指南)
│   ├── 实验分组
│   ├── 使用方法
│   ├── 配置系统
│   ├── 输出结构
│   └── 高级用法
│
└── 🟧 VERSION_GUIDE.md (版本对比)
    ├── v1原始版本
    ├── v2简化版本
    └── 差异说明
```

---

## 💡 核心概念速记

| 概念 | 说明 |
|------|------|
| **Selector** | 从49个ViT补丁中选择K个关键补丁 |
| **Fuser** | 将K个补丁融合为单个紧凑表示 |
| **Loss** | 多个目标函数的加权组合 |
| **Ablation** | 通过去掉/加入组件来验证其价值 |
| **Group** | 一个完整的实验配置组合 |
| **Seed** | 随机初始化，用于评估稳定性 |

---

## 🎯 发表前检查清单

- [ ] 5个实验组全部完成
- [ ] 3个随机种子全部完成  
- [ ] 生成了 `ablation_results.json`
- [ ] 生成了 `ABLATION_REPORT.md`
- [ ] 生成了 `ANALYSIS_REPORT.md`
- [ ] CSV已导出用于论文
- [ ] 最佳模型检查点已保存
- [ ] 配置文件已备份
- [ ] 代码已注释完整
- [ ] 训练日志已保存

---

## 🔗 快速链接

| 文档 | 最适用场景 |
|------|----------|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | 全面了解项目 |
| [EXPERIMENTS_README.md](EXPERIMENTS_README.md) | 开始实验 |
| [ABLATION_GUIDE.md](ABLATION_GUIDE.md) | 详细实验指南 |
| [VERSION_GUIDE.md](VERSION_GUIDE.md) | 理解版本差异 |
| [README.md](README.md) | 项目基础信息 |

---

## ⚡ 30秒快速开始

```bash
# 1. 查看可用实验
python run_quick_experiment.py --list

# 2. 运行Baseline
python run_quick_experiment.py --group baseline --gpu 0

# 3. 查看结果
python analyze_results.py
```

---

**页面版本**: 1.0  
**最后更新**: 2024  
**快速导航索引** - 帮你找到正确的文档和命令
