# 项目总结：模块化OOD检测框架

## 📋 项目概览

这是一个基于 **CLIP + ViT-B-16** 的模块化异常检测(OOD Detection)框架，支持多种特征选择、融合策略和灵活的训练方式。

**主要特性：**
- 6种预定义模型方法（从基础到完整）
- 支持多个数据集（ImageNet-1K ID + 4个OOD数据集）
- 自动化网格搜索和超参调优
- 全面的结果分析和对比
- 优化的检查点保存（节省98%存储空间）

---

## 📂 文件组织

### 一、核心模型文件

#### **model_modular.py** (33.9 KB)
```
核心模型架构定义
├─ 基础类
│  ├─ BaseSelector: 特征选择器基类
│  └─ BaseFuser: 特征融合器基类
├─ 选择器实现
│  ├─ MLPSelector: MLP特征选择
│  ├─ SlotAttentionSelector: Slot Attention选择
│  └─ SimpleSelector: 简单平均选择
├─ 融合器实现
│  ├─ MeanFuser: 平均融合
│  ├─ QueryAttentionFuser: 查询注意力融合
│  └─ SelfAttentionFuser: 自注意力融合
└─ 6种完整方法配置
   ├─ baseline_mean: 直接平均(最简单)
   ├─ baseline_attention: 注意力平均
   ├─ selector_mlp: MLP选择 + 平均融合
   ├─ selector_slot: Slot Attention选择 + 平均融合
   ├─ fuser_attention: MLP选择 + 注意力融合
   └─ full_model: Slot + 自注意力融合(最复杂)
```

**关键函数：**
- `build_modular_model()` - 构建指定配置的模型

---

### 二、训练脚本

#### **train_and_eval.py** ⭐ **推荐** (21.2 KB)
```
统一训练和评估框架
├─ TrainingConfig: 配置管理类
│  ├─ METHODS: 6种方法定义
│  ├─ OOD_DATASETS: 4个OOD数据集
│  └─ 默认超参数设置
├─ TrainEvalOrchestrator: 主编排类
│  ├─ setup_data(): 初始化数据加载器
│  ├─ setup_model(): 构建CLIP+模块化模型
│  ├─ train_epoch(): 单轮epoch训练
│  ├─ evaluate_id(): ID数据集准确率评估
│  ├─ evaluate_ood_dataset(): OOD评估(AUROC, FPR95)
│  ├─ save_checkpoint(): 优化的检查点保存
│  └─ train_with_eval(): 主训练循环(每epoch自动评估)
└─ 特性
   ├─ 按epoch自动评估ID+OOD性能
   ├─ 检查点优化(只保存可训练参数)
   ├─ 灵活的超参数配置
   └─ 完整的日志记录
```

**核心优化：**
- 检查点大小：420MB → 7.5MB (节省98.2%)
- 每epoch自动评估OOD性能
- 支持从检查点恢复训练

**使用方式：**
```bash
# 基本训练 (1个方法, 50个epoch)
python train_and_eval.py --method baseline_mean --epochs 50

# 高级配置
python train_and_eval.py \
    --method full_model \
    --epochs 100 \
    --lr 0.002 \
    --batch_size 64 \
    --seed 42 \
    --output_dir results/exp1
```

---

### 三、启动脚本

#### **train_all.sh** (8.9 KB)
```
简单的一键训练启动器
├─ 功能
│  ├─ 支持单个或全部方法训练
│  ├─ 彩色输出和进度指示
│  ├─ 参数灵活配置
│  ├─ 训练前确认提示
│  └─ 错误处理
└─ 使用示例
   ├─ bash train_all.sh baseline_mean   # 训练单个方法
   ├─ bash train_all.sh all             # 训练所有方法
   └─ bash train_all.sh full_model -e 100 -b 64 -lr 0.002
```

---

### 四、网格搜索工具

#### **grid_search.sh** (10.3 KB)
```
Bash版本网格搜索
├─ 功能
│  ├─ 支持多个方法同时搜索
│  ├─ 方法、种子、学习率、batch_size网格组合
│  ├─ 自动统计结果
│  └─ 完整的实验日志
└─ 使用示例
   ├─ bash grid_search.sh -m baseline_mean -e 20 -s 1 2 3 -lr 0.001 0.002 -bs 32 64
   ├─ bash grid_search.sh -m all -e 50 -s 1 -lr 0.001
   └─ bash grid_search.sh -m baseline_mean selector_mlp -e 30 -s 1 2 3 4 5
```

**关键特性：**
- 支持方法列表：`-m method1 method2 ...`
- 种子设置：`-s seed1 seed2 ...`
- 学习率网格：`-lr lr1 lr2 ...`
- Batch大小：`-bs bs1 bs2 ...`
- 总实验数 = #methods × #seeds × #lrs × #batch_sizes

---

#### **grid_search_parallel.py** ⭐ **推荐** (8.8 KB)
```
Python版本并行网格搜索(高效)
├─ 核心类
│  ├─ ExperimentConfig: 单个实验定义
│  └─ GridSearchOrchestrator: 并行执行管理
├─ 功能
│  ├─ 多进程并行执行(-j参数指定并发数)
│  ├─ 自动进度显示+ETA计算
│  ├─ JSON格式结果追踪
│  ├─ 可选自动分析(--analyze标志)
│  └─ 实验配置数据类管理
└─ 使用示例
   ├─ python grid_search_parallel.py -m all -e 50 -s 1 2 3 -j 4 --analyze
   ├─ python grid_search_parallel.py -m baseline_mean selector_mlp full_model -e 20 -j 2
   └─ python grid_search_parallel.py -m all -e 50 -j 4 --analyze
```

**并行优化：**
- `-j 4` 表示4个并发进程
- 自动计算完成时间ETA
- 结果自动保存为JSON
- 支持实时进度跟踪

---

### 五、结果分析工具

#### **analyze_results.py** (10.7 KB)
```
结果分析和对比工具
├─ 核心方法
│  ├─ show_latest(): 显示最新结果
│  ├─ compare_methods(): 方法性能对比
│  ├─ compare_seeds(): 种子稳定性分析
│  ├─ show_summary(): 综合排名显示
│  └─ export_csv(): CSV格式导出
└─ 使用示例
   ├─ python analyze_results.py --latest           # 显示最新结果
   ├─ python analyze_results.py --compare-methods  # 方法对比
   ├─ python analyze_results.py --compare-seeds    # 种子稳定性
   ├─ python analyze_results.py --summary          # 综合排名
   └─ python analyze_results.py --export results.csv
```

**分析指标：**
- ID准确率 (Accuracy)
- OOD AUROC
- OOD FPR95
- 方法间性能差异
- 跨种子的稳定性

---

### 六、辅助工具

#### **utils.py** (5.5 KB)
```
通用工具函数库
├─ Logger: 日志记录工具
├─ cls_acc(): 分类准确率计算
├─ 其他通用函数
└─ 支持在所有脚本中使用
```

#### **requirements.txt** (0.1 KB)
```
项目依赖列表
├─ 核心: torch, torchvision, pytorch-lightning
├─ CLIP: openai-clip
├─ 数据: pillow, numpy, scipy
└─ 其他: tqdm, pyyaml
```

---

### 七、文档文件

#### **TRAINING_GUIDE.md** - 训练指南
#### **GRID_SEARCH_GUIDE.md** - 网格搜索详细指南  
#### **QUICK_REFERENCE.md** - 快速参考

---

## 🚀 快速开始

### 1. 环境设置
```bash
# 安装依赖
pip install -r requirements.txt

# 如需使用CLIP
pip install openai-clip
```

### 2. 最简单的开始方式
```bash
# 训练单个方法 (50个epoch)
python train_and_eval.py --method baseline_mean

# 或使用启动器
bash train_all.sh baseline_mean
```

### 3. 快速网格搜索
```bash
# 方式A: Bash版本(简单直观)
bash grid_search.sh -m baseline_mean selector_mlp -e 20 -s 1 2 -lr 0.001 0.002

# 方式B: Python并行版本(高效快速) ⭐ 推荐
python grid_search_parallel.py -m baseline_mean selector_mlp full_model -e 30 -j 4 --analyze
```

### 4. 分析结果
```bash
# 查看最新结果
python analyze_results.py --latest

# 方法对比
python analyze_results.py --compare-methods

# 导出为CSV
python analyze_results.py --export results.csv
```

---

## 📊 模型方法对比

| 方法 | 选择器 | 融合器 | 复杂度 | 推荐场景 |
|------|--------|--------|--------|----------|
| baseline_mean | ✗ | 平均 | ⭐ 最简单 | 基准对标 |
| baseline_attention | ✗ | 查询注意力 | ⭐⭐ | 基准改进 |
| selector_mlp | MLP | 平均 | ⭐⭐ | 轻量级选择 |
| selector_slot | Slot注意力 | 平均 | ⭐⭐⭐ | 高质量选择 |
| fuser_attention | MLP | 查询注意力 | ⭐⭐⭐ | 融合改进 |
| full_model | Slot注意力 | 自注意力 | ⭐⭐⭐⭐⭐ 最复杂 | 最优性能 |

---

## 💾 数据集和结果

### 使用的数据集
- **ID数据集**: ImageNet-1K (1.2M图像, 1000类)
- **OOD数据集**: 
  - SUN397 (街景, 397类)
  - DTD (纹理, 47类)
  - EuroSAT (遥感, 10类)
  - Oxford Pets (宠物, 37类)

### 结果保存位置
```
results/
├─ {method}_{seed}/
│  ├─ checkpoints/          # 训练检查点
│  │  ├─ epoch_01.pth
│  │  ├─ epoch_02.pth
│  │  └─ best.pth
│  ├─ logs/                 # 训练日志
│  │  └─ training.log
│  └─ metrics/              # 评估指标
│     ├─ id_accuracy.json
│     └─ ood_metrics.json
└─ grid_search_results.json # 网格搜索汇总
```

---

## 🔧 常见使用场景

### 场景1: 快速验证想法
```bash
# 用最小配置快速训练
python train_and_eval.py --method baseline_mean --epochs 5 --batch_size 16
```

### 场景2: 方法对比实验
```bash
# 网格搜索不同方法，固定其他参数
python grid_search_parallel.py \
    -m baseline_mean baseline_attention selector_mlp selector_slot fuser_attention full_model \
    -e 50 \
    -s 1 2 3 \
    -j 4 \
    --analyze
```

### 场景3: 超参调优
```bash
# 对最佳方法进行学习率和batch_size调优
bash grid_search.sh \
    -m full_model \
    -e 50 \
    -s 1 2 3 \
    -lr 0.0005 0.001 0.002 0.005 \
    -bs 16 32 64
```

### 场景4: 完整再现实验
```bash
# 完整的网格搜索(所有方法、多个种子)
python grid_search_parallel.py \
    -m all \
    -e 100 \
    -s 1 2 3 4 5 \
    -lr 0.001 0.002 \
    -bs 32 64 \
    -j 8 \
    --analyze
```

---

## 📈 性能指标解释

### ID准确率 (ID Accuracy)
- 在ImageNet-1K上的分类准确率
- 越高越好 (目标: > 60%)

### AUROC (Area Under ROC Curve)
- OOD检测性能指标
- 1.0 = 完美分离, 0.5 = 随机
- 越高越好 (目标: > 85%)

### FPR95 (False Positive Rate at 95% TPR)
- 当真正率为95%时的假正率
- 越低越好 (目标: < 5%)

---

## ⚠️ 常见问题

**Q: 训练很慢，怎么加速？**
A: 
1. 增加batch_size (32→64)
2. 使用并行网格搜索: `python grid_search_parallel.py ... -j 4`
3. 减少epoch数进行快速验证
4. 使用GPU加速

**Q: 检查点太大了？**
A: `train_and_eval.py`已自动优化，只保存可训练参数，自动节省98%空间

**Q: 如何从中断处继续训练？**
A: 在`train_and_eval.py`中使用`--resume`参数恢复最后的检查点

**Q: 结果在哪里？**
A: 所有结果保存在`results/`目录下，按方法和种子组织

---

## 📝 脚本使用总结表

| 脚本 | 目的 | 用途 | 推荐场景 |
|------|------|------|----------|
| `train_and_eval.py` | 训练+评估 | 主要训练引擎 | ⭐ 日常使用 |
| `train_all.sh` | 快速启动 | 一键训练 | 快速测试 |
| `grid_search.sh` | 网格搜索(Bash) | 系统探索 | 简单调优 |
| `grid_search_parallel.py` | 网格搜索(并行) | 高效搜索 | ⭐ 深度优化 |
| `analyze_results.py` | 结果分析 | 性能对比 | 实验总结 |

---

## 🎯 推荐工作流

```
1. 快速验证
   └─> python train_and_eval.py --method baseline_mean --epochs 5
   
2. 方法对比
   └─> python grid_search_parallel.py -m all -e 20 -j 4 --analyze
   
3. 最佳方法优化
   └─> bash grid_search.sh -m full_model -e 50 -s 1 2 3 4 5 -lr 0.0005 0.001 0.002
   
4. 结果分析
   └─> python analyze_results.py --summary --export final_results.csv
```

---

**最后更新**: 2025-12-18
**项目状态**: ✅ 生产就绪
