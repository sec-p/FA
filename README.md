# 模块化OOD检测框架

基于 CLIP + ViT-B-16 的异常检测(OOD Detection)框架，支持多种特征选择、融合策略和灵活的训练方式。

## 📂 项目结构

```
FA/
├── src/                          # 核心源代码
│   ├── model_modular.py         # 6种模块化模型定义
│   ├── train_and_eval.py        # ⭐ 推荐：统一训练框架
│   └── utils.py                 # 工具函数库
│
├── scripts/                      # 执行脚本
│   ├── train_all.sh             # 一键启动训练
│   ├── grid_search.sh           # 网格搜索 (Bash版)
│   ├── grid_search_parallel.py  # ⭐ 并行网格搜索 (Python版)
│   └── analyze_results.py       # 结果分析工具
│
├── docs/                         # 文档
│   ├── START_HERE.md            # 🚀 快速入门
│   ├── PROJECT_SUMMARY.md       # 完整项目说明
│   └── GRID_SEARCH_GUIDE.md     # 网格搜索详细指南
│
├── my_dataset/                   # 数据集模块
├── clip/                         # CLIP模型模块
├── ood_utils/                    # OOD工具模块
│
├── requirements.txt              # 项目依赖
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练单个方法 (最简单，从项目根目录运行)

**方式 A: 使用 Python 启动脚本（⭐ 推荐）**
```bash
python run_training.py --method baseline_mean --epochs 50
```

**方式 B: 使用 Bash 启动脚本**
```bash
bash run_training.sh --method baseline_mean --epochs 50
```

**方式 C: 使用完整的训练脚本**
```bash
bash scripts/train_all.sh -m baseline_mean -e 50
```

### 3. 并行网格搜索 (推荐快速对比多个方法)
```bash
cd scripts
python grid_search_parallel.py -m all -e 30 -j 4 --analyze
```

### 4. 查看结果
```bash
cd scripts
python analyze_results.py --summary
```

> ⚠️ **重要**: 所有脚本都应该从项目根目录运行，或使用上面提供的启动脚本！

> 📖 详细说明请查看 `docs/START_HERE.md` 或 `IMPORT_FIX_GUIDE.md`

## 📊 6种模型方法

| 方法 | 复杂度 | 适用场景 |
|------|--------|----------|
| baseline_mean | ⭐ | 基准对标 |
| baseline_attention | ⭐⭐ | 基准改进 |
| selector_mlp | ⭐⭐ | 轻量级选择 |
| selector_slot | ⭐⭐⭐ | 高质量选择 |
| fuser_attention | ⭐⭐⭐ | 融合改进 |
| full_model | ⭐⭐⭐⭐⭐ | 最优性能 |

## 🔧 核心脚本说明

### 训练脚本

**`src/train_and_eval.py`** ⭐ 推荐
- 统一训练框架，每epoch自动OOD评估
- 检查点优化（节省98%存储空间）
- 完整的日志记录

```bash
python src/train_and_eval.py \
    --method full_model \
    --epochs 100 \
    --lr 0.002 \
    --batch_size 64
```

### 网格搜索

**`scripts/grid_search_parallel.py`** ⭐ 推荐
- 高效的并行网格搜索
- 支持多进程加速
- 自动分析结果

```bash
python scripts/grid_search_parallel.py \
    -m all \
    -e 50 \
    -s 1 2 3 \
    -j 4 \
    --analyze
```

**`scripts/grid_search.sh`**
- Bash版本网格搜索
- 简单直观

```bash
bash scripts/grid_search.sh \
    -m baseline_mean selector_mlp full_model \
    -e 50 \
    -s 1 2 3 \
    -lr 0.001 0.002 \
    -bs 32 64
```

### 结果分析

**`scripts/analyze_results.py`**
- 方法性能对比
- 种子稳定性分析
- CSV导出

```bash
# 显示最新结果
python scripts/analyze_results.py --latest

# 方法对比
python scripts/analyze_results.py --compare-methods

# 导出CSV
python scripts/analyze_results.py --export results.csv
```

## 📚 文档

- **START_HERE.md** - 🚀 快速开始指南
- **PROJECT_SUMMARY.md** - 完整项目文档
- **GRID_SEARCH_GUIDE.md** - 网格搜索详细指南

## 💡 常见工作流

### 场景1: 快速验证
```bash
cd src
python train_and_eval.py --method baseline_mean --epochs 5
```

### 场景2: 方法对比
```bash
cd scripts
python grid_search_parallel.py -m all -e 20 -j 4 --analyze
```

### 场景3: 超参优化
```bash
cd scripts
bash grid_search.sh -m full_model -e 50 -s 1 2 3 4 5
```

### 场景4: 完整实验
```bash
cd scripts
python grid_search_parallel.py -m all -e 100 -s 1 2 3 4 5 -j 8 --analyze
```

## 📈 性能指标

- **ID准确率** - ImageNet-1K分类准确率（越高越好，目标>60%）
- **AUROC** - OOD检测曲线面积（越高越好，目标>85%）
- **FPR95** - 95%真正率下的假正率（越低越好，目标<5%）

## ⚠️ 常见问题

**Q: 训练很慢？**
- 增加batch_size (32→64)
- 使用并行网格搜索: `python scripts/grid_search_parallel.py ... -j 4`
- 使用GPU加速

**Q: 检查点太大？**
- 已自动优化，只保存可训练参数

**Q: 从中断处继续？**
- 使用 `--resume` 参数

## 🎯 推荐工作流

```
1. 快速验证环境
   └─> python src/train_and_eval.py --method baseline_mean --epochs 5

2. 方法对比实验
   └─> python scripts/grid_search_parallel.py -m all -e 20 -j 4 --analyze

3. 最佳方法优化
   └─> bash scripts/grid_search.sh -m full_model -e 50 -s 1 2 3 4 5

4. 结果分析总结
   └─> python scripts/analyze_results.py --summary
```

---

**项目状态**: ✅ 生产就绪  
**最后更新**: 2025-12-18
