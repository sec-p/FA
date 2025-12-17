# 🚀 快速入门指南

## 📦 第一步：安装依赖

```bash
pip install -r requirements.txt
```

---

## 🎯 第二步：选择你的工作方式

### 选项A：最简单 - 单个方法训练 (推荐新手)
```bash
python train_and_eval.py --method baseline_mean --epochs 50
```
✅ 一条命令启动，自动进行50轮epoch的训练和OOD评估

---

### 选项B：快速对比 - 网格搜索多个方法 (推荐快速实验)
```bash
python grid_search_parallel.py \
    -m baseline_mean selector_mlp full_model \
    -e 30 \
    -j 4 \
    --analyze
```
✅ 并行运行3个方法，自动分析结果

---

### 选项C：完整优化 - 超参调优 (推荐深度研究)
```bash
bash grid_search.sh \
    -m full_model \
    -e 50 \
    -s 1 2 3 \
    -lr 0.0005 0.001 0.002 \
    -bs 32 64
```
✅ 系统搜索最优超参数组合

---

## 📊 第三步：查看结果

```bash
# 显示最新实验结果
python analyze_results.py --latest

# 比较不同方法的性能
python analyze_results.py --compare-methods

# 导出结果为CSV
python analyze_results.py --export results.csv
```

---

## 📖 文件功能速查表

| 文件 | 功能 | 何时使用 |
|------|------|----------|
| `train_and_eval.py` | 主训练引擎 | ⭐ 日常使用 |
| `train_all.sh` | 启动器脚本 | 快速开始 |
| `grid_search_parallel.py` | 并行网格搜索 | 超参优化 |
| `grid_search.sh` | Bash网格搜索 | 简单搜索 |
| `analyze_results.py` | 结果分析工具 | 实验总结 |
| `model_modular.py` | 核心模型代码 | 了解模型 |

---

## 💡 常用命令速记

```bash
# 快速验证 (5个epoch快速测试)
python train_and_eval.py --method baseline_mean --epochs 5

# 标准训练 (50个epoch完整训练)
python train_and_eval.py --method full_model --epochs 50

# 4并发的方法对比
python grid_search_parallel.py -m all -e 30 -j 4

# 显示性能排名
python analyze_results.py --summary
```

---

## 🎓 6种模型方法简介

1. **baseline_mean** - 直接平均 (最简单) ⭐
2. **baseline_attention** - 注意力平均
3. **selector_mlp** - MLP选择器
4. **selector_slot** - Slot Attention选择器
5. **fuser_attention** - 注意力融合器
6. **full_model** - 完整模型 (最强) ⭐⭐

> 💡 **建议**: 先用`baseline_mean`验证环境，再试`full_model`追求最优性能

---

## ❓ 需要帮助？

📖 详细文档: 查看 `PROJECT_SUMMARY.md`
🔧 网格搜索: 查看 `GRID_SEARCH_GUIDE.md`

---

**就这么简单！现在开始你的第一个实验吧** 🚀
