# 🔍 网格搜索指南

## 📋 概述

网格搜索工具支持对不同**方法**和**超参数**的组合进行系统性训练和评估。基于 `scripts/run_all_methods.sh` 的设计思路，提供两种实现方式：

1. **`grid_search.sh`** - Bash脚本（简单，单线程）
2. **`grid_search_parallel.py`** - Python脚本（功能完整，支持并行）

## 🚀 快速开始

### 方式1️⃣: 使用Bash脚本（简单）

```bash
# 基础网格搜索
bash grid_search.sh

# 指定方法
bash grid_search.sh -m baseline_mean selector_mlp full_model -e 20

# 自定义参数
bash grid_search.sh \
  -m all \
  -e 50 \
  -s 1 2 3 \
  -lr 0.001 0.002 \
  -bs 32 64
```

### 方式2️⃣: 使用Python脚本（功能完整）

```bash
# 基础网格搜索（并行）
python3 grid_search_parallel.py -j 4

# 指定方法和参数
python3 grid_search_parallel.py \
  -m baseline_mean selector_mlp full_model \
  -e 50 \
  -s 1 2 3 \
  -lr 0.001 0.002 \
  -bs 32 64 \
  -j 4

# 网格搜索并自动分析结果
python3 grid_search_parallel.py -j 4 --analyze
```

## 📊 配置选项

### Bash脚本 (`grid_search.sh`)

```bash
-m, --methods           指定方法（或 'all'）
-e, --epochs            每个实验的epoch数
-s, --seeds             随机种子列表
-lr, --learning-rates   学习率列表
-bs, --batch-sizes      batch大小列表
-c, --config            配置文件路径
-p, --parallel          并行工作数（预留，当前不支持）
-h, --help              显示帮助
```

### Python脚本 (`grid_search_parallel.py`)

```bash
-m, --methods           方法列表
-e, --epochs            epoch数
-s, --seeds             随机种子列表
-lr, --learning-rates   学习率列表
-bs, --batch-sizes      batch大小列表
-c, --config            配置文件
-j, --jobs              并行工作数
-l, --log-dir           日志目录
--analyze               完成后自动分析结果
-h, --help              显示帮助
```

## 📈 典型使用场景

### 场景1: 快速探索（少数实验）

```bash
# Bash脚本 - 简单快速
bash grid_search.sh \
  -m baseline_mean selector_mlp \
  -e 10 \
  -s 1 \
  -lr 0.001 \
  -bs 32
```

### 场景2: 完整实验（多个方法和参数）

```bash
# Python脚本 - 并行执行
python3 grid_search_parallel.py \
  -m baseline_mean baseline_attention selector_mlp selector_slot fuser_attention full_model \
  -e 50 \
  -s 1 2 3 \
  -lr 0.0005 0.001 0.002 \
  -bs 32 64 \
  -j 4 \
  --analyze
```

### 场景3: 消融研究（不同学习率）

```bash
# 比较学习率的影响
python3 grid_search_parallel.py \
  -m full_model \
  -e 50 \
  -s 1 2 3 \
  -lr 0.0001 0.0005 0.001 0.002 0.005 \
  -bs 32 \
  -j 4 \
  --analyze
```

### 场景4: 后台运行大规模搜索

```bash
# 使用nohup进行后台运行
nohup python3 grid_search_parallel.py \
  -m all \
  -e 50 \
  -s 1 2 3 \
  -lr 0.0005 0.001 0.002 \
  -bs 32 64 \
  -j 8 \
  --analyze > grid_search.log 2>&1 &

# 监控进度
tail -f grid_search.log
```

## 🔢 计算实验数量

实验总数 = 方法数 × 种子数 × 学习率数 × batch_size数

### 示例计算

**配置1: 快速探索**
- 方法: 2个
- 种子: 1个
- 学习率: 1个
- batch_size: 1个
- **总计: 2个实验**

**配置2: 完整搜索**
- 方法: 6个
- 种子: 3个
- 学习率: 3个
- batch_size: 2个
- **总计: 108个实验**

## ⏱️ 运行时间估计

假设单个实验时间 = 1-2小时

### 串行执行

```
实验数    时间
───────────────
10个     10-20小时
50个     50-100小时
108个    108-216小时
```

### 并行执行（4个worker）

```
实验数    时间（4并行）
──────────────────
10个     3-5小时
50个     13-25小时
108个    27-54小时
```

## 📁 输出结构

### Bash脚本输出

```
results/
├── baseline_mean_s1_lr0.001_bs32_timestamp/
│   ├── checkpoints/
│   ├── results.json
│   └── training.log
├── baseline_mean_s1_lr0.002_bs32_timestamp/
└── ...

logs/
├── grid_search.log          # 完整日志
└── grid_search_results.json # 实验结果汇总
```

### Python脚本输出

```
results/
└── [同上]

logs/
├── grid_search.log
└── grid_search_results.json  # JSON格式的实验汇总
```

## 📊 分析结果

### 自动分析（使用Python脚本的--analyze）

```bash
python3 grid_search_parallel.py \
  -m all \
  -e 20 \
  -s 1 2 \
  -lr 0.001 0.002 \
  -bs 32 \
  -j 4 \
  --analyze
```

### 手动分析

```bash
# 完整报告
python3 analyze_results.py --summary

# 对比方法
python3 analyze_results.py --compare-methods

# 对比种子（稳定性）
python3 analyze_results.py --compare-seeds

# 导出CSV
python3 analyze_results.py --export-csv grid_search_results.csv
```

## 🎯 支持的方法

| 方法 | 说明 |
|------|------|
| `baseline_mean` | 基础平均池化 |
| `baseline_attention` | 基础注意力 |
| `selector_mlp` | MLP选择器 |
| `selector_slot` | Slot选择器 |
| `fuser_attention` | 注意力融合 |
| `full_model` | 完整模型 |

## 🛠️ 高级用法

### Python脚本 - 自定义实验

```python
from grid_search_parallel import (
    ExperimentConfig,
    GridSearchOrchestrator,
    generate_experiments
)

# 生成实验列表
experiments = generate_experiments(
    methods=['baseline_mean', 'full_model'],
    seeds=[1, 2, 3],
    learning_rates=[0.001, 0.002],
    batch_sizes=[32],
    epochs=50,
    config_file='configs/my_config.yaml'
)

# 运行网格搜索
orchestrator = GridSearchOrchestrator(num_workers=4)
orchestrator.run_grid_search(experiments)
```

### Bash脚本 - 自定义参数范围

编辑 `grid_search.sh` 中的默认值：

```bash
# 修改这部分
declare -a SEEDS=(1 2 3 4 5)
declare -a LEARNING_RATES=(0.00001 0.00005 0.0001 0.0005 0.001)
declare -a BATCH_SIZES=(16 32 64 128)
```

## 💡 最佳实践

### 1. 从小到大探索

```bash
# 第1步: 快速验证
bash grid_search.sh -m baseline_mean -e 5 -s 1 -lr 0.001 -bs 32

# 第2步: 扩展方法
bash grid_search.sh -m baseline_mean selector_mlp full_model -e 20 -s 1 2

# 第3步: 完整搜索
python3 grid_search_parallel.py -m all -e 50 -j 4 --analyze
```

### 2. 后台运行并监控

```bash
# 启动
nohup python3 grid_search_parallel.py -m all -e 50 -j 4 > gs.log 2>&1 &

# 监控
watch -n 60 'tail -20 gs.log'

# 定期检查进度
tail -f gs.log | grep -E "Completed|Failed"
```

### 3. 资源管理

```bash
# 检查GPU使用
nvidia-smi

# 根据GPU显存调整batch_size和并行数
python3 grid_search_parallel.py \
  -m all \
  -bs 16 32  # 较小的batch size
  -j 2       # 较少的并行数
  --analyze
```

## 🐛 常见问题

### Q: 如何选择并行数？
**A**: 根据GPU数量或CPU核心数
- 1个GPU: -j 1
- 2个GPU: -j 2 
- 4个GPU: -j 4
- 多核CPU: -j (核心数/2)

### Q: 网格搜索太慢了
**A**: 
1. 减少参数范围
2. 增加并行数（如果资源允许）
3. 减少epoch数用于初步测试
4. 使用更小的batch_size可能会快一点

### Q: 某个实验失败了怎么办？
**A**: 
1. 检查 `results/` 目录中是否有部分结果
2. 手动运行失败的实验：`python3 train_and_eval.py --method xxx ...`
3. 查看 `logs/grid_search.log` 中的错误信息

### Q: 能否重启中断的网格搜索？
**A**: 目前需要手动重新运行失败的实验。可以检查 `logs/grid_search_results.json` 查看已完成的实验。

## 📊 与原始脚本的对比

| 功能 | 原始脚本 | 新脚本 |
|------|---------|--------|
| 网格搜索 | ✓ | ✓ |
| 并行执行 | 部分支持 | ✓ 完整支持 |
| 方法组合 | 固定 | 灵活配置 |
| 自动分析 | 无 | ✓ (Python版本) |
| 易用性 | 中等 | 高 |

## 📚 参考

- 原始设计: `scripts/run_all_methods.sh`
- 核心训练: `train_and_eval.py`
- 结果分析: `analyze_results.py`

---

**提示**: 对于大规模实验，建议使用Python脚本的并行版本！
