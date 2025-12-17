# 快速开始：并行网格搜索

## 📊 实验规模

- **方法组合**: 6 个 (selector × fuser)
- **超参数维度**: 4 个 (seed × lambda_llm × lambda_mixup × lr)
- **总实验数**: **1,296 个** = 6 × (3 × 3 × 3 × 3)
- **预计时间**: ~25-40 小时（3 GPU 并行）

## 🚀 快速开始（3 步）

### 1️⃣ 查看实验统计
```bash
bash scripts/GRID_SEARCH_STATS.sh
```

### 2️⃣ 运行并行实验
```bash
# 在 GPU 充足的情况下，自动并行运行 3 个部分
bash scripts/run_parallel.sh

# 或手动在 3 个终端分别运行
bash scripts/run_methods_part1.sh  # 终端 1
bash scripts/run_methods_part2.sh  # 终端 2
bash scripts/run_methods_part3.sh  # 终端 3
```

### 3️⃣ 分析结果
```bash
# 等待所有实验完成后
python scripts/analyze_results.py --cache_root ./my_caches --top_n 20
```

## 📁 文件结构

```
FA/
├── scripts/                    # 训练脚本
│   ├── run_all_methods.sh          # 顺序运行所有方法
│   ├── run_methods_part1.sh        # 并行部分 1 (2 个方法)
│   ├── run_methods_part2.sh        # 并行部分 2 (2 个方法)
│   ├── run_methods_part3.sh        # 并行部分 3 (2 个方法)
│   ├── run_parallel.sh             # 主脚本（并行控制）
│   ├── GRID_SEARCH_STATS.sh        # 实验统计
│   ├── organize_files.py           # 文件整理（可选）
│   └── analyze_results.py          # 结果分析
│
├── docs/                       # 文档
│   ├── FILE_STRUCTURE.md           # 文件说明
│   └── (其他 .md 文档)
│
├── ablations/                  # 实验和验证脚本
│   ├── run_ablation_study.py
│   ├── verify_implementation.py
│   └── ...
│
├── configs/                    # 配置文件
│   └── my_config.yaml          # 主配置
│
├── logs/                       # 脚本运行日志
│   ├── part1.log
│   ├── part2.log
│   └── part3.log
│
├── my_caches/                  # 训练输出（自动生成）
│   └── imagenet/ViT-B-16/...
│
├── train_modular.py            # ✓ 主训练脚本（必需）
├── model_modular.py            # ✓ 模型定义（必需）
└── utils.py                    # ✓ 工具函数（必需）
```

## 💡 使用提示

### 监控实验进度
```bash
# 在不同终端查看日志
tail -f logs/part1.log
tail -f logs/part2.log
tail -f logs/part3.log
```

### 查看方法组合详情

| 部分 | 方法组合 | 实验数 |
|------|--------|-------|
| Part 1 | mlp:mean, mlp:query_attn | 216 |
| Part 2 | mlp:self_attn, slot:mean | 216 |
| Part 3 | slot:query_attn, slot:self_attn | 216 |
| **总计** | **6 个方法** | **1,296** |

### 清理旧文件（可选）
```bash
# 查看将要移动/删除的文件
python scripts/organize_files.py

# 实际执行文件整理
python scripts/organize_files.py --execute
```

## 📈 结果位置

训练完成后，所有结果自动保存到：
```
my_caches/imagenet/ViT-B-16/selector_XXX_fuser_YYY/seedN/
├── log_train.txt           # 训练日志
├── eval_history.json       # 每 epoch 的评估指标
├── model_best.pth          # 最优模型权重
└── attention_maps/         # 可视化
```

## ⚙️ 方法组合说明

**Selector Types:**
- `mlp`: 多头 MLP 选择器（轻量级）
- `slot`: Slot Attention 选择器（复杂度高）

**Fuser Types:**
- `mean`: 简单平均融合
- `query_attn`: 文本引导注意力融合
- `self_attn`: 自注意力融合

## 🔧 常见问题

**Q: 如何中断实验？**
```bash
# 杀死所有 Python 进程
pkill -f "train_modular.py"
```

**Q: 如何在中断后继续？**
```bash
# 脚本会跳过已完成的实验（通过检查输出目录）
bash scripts/run_parallel.sh
```

**Q: 如何只运行某个方法组合？**
编辑对应的 `run_methods_partX.sh` 文件，修改 `METHODS` 数组。

**Q: 实验数据存储需求？**
- 每个实验: ~100-500 MB（取决于可视化）
- 总需求: ~500 GB - 1 TB

## 📊 分析结果示例

```bash
# 生成详细报告
python scripts/analyze_results.py \
    --cache_root ./my_caches \
    --top_n 20 \
    --export_top_csv results_top_20.csv

# 输出包括：
# - 总体统计 (平均、最优、最差精度)
# - 前 20 名配置
# - 方法组合对比
# - 导出 CSV 用于后续分析
```

## ✅ 检查清单

- [ ] GPU 充足（建议 3+ 个 GPU）
- [ ] 磁盘空间充足（~500 GB）
- [ ] 配置文件正确 (`configs/my_config.yaml`)
- [ ] 数据集路径正确 (`root_path` 在配置中)
- [ ] 所需包已安装 (`pip install -r requirements.txt`)

## 🎯 下一步

1. ✓ 查看实验规模: `bash scripts/GRID_SEARCH_STATS.sh`
2. ✓ 启动并行实验: `bash scripts/run_parallel.sh`
3. ✓ 监控进度: `tail -f logs/part*.log`
4. ✓ 分析结果: `python scripts/analyze_results.py`

---

**预计完成时间**: 25-40 小时（3 GPU 并行）

**建议**: 在有充足 GPU 资源的服务器上运行此脚本。
