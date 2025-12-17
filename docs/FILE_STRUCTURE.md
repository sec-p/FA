# 项目文件结构说明

## 核心训练文件（根目录）

| 文件 | 功能 | 是否需要 |
|------|------|--------|
| `train_modular.py` | 主训练脚本 | ✓ 必需 |
| `model_modular.py` | 模型定义 | ✓ 必需 |
| `utils.py` | 工具函数 | ✓ 必需 |
| `requirements.txt` | 依赖包 | ✓ 必需 |

## 数据集（根目录）

| 文件/文件夹 | 功能 | 是否需要 |
|-----------|------|--------|
| `my_dataset/` | 数据集处理模块 | ✓ 必需 |
| `ood_utils/` | OOD检测工具 | ✓ 必需 |
| `clip/` | CLIP模型文件 | ✓ 必需 |

## 配置文件（`configs/`）

| 文件 | 功能 | 是否需要 |
|------|------|--------|
| `my_config.yaml` | 主配置文件 | ✓ 必需 |

## 脚本文件（`scripts/`）

| 文件 | 功能 |
|------|------|
| `run_all_methods.sh` | 运行所有方法组合（顺序）|
| `run_methods_part1.sh` | 运行方法组合1-2 |
| `run_methods_part2.sh` | 运行方法组合3-4 |
| `run_methods_part3.sh` | 运行方法组合5-6 |
| `run_parallel.sh` | 并行运行所有3部分 |

## 文档（`docs/`）

应该放在这里的文档：
- README.md
- QUICK_START.md
- PROJECT_OVERVIEW.md
- VERSION_GUIDE.md
- ABLATION_GUIDE.md
- EXPERIMENTS_README.md
- 以及其他 .md 文档

## 实验（`ablations/`）

应该放在这里的文件：
- run_ablation_study.py
- run_quick_experiment.py
- verify_implementation.py
- test_eval_feature.py

## 过时/不需要的文件（可删除）

以下文件是早期版本，现已被 `model_modular.py` 和 `train_modular.py` 替代：

| 文件 | 原因 |
|------|------|
| `main.py` | 已过时，用 train_modular.py 替代 |
| `main_v2.py` | 已过时，用 train_modular.py 替代 |
| `model.py` | 已过时，用 model_modular.py 替代 |
| `model_v2.py` | 已过时，用 model_modular.py 替代 |
| `run_ablation_study.py` | 可放到 ablations/ 文件夹 |
| `run_quick_experiment.py` | 可放到 ablations/ 文件夹 |
| `verify_implementation.py` | 可放到 ablations/ 文件夹 |
| `test_eval_feature.py` | 可放到 ablations/ 文件夹 |
| `analyze_results.py` | 可放到 scripts/ 文件夹 |
| `examples_icml_features.py` | 可放到 ablations/ 文件夹 |

## 生成的文件（请勿手动编辑）

| 文件/文件夹 | 用途 |
|-----------|------|
| `__pycache__/` | Python缓存 |
| `.git/` | 版本控制 |
| `my_caches/` | 训练输出（模型、日志、可视化）|
| `logs/` | 脚本运行日志 |
| `configs/temp_grid.yaml` | 临时配置文件（脚本运行时生成）|

## 清理建议

```bash
# 删除过时的主文件
rm main.py main_v2.py model.py model_v2.py

# 移动文档到 docs/
mv *.md docs/

# 移动实验脚本到 ablations/
mv run_ablation_study.py ablations/
mv run_quick_experiment.py ablations/
mv verify_implementation.py ablations/
mv test_eval_feature.py ablations/
mv examples_icml_features.py ablations/

# 移动分析脚本到 scripts/
mv analyze_results.py scripts/

# 清理 __pycache__（如果需要）
rm -rf __pycache__
```

## 使用工作流

### 1. 快速测试（单个实验）
```bash
python train_modular.py --config configs/my_config.yaml --is_train 1
```

### 2. 顺序网格搜索（所有方法组合）
```bash
bash scripts/run_all_methods.sh
```

### 3. 并行网格搜索（推荐）
```bash
bash scripts/run_parallel.sh
```

### 4. 分析结果
```bash
python scripts/analyze_results.py --cache_root ./my_caches
```

## 文件大小查询

```bash
# 查看各类文件的大小
du -sh * | sort -h

# 查看最大的文件
ls -lhS | head -20
```
