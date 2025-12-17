# 导入问题修复指南

## 问题描述

当运行 `python src/train_and_eval.py` 时出现以下错误：
```
ModuleNotFoundError: No module named 'my_dataset'
ImportError: attempted relative import with no known parent package
```

## 根本原因

1. `train_and_eval.py` 在 `src/` 目录中，但需要导入项目根目录中的模块（`my_dataset`, `clip` 等）
2. Python 的导入系统默认不会自动查找上层目录
3. 相对导入需要项目结构正确的 `__init__.py` 文件支持

## 解决方案

### 方案1：从项目根目录运行（推荐）

```bash
# 方式A: 使用快捷脚本 (最简单)
python run_training.py --method baseline_mean --epochs 50

# 方式B: 使用bash脚本
bash run_training.sh --method baseline_mean --epochs 50

# 方式C: 直接运行 (需要设置PYTHONPATH)
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/train_and_eval.py --method baseline_mean --epochs 50
```

### 方案2：使用 scripts/ 中的启动器

```bash
# 单个方法训练
bash scripts/train_all.sh -m baseline_mean -e 50

# 所有方法
bash scripts/train_all.sh -m all -e 50

# 自定义参数
bash scripts/train_all.sh -m full_model -e 100 -lr 0.002 -bs 64 -s 42
```

### 方案3：并行网格搜索

```bash
# 从项目根目录运行
python scripts/grid_search_parallel.py -m all -e 30 -j 4 --analyze
```

## 技术改进

我们进行了以下改进来确保导入能正确工作：

### 1. 更新 `src/train_and_eval.py` 的导入

```python
# 添加项目根目录到Python路径
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用正确的导入路径
from src.model_modular import build_modular_model
from src.utils import Logger, cls_acc
from my_dataset import build_dataset
```

### 2. 创建 `src/__init__.py`

使 `src` 目录成为有效的 Python 包，支持导入：

```python
from .model_modular import build_modular_model
from .utils import Logger, cls_acc
```

### 3. 更新启动脚本

- `train_all.sh` - 自动计算项目根目录，正确传递配置文件路径
- `run_training.sh` - 从任何位置运行的简单启动脚本
- `run_training.py` - Python 版本的启动脚本

### 4. 更新 `grid_search_parallel.py`

```python
def to_command(self) -> List[str]:
    """Convert to train_and_eval.py command"""
    # 使用绝对路径
    script_dir = Path(__file__).parent.parent / "src"
    train_script = script_dir / "train_and_eval.py"
    
    return [
        "python3", str(train_script),
        # ... 参数
    ]
```

## 推荐使用方式

### 快速开始

```bash
# 方式1: 最简单 (从项目根目录)
python run_training.py --method baseline_mean --epochs 50

# 方式2: 使用 bash 脚本 (从项目根目录)
bash scripts/train_all.sh -m baseline_mean -e 50

# 方式3: 并行网格搜索 (从项目根目录)
python scripts/grid_search_parallel.py -m all -e 30 -j 4 --analyze
```

### 高级用法

```bash
# 完整的网格搜索
python scripts/grid_search_parallel.py \
    -m baseline_mean selector_mlp full_model \
    -e 50 \
    -s 1 2 3 4 5 \
    -lr 0.001 0.002 \
    -bs 32 64 \
    -j 8 \
    --analyze

# 从脚本启动 (支持后台运行)
nohup bash scripts/train_all.sh -m all -e 100 > training.log 2>&1 &

# 查看进度
tail -f training.log
```

## 文件结构（确认所有文件位置）

```
FA/
├── run_training.py         ← 新建：Python启动脚本
├── run_training.sh         ← 新建：Bash启动脚本
├── src/
│   ├── __init__.py         ← 新建：包定义
│   ├── train_and_eval.py   ← 已更新：导入修复
│   ├── model_modular.py
│   └── utils.py
├── scripts/
│   ├── train_all.sh        ← 已更新：路径修复
│   ├── grid_search_parallel.py ← 已更新：路径修复
│   ├── grid_search.sh
│   └── analyze_results.py
├── my_dataset/
├── clip/
├── configs/
└── requirements.txt
```

## 环境变量设置（可选）

如果需要手动设置 Python 路径：

```bash
# Bash
export PYTHONPATH=$PYTHONPATH:/data/ICML2026/clip/FA

# 永久设置 (添加到 ~/.bashrc)
echo 'export PYTHONPATH=$PYTHONPATH:/data/ICML2026/clip/FA' >> ~/.bashrc
source ~/.bashrc
```

## 故障排查

### 问题：仍然无法找到 my_dataset

**解决方案：**
```bash
# 确认项目结构
ls -la src/
ls -la my_dataset/

# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"

# 直接测试导入
cd /data/ICML2026/clip/FA
python -c "import my_dataset; print(my_dataset.__file__)"
```

### 问题：权限被拒绝

**解决方案：**
```bash
# 添加执行权限
chmod +x run_training.sh
chmod +x run_training.py
chmod +x scripts/train_all.sh
```

### 问题：仍然出现导入错误

**解决方案：**
```bash
# 重新安装依赖
pip install -r requirements.txt

# 检查当前Python版本
python --version

# 使用特定的Python版本
python3.10 run_training.py --method baseline_mean --epochs 5
```

## 总结

| 方式 | 位置 | 命令 | 推荐度 |
|------|------|------|--------|
| Python 启动脚本 | 项目根 | `python run_training.py` | ⭐⭐⭐ 最推荐 |
| Bash 启动脚本 | 项目根 | `bash run_training.sh` | ⭐⭐⭐ 推荐 |
| 训练启动器 | scripts/ | `bash scripts/train_all.sh` | ⭐⭐ 可用 |
| 并行搜索 | scripts/ | `python scripts/grid_search_parallel.py` | ⭐⭐⭐ 推荐 |

---

**最后更新**: 2025-12-18  
**状态**: ✅ 所有导入问题已修复
