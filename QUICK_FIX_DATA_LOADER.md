# 🚀 数据加载器错误 - 快速修复指南

## 错误现象
```
AssertionError: at line 416 in build_data_loader
assert len(data_loader) > 0
```

## 最常见的原因和快速修复

### ❌ 原因 1：数据路径不存在或错误

```bash
# 检查路径
ls -la /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/
# ↑ 这个路径不存在？

# 查找正确的路径
find /data -name "ImageNet*" -type d 2>/dev/null

# 更新配置
nano configs/my_config.yaml
# 修改 root_path 为正确的路径
```

### ❌ 原因 2：数据分割为空

```bash
# 检查 train 和 val 文件夹
ls -la /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/train/ | head -3
ls -la /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/val/ | head -3

# 如果 val 为空，两种解决方案：
# 1. 修改脚本使用 train 作为 test
# 2. 或重新组织数据
```

### ❌ 原因 3：classnames.txt 错误

```bash
# 检查格式
head -3 /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/classnames.txt

# 应该是这样：
# n01440764 tench fish
# n01443537 goldfish fish
# n01484850 great white shark

# 检查文件夹名称是否与第一列匹配
ls /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/train/ | head -3
# 应该输出 n01440764, n01443537, n01484850 等
```

---

## 运行诊断脚本

```bash
# 运行一个小实验看详细输出
python train_modular.py \
    --id_dataset imagenet \
    --root_path /data/ICML2026/clip/FA/my_dataset \
    --shots 16 \
    --seed 1 \
    2>&1 | grep -A 5 "⚠️"

# 这会打印：
# ⚠️  Dataset directory does not exist: ...
# ⚠️  Data directory does not exist: ...
# ⚠️  No class folders in ...
# 等等
```

---

## 完整检查清单

运行以下命令，全部应该显示内容而不是错误：

```bash
# 1. 检查 ImageNet-1K 目录
test -d /data/ICML2026/clip/FA/my_dataset/ImageNet-1K && echo "✓ ImageNet-1K 存在" || echo "✗ ImageNet-1K 不存在"

# 2. 检查 images 目录
test -d /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images && echo "✓ images 存在" || echo "✗ images 不存在"

# 3. 检查 train 目录
test -d /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/train && echo "✓ train 存在" || echo "✗ train 不存在"

# 4. 检查 val 目录
test -d /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/val && echo "✓ val 存在" || echo "✗ val 不存在"

# 5. 检查类文件夹
test $(ls -1 /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/train/ 2>/dev/null | wc -l) -gt 0 && echo "✓ train 有类文件夹" || echo "✗ train 为空"

# 6. 检查 classnames.txt
test -f /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/classnames.txt && echo "✓ classnames.txt 存在" || echo "✗ classnames.txt 不存在"

# 7. 检查图像文件
test $(ls -1 /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/images/train/n01440764/ 2>/dev/null | wc -l) -gt 0 && echo "✓ train 有图像" || echo "✗ train 图像不存在"
```

---

## 三个快速修复方案

### 方案 A：使用不同的 root_path

```bash
# 如果数据在其他位置
python train_modular.py \
    --id_dataset imagenet \
    --root_path /actual/path/to/imagenet \
    --shots 16 \
    --seed 1 \
    ...
```

### 方案 B：自动使用 train/val split

现在代码已支持：如果 test 为空，自动使用 val 集

```bash
# 运行脚本，会自动处理
python train_modular.py \
    --id_dataset imagenet \
    --root_path /data/ICML2026/clip/FA/my_dataset \
    ...
```

### 方案 C：查看详细的诊断输出

新增的详细输出会告诉你具体问题：

```
⚠️  Data directory does not exist: /path/to/train
   Expected: /path/to/images/{train,val}/{class_folders}/{images}

⚠️  Warning: Test dataset is empty. Using train+val split as fallback.

⚠️  Warning: data_source is empty or None (length: 0)
   This typically means:
   1. Data directory path is incorrect or data is missing
   2. Data split (train/val/test) has no samples
   3. Dataset initialization failed silently
```

---

## 数据目录正确结构

```
/data/ICML2026/clip/FA/my_dataset/
└── ImageNet-1K/
    ├── images/
    │   ├── train/
    │   │   ├── n01440764/        ← 类文件夹
    │   │   │   ├── n01440764_10026.JPEG
    │   │   │   ├── n01440764_10027.JPEG
    │   │   │   └── ...
    │   │   ├── n01443537/
    │   │   │   └── ...
    │   │   └── ... (1000 个类)
    │   └── val/
    │       ├── n01440764/
    │       └── ... (1000 个类)
    └── classnames.txt            ← 类名文件
        (格式: n01440764 tench fish)
```

---

## 需要进一步帮助？

收集以下信息：

```bash
# 1. 你的数据结构
tree -L 4 /data/ICML2026/clip/FA/my_dataset/ImageNet-1K/ 2>/dev/null | head -30

# 2. 你的配置
cat configs/my_config.yaml | grep -E "root_path|id_dataset|shots"

# 3. 诊断输出
python train_modular.py \
    --id_dataset imagenet \
    --root_path /data/ICML2026/clip/FA/my_dataset \
    --shots 16 \
    --seed 1 \
    2>&1 | head -50
```

然后查看 `TROUBLESHOOTING_DATA_LOADER.md` 获取完整的故障排除指南。

