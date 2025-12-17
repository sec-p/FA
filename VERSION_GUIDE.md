# SimpleCLIP 项目 - 版本对比与使用指南

## 📊 两个版本对比

### 版本 1：原始版本（保留 PromptLearner）
- **文件**：`model.py`, `main.py`
- **特点**：
  - ✅ 包含可学习的 PromptLearner（可训练的提示词 token）
  - ✅ 支持 Few-shot 设置
  - ⚠️ 需要训练提示词 + 门控网络
  - ⚠️ 复杂度较高

### 版本 2：简化版本（推荐）**
- **文件**：`model_v2.py`, `main_v2.py`
- **特点**：
  - ✅ **移除 PromptLearner**，使用固定的文本提示
  - ✅ **完全支持全数据集训练**（不再限制 Few-shot）
  - ✅ **仅训练门控网络**（唯一可学习组件）
  - ✅ 代码更简洁、易于维护
  - ✅ 训练更稳定（只有一个网络要优化）

---

## 🚀 快速开始 - 版本 2（推荐）

### 1. 使用 Few-shot 设置（16-shot）
```bash
python main_v2.py --config configs/my_config.yaml --is_train 1 --use_full_data 0
```

### 2. 使用完整数据集
```bash
python main_v2.py --config configs/my_config.yaml --is_train 1 --use_full_data 1
```

### 3. OOD 检测评估
```bash
python main_v2.py --config configs/my_config.yaml --is_train 0 --use_full_data 1
```

---

## ⚙️ 配置文件 - 重要参数

编辑 `configs/my_config.yaml`：

```yaml
# ===== 数据集 =====
id_dataset: 'imagenet'          # ID 数据集
shots: 16                        # Few-shot 数量（若 use_full_data=1 则忽略）
ood_dataset: 'common'            # OOD 评估集

# ===== 模型 =====
backbone: 'ViT-B/16'             # CLIP 骨干
templates: ["a photo of a"]      # 固定文本模板（可多个）
use_rl: False                    # False=软门控, True=RL 门控

# ===== 训练 =====
fine_tune_batch_size: 160        # 批大小
fine_tune_train_epoch: 50        # 训练轮数
lr: 0.002                        # 原始学习率
lr_gate: 0.0002                  # 门控网络学习率（推荐 lr/10）

# ===== 门控超参 =====
lambda_sparsity: 0.001          # 稀疏性权重
lambda_minimal: 0.01            # 最小支撑集权重
gate_hidden: 256                # 门控隐藏层宽度

# ===== 其他 =====
seed: 1
K: 1                            # （版本 2 中忽略）
csc: False                       # （版本 2 中忽略）
```

---

## 🔄 模式切换

### Soft 门控模式（推荐入门）
```yaml
use_rl: False
lambda_sparsity: 0.001
lambda_minimal: 0.01
```
- 门控输出连续 mask ∈ [0, 1]
- 损失 = CE + λ_sp × sparsity + λ_min × minimal
- 训练稳定，快收敛

### RL 门控模式（高级用法）
```yaml
use_rl: True
lambda_sparsity: 0.001
lambda_minimal: 0.01
```
- 门控采样离散 mask ∈ {0, 1}（Bernoulli）
- 使用 REINFORCE + baseline 进行策略更新
- 获得更稀疏的 token 选择，但训练需要调参

---

## 📋 数据集支持

### 支持的 ID 数据集（11 种）
```
'imagenet'              # ImageNet-1K (完整数据集)
'imagenet100'           # ImageNet-100
'oxford_pets'           # Oxford Pets
'oxford_flowers'        # Oxford Flowers-102
'stanford_cars'         # Stanford Cars-196
'dtd'                   # Describable Textures Database
'fgvc'                  # FGVC Aircraft
'food101'               # Food-101
'sun397'                # SUN-397
'ucf101'                # UCF-101 (视频)
'eurosat'               # EuroSAT (卫星)
'caltech101'            # Caltech-101
```

### 支持的 OOD 数据集
```yaml
ood_dataset: 'common'           # ['iNaturalist', 'SUN', 'Places', 'dtd']
ood_dataset: 'challenging'      # ['OpenImage_O', 'NINCO', 'imagenet-o']
ood_dataset: 'nearood'          # ['ssb_hard', 'NINCO']
ood_dataset: 'all'              # 所有上述
```

---

## 🧠 核心改进说明

### 什么被移除了？

#### ❌ PromptLearner
```python
# 旧版本 (v1)：需要优化可学习的 context tokens
self.ctx = nn.Parameter(ctx_vectors)  # [num_classes, n_ctx, 512]
```

```python
# 新版本 (v2)：直接使用固定的模板
templates = ["a photo of a", "a picture of a"]  # 静态
```

#### ❌ 复杂的 forward 逻辑
- 旧版本：tokenize → embedding → 拼接前缀后缀 → 可学习 token 插入 → 编码
- 新版本：直接调用 CLIP 的 encode_text()

### 什么被保留了？

#### ✅ 门控网络
```python
# 核心可训练部分
class GatingNetwork(nn.Module):
    # Input: [B, T, C] local features + [B, C] text features
    # Output: [B, T] per-token mask
```

#### ✅ 软/RL 双模式
```python
if use_rl:
    # 离散采样 + 策略梯度
    mask = Bernoulli(sigmoid(gate_logits)).sample()
else:
    # 连续 mask
    mask = sigmoid(gate_logits)
```

---

## 📊 性能对比

| 指标 | 版本 1（有 PromptLearner） | 版本 2（无 PromptLearner） |
|-----|--------------------------|-------------------------|
| 参数数 | ~260K（prompt） + 门控 | 仅门控 (~100K) |
| 训练时间 | 较长（多个可训练组件） | 较快（仅门控） |
| 精度 | 高（但需更多调参） | 稳定（更易收敛） |
| 内存占用 | 较大 | 较小 |
| 代码复杂度 | 复杂 | 简洁 |

---

## 🎯 使用场景推荐

### 推荐使用版本 2 的场景

✅ **开箱即用**
- 无需调优 prompt
- 快速实验新数据集

✅ **计算资源有限**
- 参数少
- 训练快

✅ **追求稳定性**
- 仅一个网络要训练
- 易于调参

✅ **完整数据集**
- 不受 Few-shot 限制
- 充分利用数据

### 需要版本 1 的场景

⚠️ **需要最大精度**
- PromptLearner 可能有额外优势
- 更灵活的设计

⚠️ **研究新 prompt 学习方法**
- 版本 1 是更好的起点

---

## 🔧 常见问题

### Q1: 如何在不同数据集间切换？
```bash
# 只需改 config
sed -i "s/id_dataset: .*/id_dataset: 'oxford_pets'/" configs/my_config.yaml
python main_v2.py --config configs/my_config.yaml --is_train 1
```

### Q2: 如何使用完整数据集而不是 Few-shot？
```bash
# 方案 A：命令行参数
python main_v2.py --config configs/my_config.yaml --is_train 1 --use_full_data 1

# 方案 B：修改 config（内部使用 shots=-1）
# 这会自动使用完整数据集
```

### Q3: Soft 还是 RL 模式？
- **入门推荐**：Soft（稳定，快速收敛）
- **想要更稀疏特征**：RL（需要更仔细调参）

### Q4: 如何调整门控网络大小？
```yaml
gate_hidden: 512  # 增大网络容量
```

### Q5: 多卡训练？
```bash
# 自动支持 DataParallel
python main_v2.py --config configs/my_config.yaml --is_train 1
```

---

## 📝 输出结构

```
my_caches/
├── imagenet/
│   ├── ViT-B-16/
│   │   ├── fulldata/              # 使用 use_full_data=1
│   │   │   └── SimpleCLIP_v2_batch160_ep50/
│   │   │       ├── K-3/lr0002/seed1/
│   │   │       │   ├── log_train.txt
│   │   │       │   ├── model.pth
│   │   │       │   └── log_test_ood_common.txt
│   │   └── 16shots/               # 使用 use_full_data=0
│   │       └── SimpleCLIP_v2_batch160_ep50/
│   │           └── ...
```

---

## ✅ 验证安装

运行简单测试：
```python
import torch
from model_v2 import SimpleCLIP
import clip

# 检查模型是否正确加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _ = clip.load("ViT-B/32", device)

cfg = {
    'use_rl': False,
    'lambda_sparsity': 0.001,
    'lambda_minimal': 0.01,
    'gate_hidden': 256,
    'templates': ["a photo of a"]
}

classnames = ["dog", "cat", "bird"]
model = SimpleCLIP(cfg, classnames, clip_model)
print("✅ Model loaded successfully!")

# 前向传播测试
x = torch.randn(2, 3, 224, 224).to(device)
logits, logits_local, aux_loss, rl_info = model(x)
print(f"✅ Forward pass successful!")
print(f"   logits shape: {logits.shape}")
print(f"   aux_loss keys: {aux_loss.keys()}")
```

---

## 🤝 版本迁移

如果你已经使用版本 1 的模型权重，不能直接加载到版本 2。
但你可以：

1. **迁移只是门控权重**（如果有）
2. **重新训练版本 2**（推荐，更快）

```python
# 加载旧模型的门控权重
old_state = torch.load('model_v1.pth')
new_model = SimpleCLIP(cfg, classnames, clip_model)

# 手动匹配门控部分
for k, v in old_state.items():
    if 'gating' in k:
        new_model.state_dict()[k] = v

new_model.load_state_dict(new_model.state_dict())
```

---

## 🎓 学习资源

- CLIP 原论文：https://arxiv.org/abs/2103.14030
- 门控机制：Query-Dependent Gate 模式
- RL (REINFORCE)：Policy Gradient 基础概念

