# 📚 DiT (Diffusion Transformer) 代码详解

## 🎯 整体架构

DiT 是一种基于 Transformer 的扩散模型，用于条件图像生成。与 UNet 不同，DiT 将图像切分成 patches 后用 Transformer 处理。

```
输入: 带噪图像 x_t + 时间步 t + 标签 y
         ↓
    [Patchify] 切分成 patches
         ↓
    [Patch Embedding] 映射到高维空间
         ↓
    [Position Encoding] 添加位置信息
         ↓
    [Time + Label Embedding] 条件编码
         ↓
    [DiT Blocks] × N 层 Transformer
         ↓
    [Un-Patchify] 重组回图像
         ↓
输出: 预测的噪音
```

---

## 📁 文件结构

### **核心模块**

| 文件 | 作用 | 关键组件 |
|------|------|---------|
| `dit.py` | DiT 主模型 | Patchify, Embedding, DiT Blocks, Un-patchify |
| `dit_block.py` | DiT Block | AdaLN, Self-Attention, FFN |
| `time_emb.py` | 时间编码 | 正弦位置编码 |
| `diffusion.py` | 扩散过程 | 前向加噪，后向去噪 |
| `dataset.py` | 数据加载 | MNIST 数据集 |
| `config.py` | 配置参数 | 超参数设置 |
| `train.py` | 训练脚本 | 训练循环 |
| `inference.py` | 推理脚本 | 生成图像 |

---

## 🔍 核心概念详解

### **1. Patchify（图像切分）**

```python
# 28×28 图像 → 7×7 个 4×4 patches
Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=4)
```

**过程**：
```
原图 (1, 28, 28)
    ↓ 卷积 (kernel=4, stride=4)
Patches (16, 7, 7)  # 16 = 1 × 4×4
    ↓ permute + view
序列 (49, 16)  # 49 个 patch，每个 16 维
```

---

### **2. AdaLN (Adaptive Layer Normalization)**

DiT 的核心创新！通过条件信息动态调整归一化参数。

**公式**：
```
y = gamma(cond) * LayerNorm(x) + beta(cond)
output = x + alpha(cond) * Module(y)
```

**代码**：
```python
# 从条件生成参数
gamma = self.gamma_linear(cond)  # scale
beta = self.beta_linear(cond)    # shift
alpha = self.alpha_linear(cond)  # gate

# 应用 AdaLN
y = LayerNorm(x)
y = y * (1 + gamma) + beta

# 模块处理（Attention 或 FFN）
y = Module(y)

# 门控残差
output = x + alpha * y
```

**为什么有效？**
- `gamma`, `beta`: 根据时间步和标签调整特征分布
- `alpha`: 控制新信息的注入强度，初始接近 0，模型从恒等映射开始学习

---

### **3. Time Embedding（时间编码）**

使用正弦位置编码将时间步 t ∈ [0, 999] 编码为高维向量。

**公式**：
```
PE(t, 2i)   = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

**特点**：
- 不同频率捕捉不同粒度的时间信息
- 相近时间步有相似的 embedding
- 可学习的 MLP 进一步处理

**代码流程**：
```python
t = 999  # 时间步
    ↓ 频率编码
[sin(999/1), sin(999/10), ..., cos(999/1), cos(999/10), ...]
    ↓ MLP
Time Embedding (64,)
```

---

### **4. Multi-Head Self-Attention**

标准 Transformer 的注意力机制。

**步骤**：
```python
# 1. 生成 Q, K, V
Q = Linear(x)  # (batch, seq_len, nhead*emb_size)
K = Linear(x)
V = Linear(x)

# 2. 拆分成多头
Q = reshape(Q, (batch, nhead, seq_len, emb_size))

# 3. 计算注意力
Attn = softmax(Q @ K^T / sqrt(d))  # (batch, nhead, seq_len, seq_len)

# 4. 应用到 Value
Output = Attn @ V  # (batch, nhead, seq_len, emb_size)

# 5. 合并多头
Output = concat(Output)  # (batch, seq_len, nhead*emb_size)
Output = Linear(Output)  # (batch, seq_len, emb_size)
```

---

### **5. Un-Patchify（重组图像）**

复杂的 reshape 操作，将 patch 序列还原为图像。

**维度变换**：
```
(batch, 49, 16)  # 49 个 patch，每个 16 维
    ↓ view
(batch, 7, 7, 1, 4, 4)  # 7×7 个 patch，每个 1×4×4
    ↓ permute(0,3,1,2,4,5)
(batch, 1, 7, 7, 4, 4)  # channel 在前
    ↓ permute(0,1,2,4,3,5)
(batch, 1, 7, 4, 7, 4)  # 调整 patch 排列
    ↓ reshape
(batch, 1, 28, 28)  # 完整图像！
```

**关键技巧**：
- 先 view 拆分 patch 的内部结构
- 用 permute 调整维度顺序
- 最后 reshape 合并相邻维度

---

## 🔢 Shape 变化全流程

### **以 MNIST 为例（28×28 灰度图）**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x:     (batch, 1, 28, 28)   ← 原始图像
t:     (batch,)              ← 时间步 [0-999]
y:     (batch,)              ← 标签 [0-9]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patchify 阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conv2d:        (batch, 16, 7, 7)
permute:       (batch, 7, 7, 16)
view:          (batch, 49, 16)
patch_emb:     (batch, 49, 64)
+ pos_emb:     (batch, 49, 64)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
条件编码阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
time_emb:      (batch, 64)
label_emb:     (batch, 64)
cond = t + y:  (batch, 64)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DiT Block 处理（×3 层）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入:          (batch, 49, 64)
  ↓ AdaLN + Attention
中间:          (batch, 49, 64)
  ↓ AdaLN + FFN
输出:          (batch, 49, 64)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Un-Patchify 阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LayerNorm:     (batch, 49, 64)
Linear:        (batch, 49, 16)
view:          (batch, 7, 7, 1, 4, 4)
permute:       (batch, 1, 7, 7, 4, 4)
permute:       (batch, 1, 7, 4, 7, 4)
reshape:       (batch, 1, 28, 28)  ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
预测噪音:      (batch, 1, 28, 28)
```

---

## 💡 DiT vs ViT 对比

| 特性 | ViT | DiT |
|------|-----|-----|
| **任务** | 图像分类 | 图像生成 |
| **输入** | 清晰图像 | 带噪图像 + 时间步 + 标签 |
| **CLS Token** | ✅ 有（用于分类） | ❌ 无（所有 patch 都用） |
| **条件信息** | 无 | 时间步 t + 标签 y |
| **归一化** | LayerNorm | AdaLN（自适应） |
| **输出** | 分类 logits | 完整图像 |
| **Un-patchify** | 不需要 | ✅ 需要重组图像 |

---

## 🧪 关键代码片段

### **AdaLN 的核心实现**

```python
# 生成条件参数
gamma = self.gamma(cond)  # (batch, emb_size)
beta = self.beta(cond)    # (batch, emb_size)
alpha = self.alpha(cond)  # (batch, emb_size)

# AdaLN: scale & shift
y = LayerNorm(x)  # (batch, seq_len, emb_size)
y = y * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

# 模块处理（Self-Attention 或 FFN）
y = Module(y)

# AdaLN-Zero: 门控残差
output = x + alpha.unsqueeze(1) * y
```

### **Un-patchify 的完整过程**

```python
# (batch, 49, 16) → (batch, 1, 28, 28)

x = x.view(batch, 7, 7, 1, 4, 4)     # 拆分 patch 结构
x = x.permute(0, 3, 1, 2, 4, 5)      # channel 在前
x = x.permute(0, 1, 2, 4, 3, 5)      # 调整 patch 排列
x = x.reshape(batch, 1, 28, 28)      # 合并维度
```

---

## 📊 模型参数量估算

**以默认配置为例**：
- img_size=28, patch_size=4, emb_size=64, dit_num=3, head=4

```
Patchify:
  - Conv2d: 1×16×4×4 = 256
  - Linear: 16×64 = 1,024
  - pos_emb: 49×64 = 3,136

Conditioning:
  - time_emb MLP: ~12K
  - label_emb: 10×64 = 640

DiT Blocks (×3):
  - AdaLN 参数: 6×(64×64) = 24,576 per block
  - Attention: 3×(64×256) = 49,152 per block
  - FFN: 64×256 + 256×64 = 32,768 per block
  - 每个 Block ≈ 106K
  - 3 个 Block ≈ 318K

Output:
  - LayerNorm: 128
  - Linear: 64×16 = 1,024

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: ~350K 参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🎓 学习建议

### **理解顺序**

1. **第一步**: 理解 `time_emb.py`（最简单）
2. **第二步**: 理解 `dit_block.py`（核心机制）
3. **第三步**: 理解 `dit.py` 的 patchify 和 un-patchify
4. **第四步**: 理解完整的前向传播流程

### **调试技巧**

在关键位置添加 shape 打印：
```python
print(f"After patchify: {x.shape}")
print(f"After DiT blocks: {x.shape}")
print(f"After un-patchify: {x.shape}")
```

### **动手实验**

1. 修改 `patch_size` 为 7，观察 shape 变化
2. 增加 `dit_num` 到 6，观察参数量变化
3. 可视化不同时间步的 time embedding

---

## 🔗 参考资料

- **DiT 论文**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **ViT 论文**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

---

**祝你学习顺利！有问题随时查阅注释或文档** 🎉
