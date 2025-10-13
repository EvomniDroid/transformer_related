# 📚 RDT (Robotics Diffusion Transformer) 代码详解

## 🎯 整体架构

RDT 是一种**专为机器人控制任务设计**的多模态扩散 Transformer 模型。它结合了语言指令、视觉观测和状态信息,生成机器人动作序列。

```
输入: 
  - 带噪动作序列 x_t (horizon 个动作)
  - 时间步 t (扩散时间)
  - 控制频率 freq
  - 当前状态 state
  - 语言指令 lang_c (文本 embedding)
  - 图像观测 img_c (视觉 embedding)
         ↓
    [Time + Freq Embedding] 时间和频率编码
         ↓
    [Position Embedding] 位置编码
         ↓
    [RDT Blocks] × 28 层 交替使用语言和视觉条件
         │
         ├─ 语言条件层: Self-Attn + Cross-Attn(语言) + FFN
         └─ 视觉条件层: Self-Attn + Cross-Attn(视觉) + FFN
         ↓
    [Final Layer] 输出层
         ↓
输出: 预测的噪声或去噪后的动作序列
```

---

## 📁 文件结构

### **核心模块**

| 文件 | 作用 | 关键组件 |
|------|------|---------|
| `models/rdt/model.py` | RDT 主模型 | 位置编码, RDT Blocks 堆叠, 最终输出层 |
| `models/rdt/blocks.py` | RDT Block 和组件 | TimestepEmbedder, CrossAttention, RDTBlock, FinalLayer |
| `models/multimodal_encoder/` | 多模态编码器 | 语言编码器(T5/CLIP), 视觉编码器(ResNet/ViT) |
| `models/rdt_runner.py` | 扩散采样器 | DDPM, DDIM 采样算法 |
| `models/ema_model.py` | EMA 模型 | 指数移动平均,稳定训练 |
| `train/train.py` | 训练脚本 | 多模态数据加载, 训练循环 |

---

## 🔍 核心概念详解

### **1. 多模态条件输入**

RDT 处理**三种**条件信息,这是与 DiT 最大的区别!

```python
输入维度说明:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
主序列 x: (B, T+1, D)
  - T = horizon (动作序列长度, 如 32 步)
  - T+1 包含: [timestep, freq, state, action_1, ..., action_T]
  - D = hidden_size (如 1152)

语言条件 lang_c: (B, L_lang, D)
  - L_lang ≤ 1024 (语言 token 数量,可变长度)
  - 例如: "拿起桌上的红色杯子"

图像条件 img_c: (B, L_img, D)
  - L_img = 4096 (图像 token 数量,固定长度)
  - 例如: 从 ResNet 提取的视觉特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**为什么需要多模态?**
- **语言**: 理解任务目标 ("拿起杯子")
- **视觉**: 感知环境状态 (杯子在哪里?)
- **状态**: 机器人当前姿态 (关节角度, 位置等)

---

### **2. 交替的跨模态注意力机制**

RDT 的创新之处:**奇数层用语言条件,偶数层用视觉条件**

```python
# 28 层 RDT Blocks 交替使用条件
for i, block in enumerate(self.blocks):
    if i % 2 == 0:
        c = lang_c      # 偶数层: 语言条件
        mask = lang_mask
    else:
        c = img_c       # 奇数层: 视觉条件
        mask = img_mask
    x = block(x, c, mask)
```

**为什么要交替?**
1. **信息融合**: 语言和视觉逐层交替融入,充分结合
2. **计算效率**: 不需要每层同时处理两种条件
3. **梯度流动**: 每种条件都有足够的更新机会

**具体流程:**
```
输入 x: [timestep, freq, state, action_1, ..., action_32]

Layer 0:  Self-Attn → Cross-Attn(语言) → FFN
          ↓ 理解任务指令
Layer 1:  Self-Attn → Cross-Attn(视觉) → FFN
          ↓ 感知视觉环境
Layer 2:  Self-Attn → Cross-Attn(语言) → FFN
          ↓ 再次关注语言细节
Layer 3:  Self-Attn → Cross-Attn(视觉) → FFN
          ↓ 再次关注视觉细节
...
Layer 27: Self-Attn → Cross-Attn(视觉) → FFN

输出: 去噪后的动作序列
```

---

### **3. RDT Block 结构**

每个 RDT Block 包含**三个**子模块:

```python
class RDTBlock:
    def forward(self, x, c, mask):
        # 1️⃣ Self-Attention: 动作序列内部的关系
        x = x + Attention(RmsNorm(x))
        
        # 2️⃣ Cross-Attention: 动作序列 attend to 条件 (语言或视觉)
        x = x + CrossAttention(RmsNorm(x), c, mask)
        
        # 3️⃣ FFN: 非线性变换
        x = x + FFN(RmsNorm(x))
        
        return x
```

**与 DiT Block 的对比:**

| 特性 | DiT Block | RDT Block |
|------|-----------|-----------|
| **归一化** | LayerNorm + AdaLN | RmsNorm (更稳定) |
| **条件机制** | AdaLN (调制归一化) | Cross-Attention (显式注意力) |
| **条件类型** | 单一条件 (时间+标签) | 双条件交替 (语言/视觉) |
| **残差连接** | AdaLN-Zero (门控) | 直接相加 |
| **FFN 激活** | GELU | GELU(tanh 近似) |

---

### **4. Cross-Attention 详解**

Cross-Attention 是 RDT 的核心机制,用于注入条件信息。

**计算流程:**
```python
# 输入:
#   x: (B, N, D) - 动作序列 (Query)
#   c: (B, L, D) - 条件序列 (Key, Value)
#   mask: (B, L) - 有效 token 掩码

# 1. 生成 Q, K, V
Q = Linear_q(x)    # (B, N, D) - 从动作序列生成
K = Linear_k(c)    # (B, L, D) - 从条件生成
V = Linear_v(c)    # (B, L, D) - 从条件生成

# 2. 拆分成多头
Q = Q.reshape(B, num_heads, N, head_dim)
K = K.reshape(B, num_heads, L, head_dim)
V = V.reshape(B, num_heads, L, head_dim)

# 3. 计算注意力分数
Attn = softmax(Q @ K^T / sqrt(head_dim))  # (B, num_heads, N, L)
# ↑ 每个动作 token 对所有条件 token 的注意力权重

# 4. 应用掩码 (屏蔽无效的条件 token)
if mask is not None:
    Attn = Attn.masked_fill(~mask, -inf)

# 5. 加权求和
Output = Attn @ V  # (B, num_heads, N, head_dim)

# 6. 合并多头
Output = Output.reshape(B, N, D)
```

**掩码的作用:**
```python
# 语言条件可能是变长的
lang_c = [
    "pick up the red cup",        # 5 个 token
    "grasp the object",           # 3 个 token (padding 到 1024)
]
lang_mask = [
    [1,1,1,1,1, 0,0,0,...,0],    # True 表示有效, False 表示 padding
    [1,1,1, 0,0,0,...,0],
]

# Cross-Attention 时只关注有效的 token,忽略 padding
```

---

### **5. Position Embedding (位置编码)**

RDT 使用**三套**独立的位置编码:

```python
# 1️⃣ 主序列位置编码 (动作序列)
self.x_pos_embed: (1, horizon+3, hidden_size)
# horizon+3 = [timestep, freq, state] + horizon 个动作

# 2️⃣ 语言条件位置编码
self.lang_cond_pos_embed: (1, 1024, hidden_size)
# 支持最多 1024 个语言 token

# 3️⃣ 视觉条件位置编码
self.img_cond_pos_embed: (1, 4096, hidden_size)
# 4096 个视觉 token (例如 64x64 的 feature map)
```

**多模态位置编码的设计:**
```python
# 主序列使用分段编码
x_pos_embed = get_multimodal_cond_pos_embed(
    embed_dim=1152,
    mm_cond_lens=OrderedDict([
        ('timestep', 1),   # 时间步占 1 个位置
        ('ctrl_freq', 1),  # 控制频率占 1 个位置
        ('state', 1),      # 状态占 1 个位置
        ('action', 32),    # 动作占 32 个位置
    ])
)

# 不同模态有不同的频率编码,帮助模型区分
```

---

### **6. Timestep Embedding (时间编码)**

与 DiT 类似,使用正弦编码:

```python
class TimestepEmbedder:
    def timestep_embedding(self, t, dim=256):
        # t: (B,) 时间步 [0, 999]
        half = dim // 2
        freqs = exp(-log(10000) * arange(0, half) / half)
        # freqs: [1, 0.87, 0.76, ..., 0.001]
        
        args = t[:, None] * freqs[None, :]
        embedding = concat([cos(args), sin(args)])  # (B, 256)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, 256)  # (B, 256)
        t_emb = MLP(t_freq)                       # (B, 1152)
        return t_emb
```

**RDT 有两个时间编码器:**
1. `t_embedder`: 扩散时间步 (0-999)
2. `freq_embedder`: 控制频率 (例如 10Hz, 20Hz)

---

### **7. Final Layer (输出层)**

```python
class FinalLayer:
    def forward(self, x):
        # x: (B, horizon+3, hidden_size)
        x = RmsNorm(x)                    # 归一化
        x = FFN(x)                        # 映射到输出维度
        # x: (B, horizon+3, output_dim)
        return x

# 在主模型中:
x = self.final_layer(x)        # (B, horizon+3, output_dim)
x = x[:, -self.horizon:]       # 只保留动作部分
                               # (B, horizon, output_dim)
```

**输出维度:**
- `output_dim = 128`: 机器人动作的维度
  - 例如: 7 个关节角度 + 1 个夹爪 = 8 维
  - 如果预测 16 步,则 8 × 16 = 128

---

## 🔢 Shape 变化全流程

### **以 RDT-1B 为例**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x:           (B, 33, 1152)      ← state + 32 个动作
t:           (B,)               ← 扩散时间步 [0, 999]
freq:        (B,)               ← 控制频率 [1, 50]
lang_c:      (B, L_lang, 1152)  ← 语言条件 (变长)
img_c:       (B, 4096, 1152)    ← 视觉条件 (固定长度)
lang_mask:   (B, L_lang)        ← 语言掩码
img_mask:    (B, 4096)          ← 视觉掩码

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Embedding 阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t_emb:       (B, 1152)
             ↓ unsqueeze
             (B, 1, 1152)

freq_emb:    (B, 1152)
             ↓ unsqueeze
             (B, 1, 1152)

x:           (B, 33, 1152)
             ↓ concat([t_emb, freq_emb, x])
             (B, 35, 1152)  ← [t, freq, state, action_1, ..., action_32]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Position Embedding 阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x:           (B, 35, 1152)
+ x_pos_emb: (1, 35, 1152)     ← 广播加法
→            (B, 35, 1152)

lang_c:      (B, L_lang, 1152)
+ lang_pos:  (1, L_lang, 1152) ← 只取前 L_lang 个位置
→            (B, L_lang, 1152)

img_c:       (B, 4096, 1152)
+ img_pos:   (1, 4096, 1152)
→            (B, 4096, 1152)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RDT Blocks 处理 (× 28 层)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 0:  (偶数层 - 语言条件)
  输入 x:        (B, 35, 1152)
  条件 lang_c:   (B, L_lang, 1152)
  
  ↓ Self-Attention
  Q,K,V from x:  (B, 16, 35, 72)      ← 16 heads, 72 = 1152/16
  Attn:          (B, 16, 35, 35)
  Output:        (B, 35, 1152)
  
  ↓ Cross-Attention (语言)
  Q from x:      (B, 16, 35, 72)
  K,V from lang: (B, 16, L_lang, 72)
  Attn:          (B, 16, 35, L_lang)  ← 35 个 query, L_lang 个 key
  Output:        (B, 35, 1152)
  
  ↓ FFN
  Input:         (B, 35, 1152)
  Hidden:        (B, 35, 1152)        ← FFN hidden = input
  Output:        (B, 35, 1152)

Layer 1:  (奇数层 - 视觉条件)
  输入 x:        (B, 35, 1152)
  条件 img_c:    (B, 4096, 1152)
  
  ↓ Self-Attention
  (同上)
  
  ↓ Cross-Attention (视觉)
  Q from x:      (B, 16, 35, 72)
  K,V from img:  (B, 16, 4096, 72)
  Attn:          (B, 16, 35, 4096)    ← 35 个 query, 4096 个 key
  Output:        (B, 35, 1152)
  
  ↓ FFN
  Output:        (B, 35, 1152)

... (重复 28 层)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final Layer 阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x:             (B, 35, 1152)
↓ RmsNorm
               (B, 35, 1152)
↓ FFN
               (B, 35, 128)           ← 映射到输出维度

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x[:, -32:]     (B, 32, 128)          ← 只保留动作部分
                                      ✅ 预测的去噪动作序列
```

---

## 💡 RDT vs DiT 对比

| 特性 | DiT | RDT |
|------|-----|-----|
| **任务** | 图像生成 | 机器人控制 |
| **输入数据** | 图像 | 动作序列 + 状态 |
| **条件信息** | 时间步 + 类别标签 | 时间步 + 频率 + 语言 + 视觉 |
| **Patchify** | ✅ 卷积切分图像 | ❌ 直接输入 token 序列 |
| **Un-patchify** | ✅ 重组图像 | ❌ 直接输出序列 |
| **Block 结构** | Self-Attn + FFN | Self-Attn + Cross-Attn + FFN |
| **条件机制** | AdaLN (调制) | Cross-Attention (显式) |
| **归一化** | LayerNorm | RmsNorm |
| **多模态** | ❌ 单模态 | ✅ 语言 + 视觉交替 |
| **序列长度** | 固定 (49 patches) | 可变 (horizon 可调) |
| **模型规模** | ~350K | ~1B (RDT-1B) |

---

## 🧪 关键代码片段

### **Cross-Attention 核心实现**

```python
class CrossAttention:
    def forward(self, x, c, mask=None):
        # x: (B, N, D) 动作序列
        # c: (B, L, D) 条件 (语言或视觉)
        # mask: (B, L) 有效 token 掩码
        
        B, N, D = x.shape
        L = c.shape[1]
        
        # 生成 Q, K, V
        Q = self.q(x).reshape(B, N, num_heads, head_dim)
                    .permute(0, 2, 1, 3)  # (B, h, N, d)
        
        KV = self.kv(c).reshape(B, L, 2, num_heads, head_dim)
                       .permute(2, 0, 3, 1, 4)  # (2, B, h, L, d)
        K, V = KV[0], KV[1]  # (B, h, L, d)
        
        # 应用归一化
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # 扩展掩码
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
                      .expand(-1, -1, N, -1)  # (B, 1, N, L)
        
        # Flash Attention
        x = F.scaled_dot_product_attention(
            query=Q,     # (B, h, N, d)
            key=K,       # (B, h, L, d)
            value=V,     # (B, h, L, d)
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.
        )  # (B, h, N, d)
        
        # 重组
        x = x.permute(0, 2, 1, 3).reshape(B, N, D)
        x = self.proj(x)
        return x
```

---

### **RDT Block 的完整流程**

```python
class RDTBlock:
    def forward(self, x, c, mask):
        # 1. Self-Attention (动作序列内部交互)
        origin_x = x
        x = self.norm1(x)              # RmsNorm
        x = self.attn(x)               # Self-Attention
        x = x + origin_x               # 残差连接
        
        # 2. Cross-Attention (注入条件信息)
        origin_x = x
        x = self.norm2(x)              # RmsNorm
        x = self.cross_attn(x, c, mask)  # Cross-Attention
        x = x + origin_x               # 残差连接
        
        # 3. FFN (非线性变换)
        origin_x = x
        x = self.norm3(x)              # RmsNorm
        x = self.ffn(x)                # Feed-Forward
        x = x + origin_x               # 残差连接
        
        return x
```

---

### **主模型前向传播**

```python
class RDT:
    def forward(self, x, freq, t, lang_c, img_c, lang_mask, img_mask):
        # 1. 时间和频率编码
        t_emb = self.t_embedder(t).unsqueeze(1)       # (B, 1, D)
        freq_emb = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        
        # 2. 拼接到主序列
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(x.shape[0], -1, -1)
        x = torch.cat([t_emb, freq_emb, x], dim=1)    # (B, T+2, D)
        
        # 3. 添加位置编码
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed
        
        # 4. 通过 RDT Blocks (交替条件)
        for i, block in enumerate(self.blocks):
            # 偶数层用语言,奇数层用视觉
            c = lang_c if i % 2 == 0 else img_c
            mask = lang_mask if i % 2 == 0 else img_mask
            x = block(x, c, mask)
        
        # 5. 输出层
        x = self.final_layer(x)                       # (B, T+2, out_dim)
        
        # 6. 只保留动作部分
        x = x[:, -self.horizon:]                      # (B, horizon, out_dim)
        
        return x
```

---

## 📊 模型参数量估算

**RDT-1B 配置**:
- hidden_size=1152, depth=28, num_heads=16
- horizon=32, output_dim=128

```
Time Embedders (2 个):
  - 每个 MLP: 256×1152 + 1152×1152 ≈ 1.6M
  - 总计: 3.2M

Position Embeddings:
  - x_pos_embed: 35×1152 = 40K
  - lang_pos_embed: 1024×1152 = 1.2M
  - img_pos_embed: 4096×1152 = 4.7M
  - 总计: 6M

RDT Blocks (×28):
  每个 Block:
    - Self-Attention:
        - QKV Linear: 3×(1152×1152) = 4.0M
        - Proj Linear: 1152×1152 = 1.3M
    - Cross-Attention:
        - Q Linear: 1152×1152 = 1.3M
        - KV Linear: 1152×(2×1152) = 2.7M
        - Proj Linear: 1152×1152 = 1.3M
    - FFN:
        - FC1: 1152×1152 = 1.3M
        - FC2: 1152×1152 = 1.3M
    - RmsNorm (3 个): 忽略不计
    
    每个 Block ≈ 13.2M
    28 个 Block ≈ 370M

Final Layer:
  - FFN: 1152×1152 + 1152×128 ≈ 1.5M

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: ~380M 参数 (不含多模态编码器)

如果加上:
  - 语言编码器 (T5-XL): ~3B 参数
  - 视觉编码器 (ResNet-50): ~25M 参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
完整 RDT-1B: ~3.4B 参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🎓 学习建议

### **理解顺序**

1. **第一步**: 理解 `TimestepEmbedder`（时间编码,最简单）
2. **第二步**: 理解 `CrossAttention`（核心机制）
3. **第三步**: 理解 `RDTBlock`（如何组合 Self-Attn + Cross-Attn）
4. **第四步**: 理解 `RDT` 主模型（交替条件的设计）
5. **第五步**: 理解训练脚本（多模态数据加载）

### **调试技巧**

在关键位置添加 shape 打印:
```python
print(f"After time embedding: {t_emb.shape}")
print(f"After concat: {x.shape}")
print(f"Lang condition: {lang_c.shape}, mask: {lang_mask.shape}")
print(f"After block {i}: {x.shape}")
print(f"Final output: {x.shape}")
```

### **动手实验**

1. **修改 horizon**: 从 32 改到 16,观察 shape 变化
2. **可视化注意力**: 打印 Cross-Attention 的 attention map,看模型关注哪些语言/视觉 token
3. **消融实验**: 只用语言条件或只用视觉条件,对比性能
4. **条件掩码**: 实验不同的 mask 策略

### **常见问题**

**Q1: 为什么交替使用语言和视觉条件?**
- A: 充分融合两种信息,每层都能访问一种条件,避免信息瓶颈

**Q2: RmsNorm vs LayerNorm 有什么区别?**
- A: RmsNorm 不减去均值,只做尺度归一化,训练更稳定,速度更快

**Q3: 为什么最后只保留动作部分?**
- A: 输入的 timestep, freq, state 只是辅助信息,我们只需要预测/去噪动作序列

**Q4: Flash Attention 是什么?**
- A: 高效的注意力实现,减少显存占用,加速计算,PyTorch 2.0+ 内置

---

## 🔗 参考资料

- **RDT 论文**: [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)
- **DiT 论文**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Flash Attention**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

---

## 📌 关键术语对照表

| 英文 | 中文 | 说明 |
|------|------|------|
| Horizon | 预测步数 | 动作序列的长度 (如 32 步) |
| Control Frequency | 控制频率 | 机器人执行动作的频率 (如 10Hz) |
| Cross-Attention | 跨注意力 | Query 和 Key 来自不同序列 |
| Self-Attention | 自注意力 | Query 和 Key 来自同一序列 |
| RmsNorm | 均方根归一化 | Root Mean Square Layer Normalization |
| EMA | 指数移动平均 | Exponential Moving Average |
| Multimodal | 多模态 | 语言 + 视觉 + 其他模态 |
| Token | 令牌 | 序列中的基本单位 |
| Embedding | 嵌入 | 将离散符号映射到连续向量 |

---

**祝你学习顺利!机器人控制和多模态学习是未来的重要方向** 🤖🎉
