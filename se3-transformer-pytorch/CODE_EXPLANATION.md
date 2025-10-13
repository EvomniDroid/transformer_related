# 📚 SE3-Transformer 代码详解

## 🎯 整体架构

SE3-Transformer 是一种**几何等变(Equivariant)** Transformer,专门设计用于处理 3D 点云和分子结构数据。关键特性是保持 **SE(3) 等变性** - 即对旋转和平移操作保持不变性。

```
输入: 
  - 特征 feats (B, N, dim)
  - 3D 坐标 coors (B, N, 3)
  - 掩码 mask (B, N)
         ↓
    [Token Embedding] (可选)
         ↓
    [邻居计算] KNN 或稀疏邻接
         ↓
    [球谐基计算] Spherical Harmonics Basis
         ↓
    [ConvSE3 输入层] Type-0 特征卷积
         ↓
    [SE3 Transformer Blocks] × depth 层
         │
         ├─ [AttentionSE3] SE(3) 等变注意力
         │   ├─ Type-0 特征 (标量)
         │   ├─ Type-1 特征 (向量)
         │   └─ Type-n 特征 (高阶张量)
         │
         └─ [FeedForwardSE3] SE(3) 等变前馈网络
         ↓
    [ConvSE3 输出层] (可选)
         ↓
    [Norm + Reduce] (可选)
         ↓
输出: 多类型特征字典 {'0': 标量, '1': 向量, ...}
```

---

## 📁 文件结构

### **核心模块**

| 文件 | 作用 | 关键组件 |
|------|------|---------|
| `se3_transformer_pytorch.py` | 主模型 | SE3Transformer, AttentionSE3, ConvSE3 |
| `basis.py` | 球谐基函数 | Wigner-D 矩阵, 旋转不变基 |
| `spherical_harmonics.py` | 球谐函数 | 实数球谐 Y_lm |
| `irr_repr.py` | 不可约表示 | SO(3) 群表示 |
| `rotary.py` | 旋转位置编码 | Rotary Position Embedding |
| `reversible.py` | 可逆 Transformer | 节省显存的可逆网络 |
| `utils.py` | 工具函数 | 批量索引, 掩码, 傅里叶编码 |

---

## 🔍 核心概念详解

### **1. SE(3) 群和等变性**

**SE(3)** = 特殊欧几里得群 = 3D 旋转 + 平移

```python
什么是等变性 (Equivariance)?

输入:
  x: 3D 点云坐标 (N, 3)
  f: 特征 (N, D)

操作:
  R: 旋转矩阵 (3, 3)
  t: 平移向量 (3,)
  
等变性质:
  Model(R·x + t, f) = R·Model(x, f) + t
  
意义: 
  无论如何旋转/平移输入,输出也会相应地旋转/平移
  这对分子、蛋白质等 3D 结构非常重要!
```

**为什么重要?**
- **物理一致性**: 分子的性质不应该因为观察角度改变
- **数据增强**: 不需要旋转增强,模型天然具有旋转不变性
- **泛化能力**: 更好地泛化到新的空间配置

---

### **2. Fiber (纤维) - 多类型特征表示**

Fiber 是 SE3-Transformer 的核心数据结构,表示**不同类型(type)**的特征。

```python
Type-0 特征: 标量 (Scalar)
  - 形状: (B, N, dim, 1)
  - 例如: 原子电荷, 能量
  - 旋转不变 ✓

Type-1 特征: 向量 (Vector)
  - 形状: (B, N, dim, 3)
  - 例如: 力, 速度, 偶极矩
  - 随旋转而旋转 (等变)

Type-2 特征: 2阶张量
  - 形状: (B, N, dim, 5)
  - 例如: 四极矩, 应力张量
  - 更复杂的等变性

Type-n 特征: n阶张量
  - 形状: (B, N, dim, 2n+1)
  - 球谐函数 Y_lm, l=0,1,2,...
```

**Fiber 结构示例:**
```python
fiber = Fiber([
    FiberEl(degrees=0, dim=64),   # Type-0: 64 个标量通道
    FiberEl(degrees=1, dim=32),   # Type-1: 32 个向量通道
    FiberEl(degrees=2, dim=16)    # Type-2: 16 个2阶张量通道
])

# 实际数据
features = {
    '0': torch.randn(B, N, 64, 1),    # Type-0
    '1': torch.randn(B, N, 32, 3),    # Type-1
    '2': torch.randn(B, N, 16, 5)     # Type-2
}
```

---

### **3. 球谐基函数 (Spherical Harmonics Basis)**

球谐基是 SE(3) 等变的数学基础。

**数学原理:**
```
相对位置向量: r_ij = x_j - x_i  (3D 向量)

球坐标表示:
  - 距离: d_ij = ||r_ij||
  - 方向: (θ, φ) 极角和方位角

球谐函数: Y_l^m(θ, φ)
  - l: 度数 (degree), l = 0, 1, 2, ...
  - m: 阶数 (order), m = -l, ..., 0, ..., +l
  - 共 2l+1 个独立分量

性质:
  1. 正交性: ∫ Y_l^m · Y_l'^m' dΩ = δ_ll' δ_mm'
  2. 旋转等变: R·Y_l^m = Σ D_l^mm'(R) Y_l^m'
```

**代码实现:**
```python
# 计算球谐基
basis = get_basis(
    rel_pos,           # (B, N, K, 3) 相对位置
    max_degree=3,      # 最大度数
    differentiable=True
)

# 返回字典
basis = {
    '(0,0)': Tensor(B, N, K, 1, 1, 1),      # Type-0 → Type-0
    '(1,0)': Tensor(B, N, K, 3, 1, 3),      # Type-1 → Type-0
    '(1,1)': Tensor(B, N, K, 3, 3, 3),      # Type-1 → Type-1
    '(1,2)': Tensor(B, N, K, 5, 3, 5),      # Type-1 → Type-2
    ...
}
```

**物理意义:**
- **Type-0 → Type-0**: 标量与标量的相互作用 (如电荷)
- **Type-1 → Type-1**: 向量与向量的相互作用 (如偶极-偶极)
- **Type-0 → Type-1**: 标量生成向量 (如梯度)

---

### **4. ConvSE3 - SE(3) 等变卷积**

ConvSE3 是 SE3-Transformer 的基础层,类似于 GNN 中的消息传递。

**核心思想:**
```python
对于节点 i:
  1. 收集邻居信息: 
     neighbors = {j : j ∈ Neighbors(i)}
  
  2. 计算相对位置:
     r_ij = x_j - x_i
  
  3. 通过球谐基转换特征类型:
     f_j^(type_out) = Σ Kernel(r_ij, edges) ⊗ Basis^(type_in → type_out) ⊗ f_j^(type_in)
  
  4. 聚合邻居:
     f_i^new = Σ_j f_j^transformed
  
  5. 自交互 (可选):
     f_i^out = f_i^new + Self_Interact(f_i^old)
```

**维度变换示例:**
```
输入 Type-1 特征: (B, N, 32, 3)
输出 Type-2 特征: (B, N, 16, 5)

步骤:
1. 邻居索引:      (B, N, 32, 3) → (B, N, K, 32, 3)
2. 径向函数:      Kernel(r_ij) → (B, N, K, 16, 32, 3)
3. 球谐基:        Basis^(1→2) → (B, N, K, 5, 3, 3)
4. Einstein 求和: (B, N, K, 16, 32, 3) ⊗ (B, N, K, 5, 3, 3) → (B, N, K, 16, 5)
5. 聚合:          mean(dim=K) → (B, N, 16, 5)
```

**关键代码:**
```python
class ConvSE3:
    def forward(self, inp, edge_info, rel_dist, basis):
        neighbor_indices, neighbor_masks, edges = edge_info
        outputs = {}
        
        for degree_out in fiber_out.degrees:
            output = 0
            
            for degree_in in fiber_in.degrees:
                # 1. 获取邻居特征
                x = inp[str(degree_in)]
                x = batched_index_select(x, neighbor_indices, dim=1)
                
                # 2. 计算核函数
                kernel = self.kernel_unary(edges, rel_dist, basis)
                
                # 3. 应用核函数和基
                output += einsum('... o i, ... i c -> ... o c', kernel, x)
            
            # 4. 聚合
            if self.pool:
                output = masked_mean(output, neighbor_masks, dim=2)
            
            outputs[str(degree_out)] = output
        
        # 5. 自交互
        if self.self_interaction:
            outputs = self.self_interact_sum(
                outputs, 
                self.self_interact(inp)
            )
        
        return outputs
```

---

### **5. AttentionSE3 - SE(3) 等变注意力**

结合了 Transformer 注意力和 SE(3) 等变性。

**架构:**
```python
Q (Query):  从自身特征生成 (LinearSE3)
K (Key):    从邻居特征生成 (ConvSE3 或 LinearSE3)
V (Value):  从邻居特征生成 (ConvSE3)

注意力计算:
  Attn = softmax(Q·K^T / sqrt(d))
  Output = Attn @ V
```

**多类型注意力:**
```python
# 对每种类型分别计算注意力
for degree in ['0', '1', '2']:
    q = queries[degree]      # (B, h, N, d, m)
    k = keys[degree]         # (B, h, N, K, d, m)
    v = values[degree]       # (B, h, N, K, d, m)
    
    # Einstein 求和: 在 d 和 m 维度上点积
    sim = einsum('b h i d m, b h i j d m -> b h i j', q, k)
    sim = sim * scale
    
    # 应用掩码
    if exists(mask):
        sim = sim.masked_fill(~mask, -inf)
    
    # Softmax
    attn = sim.softmax(dim=-1)  # (B, h, N, K)
    
    # 加权求和
    out = einsum('b h i j, b h i j d m -> b h i d m', attn, v)
    
    outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')
```

**特殊设计:**
1. **attend_self**: 是否允许节点关注自己
2. **use_null_kv**: 添加可学习的 null token
3. **global_feats**: 全局特征注入 (用于条件生成)
4. **rotary_pos_emb**: 旋转位置编码

---

### **6. RadialFunc - 径向函数**

学习边特征到核权重的映射。

```python
class RadialFunc:
    def __init__(self, num_freq, in_dim, out_dim, edge_dim):
        self.net = MLP([
            Linear(edge_dim + 1, 128),      # edge + distance
            LayerNorm(128),
            GELU(),
            Linear(128, 128),
            LayerNorm(128),
            GELU(),
            Linear(128, num_freq * in_dim * out_dim)
        ])
    
    def forward(self, x):
        # x: (B, N, K, edge_dim+1)
        y = self.net(x)
        # 重排为核形状
        return rearrange(y, '... (o i f) -> ... o () i () f', 
                        i=in_dim, o=out_dim)
```

**作用:**
- 根据距离和边特征生成自适应的卷积核
- 允许不同距离/边类型有不同的交互强度

---

### **7. NormSE3 - SE(3) 等变归一化**

保持等变性的归一化和非线性激活。

**核心思想:**
```python
对于 Type-l 特征 t:
  1. 计算范数: norm = ||t||  (标量,旋转不变)
  2. 归一化方向: phase = t / norm  (等变)
  3. 非线性变换: norm' = NonLin(norm)  (标量)
  4. 重组: output = norm' * phase  (等变)
```

**代码实现:**
```python
class NormSE3:
    def forward(self, features):
        output = {}
        for degree, t in features.items():
            # 1. 计算范数 (旋转不变)
            norm = t.norm(dim=-1, keepdim=True)  # (B, N, D, 1)
            
            # 2. 归一化方向 (等变)
            phase = t / (norm + eps)              # (B, N, D, m)
            
            # 3. 门控或缩放 (标量操作)
            norm_flat = rearrange(norm, '... () -> ...')
            if gate_weights is not None:
                scale = einsum('b n d, d e -> b n e', norm_flat, gate_weights)
            transformed = self.nonlin(norm_flat * scale)
            
            # 4. 重组 (保持等变性)
            transformed = rearrange(transformed, '... -> ... ()')
            output[degree] = transformed * phase
        
        return output
```

**为什么有效?**
- **范数是标量**: 旋转不变,可以安全地应用非线性
- **方向保持**: phase 是等变的,输出也是等变的

---

### **8. 邻居选择策略**

SE3-Transformer 支持多种邻居选择方式:

```python
1️⃣ K近邻 (KNN):
   - 选择距离最近的 K 个节点
   - neighbors = 32
   
2️⃣ 半径邻居:
   - 选择距离 < valid_radius 的所有节点
   - valid_radius = 10.0
   
3️⃣ 稀疏邻接 (Sparse Adjacency):
   - 基于化学键或预定义的连接
   - adj_mat: (N, N) 邻接矩阵
   - attend_sparse_neighbors = True
   
4️⃣ 混合策略:
   - 稀疏邻居 (化学键) + K 近邻
   - 稀疏邻居优先级更高
```

**代码示例:**
```python
# 计算距离矩阵
rel_pos = coors[:, :, None] - coors[:, None, :]  # (B, N, N, 3)
rel_dist = rel_pos.norm(dim=-1)                  # (B, N, N)

# 排除自身
exclude_self_mask = ~torch.eye(N, dtype=bool)
rel_dist_masked = rel_dist.masked_select(exclude_self_mask)

# 如果有稀疏邻接,优先级设为 0
if exists(sparse_neighbor_mask):
    rel_dist_masked = rel_dist_masked.masked_fill(
        sparse_neighbor_mask, 0.
    )

# 选择 top-K 近邻
dist_values, nearest_indices = rel_dist_masked.topk(
    K, dim=-1, largest=False
)

# 半径过滤
neighbor_mask = dist_values <= valid_radius
```

---

## 🔢 Shape 变化全流程

### **以 AlphaFold2 蛋白质结构预测为例**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
atom_feats:    (2, 256, 64)         ← 256 个原子,64 维特征
coors:         (2, 256, 3)          ← 256 个原子的 3D 坐标
mask:          (2, 256)             ← 有效原子掩码

模型配置:
- dim = 64
- depth = 2
- input_degrees = 1   (输入 Type-0)
- num_degrees = 2     (使用 Type-0, Type-1)
- output_degrees = 2  (输出 Type-1 向量)
- neighbors = 32

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
邻居计算阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rel_pos:           (2, 256, 256, 3)     ← 相对位置向量
rel_dist:          (2, 256, 256)        ← 相对距离

排除自身:
rel_pos:           (2, 256, 255, 3)
rel_dist:          (2, 256, 255)

KNN 选择:
neighbor_indices:  (2, 256, 32)         ← 每个原子的 32 个近邻索引
neighbor_rel_pos:  (2, 256, 32, 3)      ← 近邻相对位置
neighbor_rel_dist: (2, 256, 32)         ← 近邻距离
neighbor_mask:     (2, 256, 32)         ← 有效近邻掩码

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
球谐基计算阶段
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
basis = get_basis(neighbor_rel_pos, max_degree=1)

basis['0,0']:      (2, 256, 32, 1, 1, 1)    ← Type-0 → Type-0
basis['1,0']:      (2, 256, 32, 3, 1, 3)    ← Type-1 → Type-0
basis['0,1']:      (2, 256, 32, 1, 3, 3)    ← Type-0 → Type-1
basis['1,1']:      (2, 256, 32, 3, 3, 3)    ← Type-1 → Type-1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入 Embedding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
初始特征 (Type-0):
features['0']:     (2, 256, 64, 1)          ← 标量特征

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ConvSE3 输入层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fiber_in:  {0: 64}
fiber_out: {0: 64, 1: 64}

输出:
features['0']:     (2, 256, 64, 1)          ← 标量
features['1']:     (2, 256, 64, 3)          ← 向量

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1: AttentionSE3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入:  features (Type-0: (2,256,64,1), Type-1: (2,256,64,3))

Query 生成 (LinearSE3):
q['0']:            (2, 256, 512, 1)         ← 64 → 512 (8 heads × 64)
q['1']:            (2, 256, 512, 3)

Key 生成 (ConvSE3):
k['0']:            (2, 256, 32, 512, 1)     ← 邻居维度
k['1']:            (2, 256, 32, 512, 3)

Value 生成 (ConvSE3):
v['0']:            (2, 256, 32, 512, 1)
v['1']:            (2, 256, 32, 512, 3)

重排多头:
q['0']:            (2, 8, 256, 64, 1)       ← (B, h, N, d, m)
k['0']:            (2, 8, 256, 32, 64, 1)   ← (B, h, N, K, d, m)
v['0']:            (2, 8, 256, 32, 64, 1)

注意力计算 (Type-0):
sim = einsum('bhidm, bhijdm -> bhij', q['0'], k['0'])
sim:               (2, 8, 256, 32)          ← 注意力分数
attn = softmax(sim, dim=-1)
out = einsum('bhij, bhijdm -> bhidm', attn, v['0'])
out['0']:          (2, 8, 256, 64, 1)

注意力计算 (Type-1):
(同上,维度相同但 m=3)
out['1']:          (2, 8, 256, 64, 3)

合并多头:
output['0']:       (2, 256, 512, 1)
output['1']:       (2, 256, 512, 3)

输出投影 (LinearSE3):
output['0']:       (2, 256, 64, 1)
output['1']:       (2, 256, 64, 3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1: FeedForwardSE3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入:  (Type-0: (2,256,64,1), Type-1: (2,256,64,3))

Project In (mult=4):
hidden['0']:       (2, 256, 256, 1)         ← 64 × 4
hidden['1']:       (2, 256, 256, 3)

NormSE3 非线性:
(对每个 type 分别计算范数和方向)

Project Out:
output['0']:       (2, 256, 64, 1)
output['1']:       (2, 256, 64, 3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 2: (重复 Layer 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(同上)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ConvSE3 输出层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出 Type-1 (向量):
refined_coors:     (2, 256, 3)              ← 坐标修正向量

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最终坐标更新
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
new_coors = coors + refined_coors
            (2, 256, 3) + (2, 256, 3)
          = (2, 256, 3)                     ✅ 精炼后的坐标
```

---

## 💡 SE3-Transformer vs DiT vs RDT 对比

| 特性 | DiT | RDT | SE3-Transformer |
|------|-----|-----|-----------------|
| **任务** | 图像生成 | 机器人控制 | 3D 分子/蛋白质 |
| **输入数据** | 2D 图像 | 动作序列 | 3D 点云 + 坐标 |
| **核心对称性** | 无 | 无 | **SE(3) 等变** |
| **特征类型** | 单一 (标量) | 单一 (标量) | **多类型 (标量+向量+张量)** |
| **位置编码** | 可学习 | Sin-Cos | **球谐基 + Rotary** |
| **邻居机制** | 全局注意力 | 全局注意力 | **局部 KNN/稀疏邻接** |
| **卷积层** | 2D Conv | 无 | **ConvSE3 (消息传递)** |
| **归一化** | LayerNorm/AdaLN | RmsNorm | **NormSE3 (等变归一化)** |
| **应用场景** | MNIST/ImageNet | 机器人轨迹 | AlphaFold2/药物设计 |
| **模型规模** | ~350K | ~1B | ~1M - 10M |

---

## 🧪 关键代码片段

### **Fiber 数据结构**

```python
# 创建 Fiber
fiber = Fiber.create(num_degrees=3, dim=(64, 32, 16))
# 等价于
fiber = Fiber([
    FiberEl(degrees=0, dim=64),
    FiberEl(degrees=1, dim=32),
    FiberEl(degrees=2, dim=16)
])

# 访问
fiber[0]  # 返回 64
fiber[1]  # 返回 32

# Fiber 乘积 (用于构建所有 type 对)
for (degree_in, dim_in), (degree_out, dim_out) in (fiber * fiber):
    print(f"Type-{degree_in} → Type-{degree_out}: {dim_in} → {dim_out}")
```

---

### **ConvSE3 核心流程**

```python
class ConvSE3:
    def forward(self, inp, edge_info, rel_dist, basis):
        neighbor_indices, neighbor_masks, edges = edge_info
        outputs = {}
        
        # 对每个输出类型
        for degree_out in self.fiber_out.degrees:
            output = 0
            
            # 聚合所有输入类型
            for degree_in, m_in in self.fiber_in:
                x = inp[str(degree_in)]
                
                # 1. 选择邻居
                x = batched_index_select(x, neighbor_indices, dim=1)
                # (B, N, D, m) → (B, N, K, D, m)
                
                # 2. Reshape
                x = x.view(*x.shape[:3], to_order(degree_in) * m_in, 1)
                # (B, N, K, D*m, 1)
                
                # 3. 径向核
                kernel_fn = self.kernel_unary[f'({degree_in},{degree_out})']
                edge_features = torch.cat((rel_dist, edges), dim=-1)
                kernel = kernel_fn(edge_features, basis=basis)
                # (B, N, K, out_dim, in_dim, num_freq)
                
                # 4. Einstein 求和
                chunk = einsum('... o i, ... i c -> ... o c', kernel, x)
                output = output + chunk
            
            # 5. 聚合邻居
            if self.pool:
                output = masked_mean(output, neighbor_masks, dim=2)
            
            # 6. Reshape 到正确的 type 形状
            output = output.view(*leading_shape, -1, to_order(degree_out))
            outputs[str(degree_out)] = output
        
        # 7. 自交互
        if self.self_interaction:
            self_out = self.self_interact(inp)
            outputs = self.self_interact_sum(outputs, self_out)
        
        return outputs
```

---

### **AttentionSE3 多类型注意力**

```python
class AttentionSE3:
    def forward(self, features, edge_info, rel_dist, basis, ...):
        queries = self.to_q(features)      # LinearSE3
        keys = self.to_k(features, ...)    # ConvSE3
        values = self.to_v(features, ...)  # ConvSE3
        
        outputs = {}
        
        # 对每个 type 分别计算注意力
        for degree in features.keys():
            q = queries[degree]
            k = keys[degree]
            v = values[degree]
            
            # 重排为多头
            q = rearrange(q, 'b n (h d) m -> b h n d m', h=heads)
            k = rearrange(k, 'b n j (h d) m -> b h n j d m', h=heads)
            v = rearrange(v, 'b n j (h d) m -> b h n j d m', h=heads)
            
            # 旋转位置编码 (仅 Type-0)
            if exists(pos_emb) and degree == '0':
                query_pos_emb, key_pos_emb = pos_emb
                q = apply_rotary_pos_emb(q, query_pos_emb)
                k = apply_rotary_pos_emb(k, key_pos_emb)
            
            # 注意力分数: 在 d 和 m 维度上点积
            sim = einsum('bhidm, bhijdm -> bhij', q, k) * self.scale
            
            # 掩码
            if exists(neighbor_mask):
                sim = sim.masked_fill(~mask, -inf)
            
            # Softmax
            attn = sim.softmax(dim=-1)
            
            # 加权求和
            out = einsum('bhij, bhijdm -> bhidm', attn, v)
            
            # 合并多头
            outputs[degree] = rearrange(out, 'b h n d m -> b n (h d) m')
        
        return self.to_out(outputs)
```

---

### **NormSE3 等变归一化**

```python
class NormSE3:
    def forward(self, features):
        output = {}
        
        for degree, t in features.items():
            # t: (B, N, D, m) 其中 m = 2*degree + 1
            
            # 1. 计算范数 (旋转不变)
            norm = t.norm(dim=-1, keepdim=True)  # (B, N, D, 1)
            norm = norm.clamp(min=self.eps)      # 避免除零
            
            # 2. 归一化方向 (等变)
            phase = t / norm                     # (B, N, D, m)
            
            # 3. 标量变换
            norm_flat = rearrange(norm, '... () -> ...')  # (B, N, D)
            
            # 门控 (可选)
            if gate_weights is not None:
                scale = einsum('bnd, de -> bne', norm_flat, gate_weights)
            else:
                scale = self.scale  # 可学习的标量
            
            # 非线性激活
            transformed = self.nonlin(norm_flat * scale)  # (B, N, D)
            
            # 4. 重组 (保持等变性)
            transformed = rearrange(transformed, '... -> ... ()')  # (B, N, D, 1)
            output[degree] = (transformed * phase).view(*t.shape)
        
        return output
```

---

## 📊 模型参数量估算

**典型配置 (AlphaFold2 风格)**:
- dim=64, depth=6, num_degrees=4, heads=8, dim_head=64

```
Token Embedding (可选):
  - Embedding: 28 × 64 = 1.8K

Position Embedding (可选):
  - 1024 × 64 = 65K

ConvSE3 输入层:
  - Kernel MLPs: ~50K per degree pair
  - 约 4 个 degree pairs × 50K = 200K

SE3 Transformer Blocks (×6):
  每个 Block:
    AttentionSE3:
      - to_q (LinearSE3): 64×512 × num_degrees = ~200K
      - to_k (ConvSE3): ~150K
      - to_v (ConvSE3): ~150K
      - to_out: 512×64 × num_degrees = ~200K
      小计: ~700K
    
    FeedForwardSE3:
      - project_in: 64×256 × num_degrees = ~100K
      - project_out: 256×64 × num_degrees = ~100K
      小计: ~200K
    
    每个 Block: ~900K
    6 个 Block: ~5.4M

ConvSE3 输出层:
  - ~200K

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: ~6M 参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

注: 实际参数量取决于:
  - num_degrees (更多 type 需要更多参数)
  - fiber 维度配置
  - 是否使用 reduce_dim_out
```

---

## 🎓 学习建议

### **理解顺序**

1. **第一步**: 理解 Fiber 数据结构和多类型特征
2. **第二步**: 理解球谐基函数 (`basis.py`)
3. **第三步**: 理解 `NormSE3` (最简单的等变操作)
4. **第四步**: 理解 `ConvSE3` (消息传递机制)
5. **第五步**: 理解 `AttentionSE3` (等变注意力)
6. **第六步**: 理解完整的 `SE3Transformer` 流程

### **数学预备知识**

建议先了解:
- **群论基础**: 什么是群、对称性、表示
- **SO(3) 群**: 3D 旋转群
- **球谐函数**: Y_lm 的定义和性质
- **Wigner-D 矩阵**: 旋转表示

### **调试技巧**

```python
# 1. 打印所有 type 的形状
def print_fiber_shapes(features, name=""):
    print(f"{name}:")
    for degree, tensor in features.items():
        print(f"  Type-{degree}: {tensor.shape}")

# 2. 验证等变性
def test_equivariance(model, feats, coors, rotation_matrix):
    # 原始输出
    out1 = model(feats, coors)
    
    # 旋转后输出
    coors_rot = coors @ rotation_matrix.T
    out2 = model(feats, coors_rot)
    
    # Type-1 应该等变
    out1_rot = out1['1'] @ rotation_matrix.T
    print(f"Equivariance error: {(out1_rot - out2['1']).abs().max()}")

# 3. 可视化注意力
def visualize_attention(model, feats, coors):
    # 在 AttentionSE3 forward 中添加:
    # self.last_attn = attn.detach()
    
    attn = model.layers[0][0].attn.last_attn  # (B, h, N, K)
    
    import matplotlib.pyplot as plt
    plt.imshow(attn[0, 0].cpu())  # 第一个 batch, 第一个 head
    plt.colorbar()
    plt.title("Attention Map (Type-0)")
    plt.show()
```

### **动手实验**

1. **简单数据**: 用 3-4 个点测试,手动计算验证
2. **等变性测试**: 旋转输入,检查输出是否相应旋转
3. **可视化**: 可视化注意力权重和特征
4. **消融实验**: 
   - 只用 Type-0 vs Type-0+1
   - ConvSE3 vs LinearSE3 for keys
   - 不同的 num_degrees

### **常见问题**

**Q1: 为什么需要多种 Type?**
- A: 不同物理量有不同的变换性质。标量 (能量) 不变,向量 (力) 旋转,需要不同的 Type 表示。

**Q2: 球谐基是什么?**
- A: 球面上的"傅里叶基",可以表示任何方向依赖的函数,且具有良好的旋转性质。

**Q3: 为什么 Type-l 有 2l+1 个分量?**
- A: 对应球谐函数 Y_l^m 的 2l+1 个独立模式 (m=-l,...,+l)。

**Q4: ConvSE3 和 AttentionSE3 的区别?**
- A: ConvSE3 类似 GNN 消息传递,所有邻居平等加权;AttentionSE3 用注意力动态加权。

**Q5: 如何选择 num_degrees?**
- A: 从 1-2 开始,任务需要更复杂几何时增加。更多 degrees = 更强表达 + 更多计算。

---

## 🔗 参考资料

- **SE(3)-Transformers 论文**: [SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://arxiv.org/abs/2006.10503)
- **E(3)NN 框架**: [e3nn: Euclidean Neural Networks](https://github.com/e3nn/e3nn)
- **AlphaFold2**: [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)
- **球谐函数**: [Spherical Harmonics (Wikipedia)](https://en.wikipedia.org/wiki/Spherical_harmonics)
- **群论**: [Group Theory and Equivariant Networks (Tutorial)](https://arxiv.org/abs/2008.09054)

---

## 📌 关键术语对照表

| 英文 | 中文 | 说明 |
|------|------|------|
| SE(3) | 特殊欧几里得群(3D) | 3D 旋转 + 平移 |
| Equivariance | 等变性 | 变换输入 = 相应变换输出 |
| Invariance | 不变性 | 变换输入,输出不变 |
| Fiber | 纤维 | 多类型特征的集合 |
| Type-l | l阶类型 | 对应 l 阶球谐函数 |
| Spherical Harmonics | 球谐函数 | 球面上的正交基 |
| Wigner-D Matrix | Wigner-D 矩阵 | SO(3) 群的表示矩阵 |
| Irreducible Representation | 不可约表示 | 群表示的基本单元 |
| Radial Function | 径向函数 | 只依赖距离的函数 |
| Tensor Product | 张量积 | 两个表示的组合 |
| Message Passing | 消息传递 | GNN 中的邻居信息聚合 |
| Basis | 基函数 | 函数空间的正交基 |

---

**祝你学习顺利! SE(3) 等变网络是几何深度学习的前沿,掌握它将打开 3D AI 的大门!** 🧬🔬🎉
