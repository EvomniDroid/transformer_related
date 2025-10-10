# 🔍 ViT 模型调试指南

## 📌 断点分布

### **vit.py - 模型定义文件**

#### **__init__ 方法（模型初始化）**
- **断点1**: 查看初始化参数 `emb_size`
- **断点2**: 理解卷积层如何切分 patch（Conv2d 参数）
- **断点3**: 理解 patch embedding（Linear 映射）
- **断点4**: 理解 CLS token 和位置编码的初始化
- **断点5**: 理解 Transformer Encoder 结构

#### **forward 方法（前向传播）**
- **断点6**: 查看输入图像 shape `(batch, 1, 28, 28)`
- **断点7**: 卷积后的 patch `(batch, 16, 7, 7)`
- **断点8**: view 重塑后 `(batch, 16, 49)`
- **断点9**: permute 转置后 `(batch, 49, 16)`
- **断点10**: patch embedding 后 `(batch, 49, emb_size)`
- **断点11**: CLS token 扩展 `(batch, 1, emb_size)`
- **断点12**: 拼接 CLS 后 `(batch, 50, emb_size)`
- **断点13**: 加位置编码后 `(batch, 50, emb_size)`
- **断点14**: Transformer 编码后 `(batch, 50, emb_size)`
- **断点15**: 提取 CLS 输出 `(batch, emb_size)`
- **断点16**: 最终分类 logits `(batch, 10)`

### **inference.py - 推理脚本**
- **断点1**: 查看设备信息（CPU/GPU）
- **断点2**: 查看数据集大小
- **断点3**: 模型加载前检查
- **断点4**: 查看模型参数量
- **断点5**: 查看图像数据（shape, 取值范围, 标签）
- **断点6**: 准备推理（输入 tensor 信息）
- **断点7**: 查看推理结果（logits, 预测类别, 概率分布）

---

## 🚀 调试命令速查表

### **运行调试**
```bash
cd mnist-vit
python inference.py
```

### **PDB 常用命令**

| 命令 | 简写 | 作用 |
|------|------|------|
| `continue` | `c` | 继续执行到下一个断点 |
| `next` | `n` | 执行下一行（不进入函数） |
| `step` | `s` | 执行下一行（进入函数内部） |
| `print 变量名` | `p 变量名` | 打印变量值 |
| `list` | `l` | 显示当前代码上下文 |
| `where` | `w` | 显示调用栈 |
| `args` | `a` | 显示当前函数参数 |
| `quit` | `q` | 退出调试 |
| `help` | `h` | 显示帮助 |

### **查看 Tensor 的技巧**

```python
# 在 pdb 提示符下输入：

# 查看 shape
p x.shape

# 查看 device
p x.device

# 查看数据类型
p x.dtype

# 查看具体数值（小tensor）
p x

# 查看统计信息
p x.min(), x.max(), x.mean()

# 查看第一个样本
p x[0]

# 查看第一个样本的第一个通道
p x[0, 0]

# 转为 numpy 查看
p x.cpu().numpy()
```

---

## 📊 关键步骤详解

### **1. 图像切分 Patch（断点7）**
```
输入: (1, 1, 28, 28) → 单张灰度图
     ↓ Conv2d(kernel=4, stride=4)
输出: (1, 16, 7, 7) → 7×7 个 patch，每个 patch 16 维

理解：
- 28×28 图像 → 切成 7×7 = 49 个 patch
- 每个 patch 是 4×4 = 16 个像素
```

### **2. 序列化 Patch（断点8-9）**
```
(1, 16, 7, 7) 
    ↓ view
(1, 16, 49) → 16维特征 × 49个patch
    ↓ permute
(1, 49, 16) → 49个token，每个16维
```

### **3. 添加 CLS Token（断点11-12）**
```
patch序列: (1, 49, 16)
CLS token: (1,  1, 16)
    ↓ concat
拼接结果: (1, 50, 16) → 序列长度变为50
```

### **4. Transformer 处理（断点14）**
```
输入: (1, 50, emb_size)
    ↓ Self-Attention × 3 layers
输出: (1, 50, emb_size) → shape不变，特征融合
```

### **5. 分类输出（断点15-16）**
```
取CLS: y[:, 0, :] → (1, emb_size)
    ↓ Linear(emb_size → 10)
Logits: (1, 10) → 10个类别的分数
```

---

## 🎯 调试任务清单

### **初学者任务**
- [ ] 在断点7处，用 `p x.shape` 查看卷积后的 shape
- [ ] 在断点9处，比较 permute 前后的数据布局差异
- [ ] 在断点12处，验证 CLS token 是否正确拼接
- [ ] 在断点16处，查看最终的分类 logits

### **进阶任务**
- [ ] 在断点10处，查看 `self.patch_emb.weight.shape`
- [ ] 在断点13处，验证位置编码是加法还是拼接
- [ ] 在断点14处，进入 Transformer 内部（用 `s` 命令）
- [ ] 在断点7处，手动计算 patch 数量是否等于 49

### **高级任务**
- [ ] 修改 `emb_size` 参数，观察模型变化
- [ ] 打印 `self.tranformer_enc` 的结构
- [ ] 查看 CLS token 的初始值分布
- [ ] 比较训练前后 CLS token 的变化

---

## 💡 常见问题排查

### **Q: 断点太多，如何跳过？**
A: 使用 `c` 命令跳到下一个断点，或者注释掉不需要的 `pdb.set_trace()`

### **Q: 如何临时关闭所有断点？**
A: 在文件开头添加：
```python
import pdb
pdb.set_trace = lambda: None  # 禁用所有断点
```

### **Q: 如何只在特定条件下断点？**
A: 改为条件断点：
```python
if epoch == 0:  # 只在第一个 epoch 断点
    pdb.set_trace()
```

### **Q: GPU tensor 无法直接打印？**
A: 先转到 CPU：
```python
p x.cpu()
```

---

## 📝 学习路径建议

### **第一遍：快速浏览**
只关注 shape 变化，在每个断点用 `p x.shape` 查看，然后 `c` 继续

### **第二遍：深入数据**
在关键断点（7, 12, 14, 16）用 `p x` 查看实际数值

### **第三遍：代码逻辑**
用 `s` 命令进入 Transformer 内部，理解 Attention 机制

### **第四遍：参数探索**
查看所有权重矩阵的 shape，理解参数如何组织

---

## 🔗 相关资料

- ViT 论文: "An Image is Worth 16x16 Words"
- PyTorch Transformer 文档: https://pytorch.org/docs/stable/nn.html#transformer
- PDB 官方文档: https://docs.python.org/3/library/pdb.html

---

**提示**: 建议准备笔记本，记录每个断点处的 shape 和理解，形成自己的知识体系！
