# 🔍 ViT 调试快速开始

## 📌 三种调试模式

### **模式 1: 只调试 forward（推荐新手）** ⭐
```bash
python debug_simple.py
```
- ✅ 跳过模型初始化的断点
- ✅ 只在 forward 方法中逐步调试
- ✅ 关注数据流和 shape 变化

### **模式 2: 完整调试（包括初始化）**
```bash
# Windows PowerShell
$env:DEBUG_INIT="1"; python inference.py

# Linux/Mac
DEBUG_INIT=1 python inference.py
```
- 包含 `__init__` 方法的断点
- 理解模型参数如何初始化

### **模式 3: 原始 inference（所有断点）**
```bash
python inference.py
```
- 完整的调试流程
- 从设备检测到最终推理

---

## 🎯 推荐学习路径

### **Day 1: 理解数据流（使用 debug_simple.py）**
重点断点：
- 断点6: 输入 shape
- 断点7: 卷积后 patch
- 断点9: 序列化
- 断点12: 添加 CLS
- 断点16: 最终输出

每个断点执行：
```python
(Pdb) p x.shape    # 查看形状
(Pdb) c            # 继续到下一个断点
```

### **Day 2: 深入数据内容**
在断点7查看 patch 的实际值：
```python
(Pdb) p x[0, :, 0, 0]  # 第一个样本，所有通道，第一个patch位置
(Pdb) p x.min(), x.max()  # 数值范围
```

### **Day 3: 理解 Transformer**
在断点14进入 Transformer 内部：
```python
(Pdb) s            # step into
# 然后逐行跟踪 Self-Attention 计算
```

### **Day 4: 探索参数**
查看模型参数：
```python
(Pdb) p self.conv.weight.shape
(Pdb) p self.patch_emb.weight.shape
(Pdb) p self.cls_token
(Pdb) p self.pos_emb.shape
```

---

## 💡 PDB 命令速记卡

| 命令 | 说明 | 使用场景 |
|------|------|---------|
| `c` (continue) | 继续到下一个断点 | 快速浏览 |
| `n` (next) | 下一行（不进入函数） | 逐行执行 |
| `s` (step) | 下一行（进入函数） | 深入理解 |
| `p x` | 打印变量 x | 查看数据 |
| `pp x` | 美化打印 | 复杂结构 |
| `l` (list) | 显示代码 | 定位位置 |
| `w` (where) | 调用栈 | 了解层级 |
| `q` (quit) | 退出调试 | 结束 |

---

## 📊 Shape 变化总览

```
输入图像:      (1, 1, 28, 28)   ← 单张灰度图
                    ↓ Conv2d(4×4, stride=4)
切分patch:     (1, 16, 7, 7)    ← 7×7个patch，每个16通道
                    ↓ view
展平:          (1, 16, 49)      ← 49个位置
                    ↓ permute
序列化:        (1, 49, 16)      ← 49个token，每个16维
                    ↓ Linear
Embedding:     (1, 49, 16)      ← patch embedding
                    ↓ cat([CLS])
添加CLS:       (1, 50, 16)      ← 序列长度+1
                    ↓ add pos_emb
位置编码:      (1, 50, 16)      ← 加上位置信息
                    ↓ TransformerEncoder
Transformer:   (1, 50, 16)      ← Self-Attention × 3
                    ↓ select [CLS]
提取CLS:       (1, 16)          ← 取第一个token
                    ↓ Linear
分类输出:      (1, 10)          ← 10个类别的logits
```

---

## 🎓 学习检查点

完成以下任务，标记 ✅：

### **初级**
- [ ] 能说出 28×28 如何变成 7×7 的 patch
- [ ] 理解为什么序列长度是 50（49个patch + 1个CLS）
- [ ] 知道 CLS token 的作用
- [ ] 能解释位置编码的必要性

### **中级**
- [ ] 能手算每一步的 shape 变化
- [ ] 理解 view 和 permute 的区别
- [ ] 知道 Transformer 如何处理序列
- [ ] 理解为什么只用 CLS token 输出做分类

### **高级**
- [ ] 能修改 patch_size 并预测 shape 变化
- [ ] 理解 Self-Attention 的计算过程
- [ ] 能解释模型参数量的计算
- [ ] 可以自己实现一个简化版 ViT

---

## 🚨 常见问题

### Q: 断点太多，跳过太累？
**A**: 使用 `debug_simple.py`，它自动跳过初始化断点

### Q: 想只在某个断点停下？
**A**: 在其他断点按 `c` 快速跳过，或者注释掉不需要的 `pdb.set_trace()`

### Q: 如何查看 GPU 上的 tensor？
**A**: `p x.cpu()` 先转到 CPU

### Q: 想看完整的数组值？
**A**: 
```python
(Pdb) import numpy as np
(Pdb) np.set_printoptions(threshold=np.inf)
(Pdb) p x.cpu().numpy()
```

---

## 📚 扩展阅读

- **ViT 论文**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **可视化工具**: [Netron](https://netron.app/) - 查看模型结构
- **Transformer 详解**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

**祝你调试愉快！有问题随时查阅 DEBUG_GUIDE.md** 🎉
