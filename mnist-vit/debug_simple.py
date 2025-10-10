import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from vit import ViT
import torch.nn.functional as F
import pdb

# 🎯 简化版：关闭 __init__ 中的断点，只保留 forward 中的
import vit
# 暂时禁用 pdb（用于跳过初始化断点）
original_set_trace = pdb.set_trace
pdb.set_trace = lambda: None

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")

dataset=MNIST()
print(f"✅ 数据集大小: {len(dataset)}")

model=ViT().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 恢复 pdb（开始在 forward 中断点）
pdb.set_trace = original_set_trace

print("\n" + "="*60)
print("🔍 开始逐步调试 ViT 模型的 forward 过程")
print("="*60 + "\n")

image, label = dataset[13]
print(f"📌 正确分类: {label}")
print(f"📌 图像 shape: {image.shape}")
print(f"📌 图像取值范围: [{image.min():.3f}, {image.max():.3f}]")

# 🔍 主调试入口：这里会进入 vit.py 的 forward 方法
input_tensor = image.unsqueeze(0).to(DEVICE)
print(f"\n开始推理，输入 shape: {input_tensor.shape}\n")

logits = model(input_tensor)

print(f"\n✅ 最终结果:")
print(f"   预测类别: {logits.argmax(-1).item()}")
print(f"   正确类别: {label}")
print(f"   预测正确: {logits.argmax(-1).item() == label}")

# 显示图像
plt.imshow(image.permute(1,2,0))
plt.title(f"True: {label}, Pred: {logits.argmax(-1).item()}")
plt.show()
