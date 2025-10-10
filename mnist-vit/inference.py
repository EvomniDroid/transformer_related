import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'  # 解决OpenMP库冲突

from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from vit import ViT
import torch.nn.functional as F
import pdb  # 调试工具

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

# 🔍 断点1：查看设备信息
print(f"使用设备: {DEVICE}")
pdb.set_trace()

dataset=MNIST() # 数据集

# 🔍 断点2：查看数据集信息
print(f"数据集大小: {len(dataset)}")
pdb.set_trace()

model=ViT().to(DEVICE) # 模型

# 🔍 断点3：查看模型加载前
print("开始加载模型权重...")
pdb.set_trace()

model.load_state_dict(torch.load('model.pth'))

# 🔍 断点4：查看模型参数
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
pdb.set_trace()

model.eval()    # 预测模式

'''
对图片分类
'''
image,label=dataset[13]

# 🔍 断点5：查看图像数据
print(f"图像shape: {image.shape}")
print(f"图像取值范围: [{image.min():.3f}, {image.max():.3f}]")
print(f"正确分类: {label}")
pdb.set_trace()

plt.imshow(image.permute(1,2,0))
plt.show()

# 🔍 断点6：准备推理
print("开始推理...")
input_tensor = image.unsqueeze(0).to(DEVICE)
print(f"输入tensor shape: {input_tensor.shape}")
print(f"输入tensor device: {input_tensor.device}")
pdb.set_trace()

logits=model(input_tensor)

# 🔍 断点7：查看推理结果
print(f"logits: {logits}")
print(f"logits shape: {logits.shape}")
print(f"预测分类: {logits.argmax(-1).item()}")
print(f"预测概率分布: {torch.softmax(logits, dim=-1)}")
pdb.set_trace()

print('预测分类:',logits.argmax(-1).item())