"""
MNIST 数据集加载器

功能：
1. 加载 MNIST 手写数字数据集
2. 将 PIL 图像转换为 Tensor
3. 归一化到 [0, 1] 范围
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 OpenMP 库冲突

from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor, Compose
import torchvision

# 手写数字数据集
class MNIST(Dataset):
    """
    MNIST 数据集包装器
    
    参数：
        is_train: 是否使用训练集（True）或测试集（False）
    
    返回：
        img: 归一化的图像 Tensor，shape=(1, 28, 28)，取值范围 [0, 1]
        label: 类别标签，0-9 的整数
    """
    def __init__(self, is_train=True):
        super().__init__()
        
        # 下载并加载 MNIST 数据集
        # train=True: 60,000 张训练图像
        # train=False: 10,000 张测试图像
        self.ds = torchvision.datasets.MNIST(
            './mnist/',       # 数据保存路径
            train=is_train,   # 训练集或测试集
            download=True     # 自动下载
        )
        
        # 图像转换流程：PIL Image → Tensor
        self.img_convert = Compose([
            PILToTensor(),  # PIL Image (28, 28) → Tensor (1, 28, 28)，值范围 [0, 255]
        ])
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.ds)
    
    def __getitem__(self, index):
        """
        获取单个样本
        
        参数：
            index: 样本索引 (0 到 len-1)
        
        返回：
            img: 归一化后的图像，shape=(1, 28, 28)，范围 [0, 1]
            label: 类别标签 (0-9)
        """
        # 从原始数据集获取 PIL 图像和标签
        img, label = self.ds[index]
        
        # 转换为 Tensor 并归一化到 [0, 1]
        # PILToTensor 输出范围是 [0, 255]，除以 255 归一化
        return self.img_convert(img) / 255.0, label
    
if __name__ == '__main__':
    # ===== 测试代码 =====
    import matplotlib.pyplot as plt 
    
    print("📦 加载 MNIST 数据集...")
    ds = MNIST()
    
    print(f"✅ 数据集大小: {len(ds)}")
    
    # 获取第一个样本
    img, label = ds[0]
    
    print(f"📊 图像信息:")
    print(f"   Shape: {img.shape}")
    print(f"   数据类型: {img.dtype}")
    print(f"   取值范围: [{img.min():.3f}, {img.max():.3f}]")
    print(f"   标签: {label}")
    
    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(img.permute(1, 2, 0), cmap='gray')  # (1,28,28) → (28,28,1)
    plt.title(f"MNIST Sample - Label: {label}")
    plt.axis('off')
    plt.show()
    
    # 测试多个样本
    print(f"\n🔍 前 10 个样本的标签:")
    for i in range(10):
        _, label = ds[i]
        print(f"  样本 {i}: 标签 {label}")
    
    print(f"\n✅ 测试完成！")