"""
DiT 模型配置文件

包含训练和推理的超参数设置
"""

import torch 

# ===== 扩散过程参数 =====
T = 1000  # 扩散总步数（最大时间步）
# T=1000 意味着：
# - 前向扩散：从 t=0 (原图) 到 t=999 (纯噪声)，逐步加噪
# - 反向去噪：从 t=999 (纯噪声) 到 t=0 (还原图)，逐步去噪

# ===== 设备配置 =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== 模型参数 =====
IMG_SIZE = 28        # 图像大小（MNIST 为 28×28）
PATCH_SIZE = 4       # Patch 大小（4×4 的 patch）
CHANNEL = 1          # 通道数（灰度图为 1）
EMB_SIZE = 64        # Embedding 维度
LABEL_NUM = 10       # 类别数量（MNIST 有 10 个数字）
DIT_NUM = 3          # DiT Block 层数
HEAD = 4             # Multi-head Attention 的头数

# ===== 训练参数 =====
BATCH_SIZE = 128     # 批次大小
LEARNING_RATE = 1e-4 # 学习率
EPOCHS = 100         # 训练轮数

# ===== 噪声调度参数 =====
BETA_START = 0.0001  # 初始噪声强度（t=0 附近）
BETA_END = 0.02      # 最终噪声强度（t=T-1 附近）

# 打印配置信息
if __name__ == '__main__':
    print("🔧 DiT 模型配置")
    print("=" * 50)
    print(f"扩散步数 T: {T}")
    print(f"设备: {DEVICE}")
    print(f"图像大小: {IMG_SIZE}×{IMG_SIZE}")
    print(f"Patch 大小: {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"Patch 数量: {(IMG_SIZE//PATCH_SIZE)**2}")
    print(f"通道数: {CHANNEL}")
    print(f"Embedding 维度: {EMB_SIZE}")
    print(f"DiT 层数: {DIT_NUM}")
    print(f"注意力头数: {HEAD}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print("=" * 50)