"""
扩散过程 (Diffusion Process) 实现

扩散模型的核心：
1. 前向过程（Forward）：逐步向图像添加噪声，直到变成纯噪声
2. 反向过程（Reverse）：训练模型学习去噪，从噪声还原图像

数学原理：
- 前向扩散：q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
- 即：x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

其中：
- x_0: 原始清晰图像
- x_t: t 时刻的带噪图像
- noise: 标准高斯噪声 N(0, I)
- alpha_t = 1 - beta_t
- alpha_cumprod_t = alpha_1 * alpha_2 * ... * alpha_t
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 OpenMP 库冲突

import torch 
from config import T

# ===== 噪声调度参数 =====
# Beta: 控制每一步添加多少噪声
# beta_t 从 0.0001 逐渐增加到 0.02
# - 早期（t小）: beta 小，加噪慢，保留更多原图信息
# - 后期（t大）: beta 大，加噪快，快速变成纯噪声
betas = torch.linspace(0.0001, 0.02, T)  # shape: (T,)

# Alpha: alpha_t = 1 - beta_t
# 表示保留原图的比例
alphas = 1 - betas  # shape: (T,)

# Alpha 累积乘积: alpha_cumprod_t = alpha_1 * alpha_2 * ... * alpha_t
# 这是从 x_0 直接跳到 x_t 的关键参数
# 例如：alpha_cumprod[100] 表示从原图一步跳到第 100 步的衰减系数
alphas_cumprod = torch.cumprod(alphas, dim=-1)  # shape: (T,)
# [a_1, a_1*a_2, a_1*a_2*a_3, ...]

# Alpha 累积乘积（前一步）
# 用于反向去噪时的方差计算
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)
# shape: (T,)
# [1, a_1, a_1*a_2, a_1*a_2*a_3, ...]

# 反向去噪的方差
# 公式：variance_t = (1 - alpha_t) * (1 - alpha_cumprod_{t-1}) / (1 - alpha_cumprod_t)
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # shape: (T,)

def forward_add_noise(x, t):
    """
    前向加噪：一步到位地从 x_0 生成 x_t
    
    公式：x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    
    参数：
        x: 原始图像，shape=(batch, channel, height, width)
           取值范围应该是 [-1, 1]（标准化后）
        t: 时间步，shape=(batch,)
           例如：[500, 200, 800, ...] - 每个样本的时间步可以不同
    
    返回：
        x_t: 加噪后的图像，shape=(batch, channel, height, width)
        noise: 添加的高斯噪声，shape=(batch, channel, height, width)
    """
    
    # 生成标准高斯噪声 N(0, I)
    noise = torch.randn_like(x)  # shape: (batch, channel, height, width)
    
    # 根据时间步 t 获取对应的 alpha_cumprod
    # alphas_cumprod[t] → (batch,)
    # 然后 reshape 成 (batch, 1, 1, 1) 用于广播
    batch_alphas_cumprod = alphas_cumprod[t].view(x.size(0), 1, 1, 1)
    
    # 应用前向扩散公式
    # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
    x_t = torch.sqrt(batch_alphas_cumprod) * x + torch.sqrt(1 - batch_alphas_cumprod) * noise
    
    # 返回加噪图像和噪声（训练时需要预测这个噪声）
    return x_t, noise

if __name__ == '__main__':
    # ===== 测试代码：可视化扩散过程 =====
    import matplotlib.pyplot as plt 
    from dataset import MNIST
    
    print("🧪 测试前向扩散过程")
    
    # 加载数据集
    dataset = MNIST()
    
    # 获取 2 张图像组成 batch
    x = torch.stack((dataset[0][0], dataset[1][0]), dim=0)  # shape: (2, 1, 28, 28)
    print(f"原始图像 shape: {x.shape}")
    print(f"原始图像取值范围: [{x.min():.3f}, {x.max():.3f}]")

    # ===== 显示原图 =====
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0].permute(1, 2, 0), cmap='gray')
    plt.title("Original Image 1")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x[1].permute(1, 2, 0), cmap='gray')
    plt.title("Original Image 2")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # ===== 随机时间步 =====
    t = torch.randint(0, T, size=(x.size(0),))
    print(f'\n时间步 t: {t}')
    
    # ===== 标准化到 [-1, 1] =====
    # 原始图像是 [0, 1]，扩散模型通常使用 [-1, 1]
    x = x * 2 - 1  # [0, 1] → [-1, 1]
    print(f"标准化后取值范围: [{x.min():.3f}, {x.max():.3f}]")
    
    # ===== 执行加噪 =====
    x_noisy, noise = forward_add_noise(x, t)
    print(f'\n加噪图像 shape: {x_noisy.shape}')
    print(f'噪声 shape: {noise.shape}')
    print(f'加噪图像取值范围: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]')

    # ===== 显示加噪图 =====
    # 从 [-1, 1] 还原到 [0, 1] 用于显示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(((x_noisy[0] + 1) / 2).permute(1, 2, 0), cmap='gray')
    plt.title(f"Noisy Image 1 (t={t[0]})")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(((x_noisy[1] + 1) / 2).permute(1, 2, 0), cmap='gray')
    plt.title(f"Noisy Image 2 (t={t[1]})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # ===== 可视化不同时间步的加噪效果 =====
    print("\n📊 不同时间步的加噪效果:")
    test_times = [0, 100, 300, 500, 700, 900, 999]
    
    plt.figure(figsize=(14, 4))
    for i, tt in enumerate(test_times):
        t_tensor = torch.tensor([tt])
        x_test = dataset[0][0].unsqueeze(0) * 2 - 1  # (1, 1, 28, 28), [-1, 1]
        x_noisy_test, _ = forward_add_noise(x_test, t_tensor)
        
        plt.subplot(1, len(test_times), i + 1)
        plt.imshow(((x_noisy_test[0] + 1) / 2).permute(1, 2, 0), cmap='gray')
        plt.title(f"t={tt}")
        plt.axis('off')
    
    plt.suptitle("Forward Diffusion Process (从清晰到噪声)")
    plt.tight_layout()
    plt.show()
    
    print("\n✅ 测试完成！")
    plt.show()