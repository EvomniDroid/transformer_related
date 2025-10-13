"""
DiT 模型训练脚本

训练流程：
1. 从数据集加载清晰图像 x_0
2. 随机采样时间步 t
3. 使用前向扩散添加噪声：x_0 → x_t
4. 用 DiT 模型预测噪声
5. 计算预测噪声与真实噪声的损失
6. 反向传播更新模型参数
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 OpenMP 库冲突

from config import *
from torch.utils.data import DataLoader
from dataset import MNIST
from diffusion import forward_add_noise
import torch 
from torch import nn 
from dit import DiT

# ===== 设备配置 =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  使用设备: {DEVICE}")

# ===== 数据集 =====
dataset = MNIST()
print(f"📦 数据集大小: {len(dataset)}")

# ===== 模型初始化 =====
model = DiT(
    img_size=28,      # MNIST 图像大小
    patch_size=4,     # Patch 大小
    channel=1,        # 灰度图
    emb_size=64,      # Embedding 维度
    label_num=10,     # 10 个数字类别
    dit_num=3,        # 3 层 DiT Block
    head=4            # 4 个注意力头
).to(DEVICE)

print(f"🤖 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ===== 加载预训练模型（如果存在）=====
try:
    model.load_state_dict(torch.load('model.pth'))
    print("✅ 加载预训练模型成功")
except:
    print("⚠️  未找到预训练模型，从头开始训练")

# ===== 优化器 =====
# Adam 优化器：自适应学习率优化算法
# - model.parameters(): 模型的所有可训练参数
# - lr=1e-3: 学习率 0.001
optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"🔧 优化器: Adam, 学习率: 1e-3")

# ===== 损失函数 =====
# L1Loss (Mean Absolute Error): 计算预测噪声和真实噪声的绝对值误差
# Loss = mean(|pred_noise - true_noise|)
loss_fn = nn.L1Loss()
print(f"📉 损失函数: L1Loss (MAE)")

# ===== 训练循环 =====
if __name__ == '__main__':
    # 训练参数
    EPOCH = 500        # 训练轮数
    BATCH_SIZE = 300   # 批次大小
    
    # Windows 下使用 num_workers=0 避免多进程问题
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,          # 每个 epoch 打乱数据
        num_workers=0          # Windows 兼容性
    )
    
    # ===== 查看单张图片的详细信息 =====
    for inputs, labels in dataloader:
        print("\n" + "="*60)
        print("📊 数据批次信息:")
        print(f"  - Batch 大小: {inputs.shape[0]} 张图片")
        print(f"  - 图片形状: {inputs.shape}  (batch, channel, height, width)")
        print(f"  - 标签形状: {labels.shape}")
        print("="*60)
        
        # 只看第一张图片
        first_img = inputs[0]      # 形状: (1, 28, 28)
        first_label = labels[0]    # 形状: 标量
        
        print(f"\n🖼️  第一张图片的标签: {first_label.item()} (数字 {first_label.item()})")
        print(f"   图片形状: {first_img.shape}")
        print(f"   像素值范围: [{first_img.min():.4f}, {first_img.max():.4f}]")
        
        # 显示图片的像素矩阵 (28x28)
        print(f"\n   像素矩阵 (28x28):")
        print(f"   {'─'*56}")
        # 去掉 channel 维度,只看二维矩阵
        img_2d = first_img.squeeze(0)  # (28, 28)
        
        # 打印前 10 行,每行前 10 个像素
        for i in range(10):
            row = img_2d[i, :10]
            print(f"   Row {i:2d}: [{', '.join([f'{x:.2f}' for x in row])} ...]")
        print(f"   {'─'*56}")
        print(f"   (仅显示前 10x10 像素,完整图片是 28x28)")
        
        # 查看哪些位置有非零像素 (数字笔画的位置)
        nonzero_count = (img_2d > 0).sum().item()
        print(f"\n   ✏️  非零像素数量: {nonzero_count} / 784 ({nonzero_count/784*100:.1f}%)")
        print(f"   💡 这些非零像素就是数字 '{first_label.item()}' 的笔画部分")
        
        break  # 只看第一个 batch
    
    import pdb; pdb.set_trace()
    # 设置为训练模式（启用 Dropout, BatchNorm 等）
    model.train()
    
    iter_count = 0  # 迭代计数器
    
    # ===== Epoch 循环 =====
    for epoch in range(EPOCH):
        epoch_loss = 0.0  # 记录当前 epoch 的总损失
        batch_count = 0
        
        print(f"\n📊 Epoch {epoch+1}/{EPOCH}")
  
        # ===== Batch 循环 =====
        for imgs, labels in dataloader:
            # imgs: (batch_size, 1, 28, 28), 取值范围 [0, 1]
            # labels: (batch_size,), 类别标签 0-9
            
            # ===== 步骤1: 图像标准化 =====
            # 将像素值从 [0, 1] 转换到 [-1, 1]
            # 这样可以和高斯噪声 N(0, 1) 的范围匹配
            x = imgs * 2 - 1  # (batch_size, 1, 28, 28), 范围 [-1, 1]
            
            # ===== 步骤2: 随机采样时间步 =====
            # 为每张图片随机选择一个时间步 t ∈ [0, T-1]
            # 这样模型可以学习在不同噪声水平下预测噪声
            t = torch.randint(0, T, (imgs.size(0),))  # (batch_size,)
            
            # ===== 步骤3: 准备条件标签 =====
            y = labels  # (batch_size,)
            
            # ===== 步骤4: 前向扩散（加噪）=====
            # x_0 → x_t，同时返回添加的噪声
            x_noisy, noise = forward_add_noise(x, t)
            # x_noisy: (batch_size, 1, 28, 28) - 加噪后的图像
            # noise: (batch_size, 1, 28, 28) - 添加的噪声（ground truth）
            
            # ===== 步骤5: 模型预测噪声 =====
            # 输入：带噪图像 x_t, 时间步 t, 标签 y
            # 输出：预测的噪声
            pred_noise = model(
                x_noisy.to(DEVICE), 
                t.to(DEVICE), 
                y.to(DEVICE)
            )  # (batch_size, 1, 28, 28)
            
            # ===== 步骤6: 计算损失 =====
            # L1 损失：|预测噪声 - 真实噪声|
            loss = loss_fn(pred_noise, noise.to(DEVICE))
            
            # ===== 步骤7: 反向传播 =====
            optimzer.zero_grad()  # 清空之前的梯度
            loss.backward()        # 计算梯度
            optimzer.step()        # 更新参数
            
            # ===== 记录损失 =====
            epoch_loss += loss.item()
            batch_count += 1
            
            # ===== 定期保存模型和打印日志 =====
            if iter_count % 100 == 0:
                print(f"  Iter {iter_count:5d} | Loss: {loss.item():.6f}")
            
            if iter_count % 1000 == 0:
                # 保存模型
                torch.save(model.state_dict(), 'model.pth')
                print(f"  💾 模型已保存 (iter={iter_count})")
            
            iter_count += 1
        
        # ===== Epoch 结束统计 =====
        avg_loss = epoch_loss / batch_count
        print(f"  ✅ Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.6f}")
        
        # 每个 epoch 结束保存一次
        torch.save(model.state_dict(), 'model.pth')
    
    print(f"\n{'='*60}")
    print(f"🎉 训练完成！")
    print(f"{'='*60}")
    print(f"总迭代次数: {iter_count}")
    print(f"模型已保存到: model.pth")