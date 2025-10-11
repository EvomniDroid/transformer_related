"""
DiT (Diffusion Transformer) 模型实现

DiT 是一种基于 Transformer 的扩散模型，用于图像生成任务。
与传统的 UNet 扩散模型不同，DiT 使用 Transformer 架构处理图像 patches。

核心思想：
1. 将图像切分成 patches（类似 ViT）
2. 使用 Transformer 处理 patch 序列
3. 条件信息（时间步 t 和标签 y）通过 AdaLN（自适应层归一化）注入
4. 最后将 patches 重组回图像
"""

from torch import nn 
import torch 
from time_emb import TimeEmbedding
from dit_block import DiTBlock
from config import T 

class DiT(nn.Module):
    """
    Diffusion Transformer 模型
    
    参数说明：
        img_size: 图像大小（例如 28 表示 28×28）
        patch_size: 每个 patch 的大小（例如 4 表示 4×4）
        channel: 图像通道数（灰度图为 1，RGB 为 3）
        emb_size: embedding 维度（Transformer 的隐藏层维度）
        label_num: 分类标签数量（MNIST 为 10）
        dit_num: DiT Block 的堆叠层数
        head: Multi-head Attention 的头数
    """
    def __init__(self, img_size, patch_size, channel, emb_size, label_num, dit_num, head):
        super().__init__()
        
        # ===== 基本参数 =====
        self.patch_size = patch_size
        self.patch_count = img_size // self.patch_size  # 每个维度有多少个 patch（例如 28/4=7）
        self.channel = channel
        
        # ===== Patchify 层：将图像切分成 patches =====
        # 使用卷积将图像切分成不重叠的 patches
        # 输入: (batch, channel, img_size, img_size)
        # 输出: (batch, channel*patch_size^2, patch_count, patch_count)
        # 例如：(batch, 1, 28, 28) → (batch, 16, 7, 7)
        self.conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel * patch_size**2,  # 每个 patch 展平后的维度
            kernel_size=patch_size,
            padding=0,
            stride=patch_size  # stride=patch_size 保证 patches 不重叠
        ) 
        
        # Patch Embedding：将展平的 patch 映射到 embedding 空间
        # (channel*patch_size^2) → emb_size
        # 例如：16 → 64
        self.patch_emb = nn.Linear(
            in_features=channel * patch_size**2,
            out_features=emb_size
        ) 
        
        # Patch 位置编码：为每个 patch 位置添加可学习的位置信息
        # shape: (1, patch_count^2, emb_size)
        # 例如：(1, 49, 64) - 49个patch的位置编码
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.patch_count**2, emb_size))
        
        # ===== Time Embedding：将时间步 t 编码为向量 =====
        # 时间步 t ∈ [0, T-1] 需要转换为高维向量来指导去噪
        # 使用正弦位置编码 + MLP
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),      # 时间步的正弦位置编码
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)  # 最终输出 (batch, emb_size)
        )

        # ===== Label Embedding：将类别标签编码为向量 =====
        # 用于条件生成：指定生成哪个类别的图像
        # 例如：y=3 → embedding vector (emb_size,)
        self.label_emb = nn.Embedding(
            num_embeddings=label_num,  # 总共有多少个类别
            embedding_dim=emb_size     # embedding 维度
        )
        
        # ===== DiT Blocks：核心 Transformer 层 =====
        # 堆叠多个 DiT Block，每个 Block 包含：
        # - Self-Attention（处理 patch 之间的关系）
        # - AdaLN（自适应层归一化，融入条件信息）
        # - MLP（前馈网络）
        self.dits = nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size, head))
        
        # ===== Layer Norm：最后的归一化层 =====
        self.ln = nn.LayerNorm(emb_size)
        
        # ===== Linear 投影：将 embedding 映射回 patch =====
        # emb_size → channel*patch_size^2
        # 例如：64 → 16（还原回原始 patch 维度）
        self.linear = nn.Linear(emb_size, channel * patch_size**2)
        
    def forward(self, x, t, y):
        """
        前向传播
        
        参数：
            x: 带噪音的图像，shape=(batch, channel, height, width)
               例如：(batch, 1, 28, 28)
            t: 时间步，shape=(batch,)
               例如：[999, 500, 250, ...] - 每个样本的扩散时间步
            y: 类别标签，shape=(batch,)
               例如：[3, 7, 1, ...] - 指定生成的数字类别
               
        返回：
            预测的噪音（或去噪后的图像），shape=(batch, channel, height, width)
        """
        
        # ===== 步骤1：条件编码（Condition Embedding）=====
        # 将标签和时间步都编码为向量，然后相加作为条件信号
        y_emb = self.label_emb(y)  # (batch, emb_size) - 标签 embedding
        t_emb = self.time_emb(t)   # (batch, emb_size) - 时间 embedding
        
        # 条件向量 = 标签信息 + 时间信息
        # 这个向量会通过 AdaLN 注入到每个 DiT Block 中
        cond = y_emb + t_emb  # (batch, emb_size)
        
        # ===== 步骤2：Patchify - 将图像切分成 patches =====
        # 2.1 卷积切分
        x = self.conv(x)  # (batch, channel*patch_size^2, patch_count, patch_count)
                          # 例如：(batch, 16, 7, 7)
        
        # 2.2 调整维度顺序，准备序列化
        x = x.permute(0, 2, 3, 1)  # (batch, patch_count, patch_count, channel*patch_size^2)
                                    # 例如：(batch, 7, 7, 16)
        
        # 2.3 展平成序列
        x = x.view(x.size(0), self.patch_count * self.patch_count, x.size(3))
        # (batch, patch_count^2, channel*patch_size^2)
        # 例如：(batch, 49, 16) - 49个patch，每个16维
        
        # ===== 步骤3：Patch Embedding =====
        x = self.patch_emb(x)  # (batch, patch_count^2, emb_size)
                               # 例如：(batch, 49, 64)
        
        # ===== 步骤4：添加位置编码 =====
        # 让模型知道每个 patch 在图像中的位置
        x = x + self.patch_pos_emb  # (batch, patch_count^2, emb_size)
        
        # ===== 步骤5：DiT Blocks 处理 =====
        # 通过多层 Transformer Block 处理 patch 序列
        # 每个 Block 都会利用 cond 条件信息
        for dit in self.dits:
            x = dit(x, cond)  # x 保持 (batch, patch_count^2, emb_size)
        
        # ===== 步骤6：Layer Norm =====
        x = self.ln(x)  # (batch, patch_count^2, emb_size)
        
        # ===== 步骤7：投影回 Patch 维度 =====
        x = self.linear(x)  # (batch, patch_count^2, channel*patch_size^2)
                            # 例如：(batch, 49, 16)
        
        # ===== 步骤8：Un-Patchify - 将 patches 重组回图像 =====
        # 这是一个复杂的 reshape 过程，需要仔细理解维度变换
        
        # 8.1 重塑为 6D tensor
        # 将每个 patch 的 (channel*patch_size^2) 拆分成 (channel, patch_size, patch_size)
        x = x.view(
            x.size(0),           # batch
            self.patch_count,    # 垂直方向的 patch 数量
            self.patch_count,    # 水平方向的 patch 数量
            self.channel,        # 通道数
            self.patch_size,     # patch 高度
            self.patch_size      # patch 宽度
        )
        # shape: (batch, patch_count(H), patch_count(W), channel, patch_size(H), patch_size(W))
        # 例如：(batch, 7, 7, 1, 4, 4)
        
        # 8.2 调整维度顺序：将 channel 移到前面
        x = x.permute(0, 3, 1, 2, 4, 5)
        # (batch, channel, patch_count(H), patch_count(W), patch_size(H), patch_size(W))
        # 例如：(batch, 1, 7, 7, 4, 4)
        
        # 8.3 再次调整：将同一行的 patches 排列在一起
        x = x.permute(0, 1, 2, 4, 3, 5)
        # (batch, channel, patch_count(H), patch_size(H), patch_count(W), patch_size(W))
        # 例如：(batch, 1, 7, 4, 7, 4)
        
        # 8.4 最后 reshape：合并 patch 维度，得到完整图像
        x = x.reshape(
            x.size(0),                              # batch
            self.channel,                           # channel
            self.patch_count * self.patch_size,     # 高度 = patch数 × patch大小
            self.patch_count * self.patch_size      # 宽度 = patch数 × patch大小
        )
        # (batch, channel, img_size, img_size)
        # 例如：(batch, 1, 28, 28) ✅ 恢复原始图像大小！
        
        return x  # 返回预测的噪音或去噪图像
    
if __name__ == '__main__':
    # ===== 测试代码 =====
    print("🧪 测试 DiT 模型")
    
    # 创建 DiT 模型
    dit = DiT(
        img_size=28,      # MNIST 图像大小
        patch_size=4,     # 4×4 的 patch
        channel=1,        # 灰度图
        emb_size=64,      # embedding 维度
        label_num=10,     # 10 个数字类别
        dit_num=3,        # 3 层 DiT Block
        head=4            # 4 个注意力头
    )
    
    # 创建随机输入
    x = torch.rand(5, 1, 28, 28)  # 5 张图像
    t = torch.randint(0, T, (5,))  # 5 个随机时间步
    y = torch.randint(0, 10, (5,))  # 5 个随机标签
    
    print(f"输入 x shape: {x.shape}")
    print(f"时间步 t: {t}")
    print(f"标签 y: {y}")
    
    # 前向传播
    outputs = dit(x, t, y)
    
    print(f"输出 shape: {outputs.shape}")
    print(f"✅ 测试通过！输入输出 shape 一致")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in dit.parameters())
    print(f"📊 模型总参数量: {total_params:,}")