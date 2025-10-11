"""
Time Embedding - 时间步的正弦位置编码

在扩散模型中，时间步 t 是一个非常重要的信息：
- t=0: 原始清晰图像
- t=T: 完全噪声
- 模型需要知道当前在哪个时间步，才能预测正确的噪音量

这个模块使用 Transformer 中的正弦位置编码方法将时间步编码为高维向量。

核心思想：
- 使用不同频率的正弦和余弦函数
- 低频部分捕捉粗粒度的时间信息
- 高频部分捕捉细粒度的时间变化
"""

import torch 
from torch import nn 
import math 
from config import T

class TimeEmbedding(nn.Module):
    """
    正弦位置编码（Sinusoidal Position Encoding）for 时间步
    
    公式：
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    
    其中：
        t: 时间步（0 到 T-1）
        i: 维度索引（0 到 emb_size/2-1）
        d: embedding 维度
    
    参数：
        emb_size: embedding 维度（必须是偶数）
    """
    def __init__(self, emb_size):
        super().__init__()
        
        self.half_emb_size = emb_size // 2  # 一半用 sin，一半用 cos
        
        # ===== 预计算频率权重 =====
        # 生成一组频率，从高频到低频
        # 公式：exp(-log(10000) * i / (half_emb_size - 1))
        #     = 10000^(-i / (half_emb_size - 1))
        #     = 1 / 10000^(i / (half_emb_size - 1))
        
        # i: [0, 1, 2, ..., half_emb_size-1]
        half_emb = torch.exp(
            torch.arange(self.half_emb_size) * 
            (-1 * math.log(10000) / (self.half_emb_size - 1))
        )
        # half_emb: [1.0, 0.xxx, 0.0xxx, ..., 0.0001]
        # 从 1 递减到接近 0，对应从高频到低频
        
        # 将频率权重注册为 buffer（不参与梯度更新，但会保存在模型中）
        self.register_buffer('half_emb', half_emb)

    def forward(self, t):
        """
        前向传播：将时间步编码为向量
        
        参数：
            t: 时间步，shape=(batch,)
               例如：[999, 500, 250, ...]
        
        返回：
            时间 embedding，shape=(batch, emb_size)
        """
        
        # ===== 步骤1：调整时间步形状 =====
        # (batch,) → (batch, 1)
        t = t.view(t.size(0), 1)
        
        # ===== 步骤2：扩展频率权重到 batch =====
        # (half_emb_size,) → (batch, half_emb_size)
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size)
        # 每个样本使用相同的频率权重
        
        # ===== 步骤3：计算 t * 频率 =====
        half_emb_t = half_emb * t  # (batch, half_emb_size)
        # 例如：t=999 时
        # half_emb_t[0] = [999.0, 999*0.xxx, 999*0.0xxx, ...]
        
        # ===== 步骤4：应用正弦和余弦函数 =====
        # sin 部分：捕捉周期性模式
        sin_emb = half_emb_t.sin()  # (batch, half_emb_size)
        
        # cos 部分：与 sin 正交，提供额外信息
        cos_emb = half_emb_t.cos()  # (batch, half_emb_size)
        
        # ===== 步骤5：拼接 sin 和 cos =====
        embs_t = torch.cat((sin_emb, cos_emb), dim=-1)  # (batch, emb_size)
        # 前半部分是 sin，后半部分是 cos
        
        return embs_t
        # 返回的 embedding 向量对每个时间步都是唯一的
        # 相近的时间步会有相似的 embedding
    
if __name__ == '__main__':
    # ===== 测试代码 =====
    print("🧪 测试 Time Embedding")
    
    time_emb = TimeEmbedding(16)
    
    # 随机生成 2 个时间步
    t = torch.randint(0, T, (2,))
    print(f"时间步 t: {t}")
    
    # 生成 embedding
    embs = time_emb(t) 
    print(f"Time embedding shape: {embs.shape}")
    print(f"Time embedding:\n{embs}")
    
    # ===== 可视化不同时间步的 embedding =====
    print("\n📊 不同时间步的 embedding 对比:")
    test_times = torch.tensor([0, 250, 500, 750, 999])
    for tt in test_times:
        emb = time_emb(tt.unsqueeze(0))
        print(f"t={tt:3d}: embedding 范围 [{emb.min():.3f}, {emb.max():.3f}], "
              f"均值 {emb.mean():.3f}")
    
    # ===== 验证相似性 =====
    print("\n🔍 验证时间步的相似性（余弦相似度）:")
    t1 = time_emb(torch.tensor([100]))
    t2 = time_emb(torch.tensor([105]))  # 相近的时间步
    t3 = time_emb(torch.tensor([900]))  # 相距很远的时间步
    
    cos_sim = nn.CosineSimilarity(dim=1)
    sim_near = cos_sim(t1, t2)
    sim_far = cos_sim(t1, t3)
    
    print(f"t=100 vs t=105 (相近): {sim_near.item():.4f}")
    print(f"t=100 vs t=900 (相远): {sim_far.item():.4f}")
    print("✅ 相近的时间步应该有更高的相似度")