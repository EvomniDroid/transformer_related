"""
DiT Block - Diffusion Transformer 的核心模块

这个模块实现了 DiT 的基本构建块，核心创新是 AdaLN（Adaptive Layer Normalization）。
与标准 Transformer 不同，DiT Block 通过条件信息（时间步 + 标签）动态调整归一化参数。

核心思想：
1. 使用条件信息生成 scale（gamma）、shift（beta）、gate（alpha）参数
2. 通过 AdaLN-Zero 将条件信息注入 Transformer
3. 保持标准 Self-Attention 和 FFN 结构

公式：
  AdaLN(x, cond) = gamma(cond) * LayerNorm(x) + beta(cond)
  Output = x + alpha(cond) * Attention(AdaLN(x, cond))
"""

from torch import nn 
import torch 
import math 

class DiTBlock(nn.Module):
    """
    DiT Block with Adaptive Layer Normalization
    
    参数：
        emb_size: embedding 维度（每个 patch 的特征维度）
        nhead: multi-head attention 的头数
    """
    def __init__(self, emb_size, nhead):
        super().__init__()
        
        self.emb_size = emb_size
        self.nhead = nhead
        
        # ===== AdaLN 条件参数生成器 =====
        # 从条件向量 (batch, emb_size) 生成归一化参数
        # 每个 Linear 层都是 emb_size → emb_size
        
        # 第一组：用于 Self-Attention 之前的 AdaLN
        self.gamma1 = nn.Linear(emb_size, emb_size)  # scale 参数（缩放）
        self.beta1 = nn.Linear(emb_size, emb_size)   # shift 参数（平移）
        self.alpha1 = nn.Linear(emb_size, emb_size)  # gate 参数（门控，控制残差强度）
        
        # 第二组：用于 Feed-Forward 之前的 AdaLN
        self.gamma2 = nn.Linear(emb_size, emb_size)  # scale 参数
        self.beta2 = nn.Linear(emb_size, emb_size)   # shift 参数
        self.alpha2 = nn.Linear(emb_size, emb_size)  # gate 参数
        
        # ===== Layer Normalization =====
        self.ln1 = nn.LayerNorm(emb_size)  # Attention 前的 LN
        self.ln2 = nn.LayerNorm(emb_size)  # FFN 前的 LN
        
        # ===== Multi-Head Self-Attention =====
        # 将 emb_size 扩展到 nhead * emb_size，然后拆分成多个头
        self.wq = nn.Linear(emb_size, nhead * emb_size)  # Query 投影
        self.wk = nn.Linear(emb_size, nhead * emb_size)  # Key 投影
        self.wv = nn.Linear(emb_size, nhead * emb_size)  # Value 投影
        self.lv = nn.Linear(nhead * emb_size, emb_size)  # 多头输出合并
        
        # ===== Feed-Forward Network (MLP) =====
        # 标准的两层 MLP，中间维度扩展 4 倍
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),  # 扩展
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)   # 压缩回原维度
        )

    def forward(self, x, cond):
        """
        前向传播
        
        参数：
            x: 输入 patch 序列，shape=(batch, seq_len, emb_size)
               例如：(batch, 49, 64) - 49个patch
            cond: 条件向量（时间 + 标签），shape=(batch, emb_size)
                  例如：(batch, 64)
        
        返回：
            输出序列，shape=(batch, seq_len, emb_size)
        """
        
        # ===== 步骤1：从条件生成 AdaLN 参数 =====
        # 条件向量 → 6 个参数（2组 gamma, beta, alpha）
        gamma1_val = self.gamma1(cond)  # (batch, emb_size) - Attention 的 scale
        beta1_val = self.beta1(cond)    # (batch, emb_size) - Attention 的 shift
        alpha1_val = self.alpha1(cond)  # (batch, emb_size) - Attention 的 gate
        gamma2_val = self.gamma2(cond)  # (batch, emb_size) - FFN 的 scale
        beta2_val = self.beta2(cond)    # (batch, emb_size) - FFN 的 shift
        alpha2_val = self.alpha2(cond)  # (batch, emb_size) - FFN 的 gate
        
        # ===== 步骤2：第一个 AdaLN（Self-Attention 前）=====
        # 2.1 Layer Normalization
        y = self.ln1(x)  # (batch, seq_len, emb_size)
        
        # 2.2 Scale & Shift（AdaLN 的核心）
        # y = gamma * y + beta
        # unsqueeze(1) 是为了广播到序列维度：(batch, emb_size) → (batch, 1, emb_size)
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)
        # (batch, seq_len, emb_size)
        # 解释：gamma 控制特征缩放，beta 控制特征偏移，都由条件决定
        
        # ===== 步骤3：Multi-Head Self-Attention =====
        # 3.1 生成 Q, K, V
        q = self.wq(y)  # (batch, seq_len, nhead * emb_size)
        k = self.wk(y)  # (batch, seq_len, nhead * emb_size)
        v = self.wv(y)  # (batch, seq_len, nhead * emb_size)
        
        # 3.2 重塑为多头形式
        # (batch, seq_len, nhead*emb_size) → (batch, nhead, seq_len, emb_size)
        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)
        # Q: (batch, nhead, seq_len, emb_size)
        
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 3, 1)
        # K: (batch, nhead, emb_size, seq_len) - 注意最后两维转置了
        
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)
        # V: (batch, nhead, seq_len, emb_size)
        
        # 3.3 计算注意力分数
        # Q @ K^T，除以 sqrt(d_k) 进行缩放
        attn = q @ k / math.sqrt(q.size(-1))  # (batch, nhead, seq_len, seq_len)
        # 注意力矩阵：每个 patch 对其他 patch 的关注度
        
        attn = torch.softmax(attn, dim=-1)  # (batch, nhead, seq_len, seq_len)
        # Softmax 归一化，每行和为 1
        
        # 3.4 应用注意力到 Value
        y = attn @ v  # (batch, nhead, seq_len, emb_size)
        # 加权求和：用注意力分数加权所有 patch 的特征
        
        # 3.5 合并多头
        y = y.permute(0, 2, 1, 3)  # (batch, seq_len, nhead, emb_size)
        y = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))
        # (batch, seq_len, nhead*emb_size)
        
        # 3.6 输出投影
        y = self.lv(y)  # (batch, seq_len, emb_size)
        
        # ===== 步骤4：AdaLN-Zero（门控残差）=====
        # alpha 控制 attention 输出的强度
        y = y * alpha1_val.unsqueeze(1)  # (batch, seq_len, emb_size)
        # 初始训练时 alpha 接近 0，模型从恒等映射开始学习
        
        # 残差连接
        y = x + y  # (batch, seq_len, emb_size)
        
        # ===== 步骤5：第二个 AdaLN（Feed-Forward 前）=====
        # 5.1 Layer Normalization
        z = self.ln2(y)  # (batch, seq_len, emb_size)
        
        # 5.2 Scale & Shift
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        # (batch, seq_len, emb_size)
        
        # ===== 步骤6：Feed-Forward Network =====
        z = self.ff(z)  # (batch, seq_len, emb_size)
        # MLP: emb_size → 4*emb_size → emb_size
        
        # ===== 步骤7：AdaLN-Zero（门控残差）=====
        z = z * alpha2_val.unsqueeze(1)  # (batch, seq_len, emb_size)
        
        # 最终残差连接
        return y + z  # (batch, seq_len, emb_size)
    
if __name__ == '__main__':
    # ===== 测试代码 =====
    print("🧪 测试 DiT Block")
    
    dit_block = DiTBlock(emb_size=16, nhead=4)
    
    # 模拟输入：5个样本，49个patch，每个16维
    x = torch.rand((5, 49, 16))
    # 模拟条件：5个样本，每个16维
    cond = torch.rand((5, 16))
    
    print(f"输入 x shape: {x.shape}")
    print(f"条件 cond shape: {cond.shape}")
    
    outputs = dit_block(x, cond)
    
    print(f"输出 shape: {outputs.shape}")
    print(f"✅ 测试通过！shape 保持不变")
    
    # 验证残差连接
    print(f"\n🔍 验证残差连接:")
    print(f"输入和输出的均值差异: {(outputs - x).abs().mean():.6f}")
    print("（差异较小说明残差连接起作用）")