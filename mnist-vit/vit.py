from torch import nn 
import torch 
import pdb  # 调试工具
import os

# 🎛️ 调试开关：设置环境变量 DEBUG_INIT=1 来启用 __init__ 中的断点
DEBUG_INIT = os.getenv('DEBUG_INIT', '0') == '1'

class ViT(nn.Module):
    print("进入 ViT 类定义")
    def __init__(self,emb_size=16):
        super().__init__()
        
        # 🔍 断点1：查看初始化参数
        if DEBUG_INIT: pdb.set_trace()
        
        self.patch_size=4
        self.patch_count=28//self.patch_size # 7
        
        # 🔍 断点2：理解卷积层如何切分patch
        # Conv2d: 把28x28图像切成7x7个patch，每个patch是4x4=16个像素
        if DEBUG_INIT: pdb.set_trace()
        self.conv=nn.Conv2d(in_channels=1,out_channels=self.patch_size**2,kernel_size=self.patch_size,padding=0,stride=self.patch_size) # 图片转patch
        
        # 🔍 断点3：理解patch embedding
        # Linear: 把16维的patch展平向量映射到emb_size维
        if DEBUG_INIT: pdb.set_trace()
        self.patch_emb=nn.Linear(in_features=self.patch_size**2,out_features=emb_size)    # patch做emb
        
        # 🔍 断点4：理解CLS token和位置编码
        if DEBUG_INIT: pdb.set_trace()
        self.cls_token=nn.Parameter(torch.rand(1,1,emb_size))   # 分类头输入
        self.pos_emb=nn.Parameter(torch.rand(1,self.patch_count**2+1,emb_size))   # position位置向量 (1,seq_len,emb_size)
        
        # 🔍 断点5：理解Transformer Encoder结构
        if DEBUG_INIT: pdb.set_trace()
        #encoder 灰色部分堆三个
        self.tranformer_enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_size,nhead=2,batch_first=True),num_layers=3)   # transformer编码器
        self.cls_linear=nn.Linear(in_features=emb_size,out_features=10) # 手写数字10分类
        
    def forward(self,x): # (batch_size,channel=1,width=28,height=28)
        print("进入 ViT 前向传播")
        # 🔍 断点6：查看输入图像shape
        print(f"输入 x.shape: {x.shape}")  # 应该是 (batch, 1, 28, 28)
        pdb.set_trace()
        print("9.1")
        # 步骤1: 卷积切分成patch
        x=self.conv(x) # (batch_size,channel=16,width=7,height=7)
        
        # 🔍 断点7：查看卷积后的patch
        print(f"卷积后 x.shape: {x.shape}")  # 应该是 (batch, 16, 7, 7)
        pdb.set_trace()
        print("9.2")
        # 步骤2: 重塑为序列形式
        x=x.view(x.size(0),x.size(1),self.patch_count**2)   # (batch_size,channel=16,seq_len=49)
        
        # 🔍 断点8：查看view后的shape
        print(f"view后 x.shape: {x.shape}")  # 应该是 (batch, 16, 49)
        pdb.set_trace()
        print("9.3")
        x=x.permute(0,2,1)  # (batch_size,seq_len=49,channel=16)
        
        # 🔍 断点9：查看permute后，序列形式
        print(f"permute后 x.shape: {x.shape}")  # 应该是 (batch, 49, 16)
        pdb.set_trace()
        print("9.4")
        # 步骤3: patch embedding
        x=self.patch_emb(x)   # (batch_size,seq_len=49,emb_size)
        
        # 🔍 断点10：查看embedding后的特征
        print(f"patch_emb后 x.shape: {x.shape}")  # 应该是 (batch, 49, emb_size)
        pdb.set_trace()
        print("9.5")
        # 步骤4: 添加CLS token
        cls_token=self.cls_token.expand(x.size(0),1,x.size(2))  # (batch_size,1,emb_size)
        
        # 🔍 断点11：查看CLS token
        print(f"cls_token.shape: {cls_token.shape}")  # 应该是 (batch, 1, emb_size)
        pdb.set_trace()
        print("9.6")
        x=torch.cat((cls_token,x),dim=1)   # add [cls] token
        
        # 🔍 断点12：查看拼接CLS后的序列
        print(f"拼接CLS后 x.shape: {x.shape}")  # 应该是 (batch, 50, emb_size)
        pdb.set_trace()
        print("9.7")
        # 步骤5: 添加位置编码
        x=self.pos_emb+x
        
        # 🔍 断点13：查看加位置编码后
        print(f"加位置编码后 x.shape: {x.shape}")  # 应该是 (batch, 50, emb_size)
        pdb.set_trace()
        print("9.8")
        # 步骤6: Transformer编码
        y=self.tranformer_enc(x) # 不涉及padding，所以不需要mask
        
        # 🔍 断点14：查看Transformer输出
        print(f"Transformer后 y.shape: {y.shape}")  # 应该是 (batch, 50, emb_size)
        pdb.set_trace()
        print("9.9")
        # 步骤7: 取CLS token输出做分类
        cls_output = y[:,0,:]  # 取第一个token (CLS)
        
        # 🔍 断点15：查看CLS输出
        print(f"CLS输出 cls_output.shape: {cls_output.shape}")  # 应该是 (batch, emb_size)
        pdb.set_trace()
        print("9.10")
        logits = self.cls_linear(cls_output)
        
        # 🔍 断点16：查看最终分类logits
        print(f"最终logits.shape: {logits.shape}")  # 应该是 (batch, 10)
        print(f"预测类别: {logits.argmax(-1)}")
        pdb.set_trace()
        
        return logits   # 对[CLS] token输出做分类
    
if __name__=='__main__':
    vit=ViT()
    print("vit")
    x=torch.rand(5,1,28,28)
    y=vit(x)
    print(y.shape)