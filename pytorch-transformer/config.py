import torch 

# 设备 - 使用CPU
DEVICE = torch.device('cpu')
print(f"使用设备: {DEVICE}")

# 最长序列（受限于postition emb）
SEQ_MAX_LEN=5000