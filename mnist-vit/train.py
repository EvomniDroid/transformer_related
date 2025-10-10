import torch 
from dataset import MNIST
from vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备
print("1")
dataset=MNIST() # 数据集
print("2")
model=ViT().to(DEVICE) # 模型
print("3")
try:    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 
print("4")
optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

'''
    训练模型
'''
print("5")
if __name__=='__main__':
    EPOCH=50
    BATCH_SIZE=64   # 从batch内选出10个不一样的数字
    print("6")
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)  # 数据加载器，Windows下设置num_workers=0
    print("7")
    iter_count=0
    for epoch in range(EPOCH):
        print(f"Epoch {epoch+1}/{EPOCH}")
        for imgs,labels in dataloader:
            print(f"  Iteration {iter_count+1}")
            logits=model(imgs.to(DEVICE))
            print("8")
            loss=F.cross_entropy(logits,labels.to(DEVICE))
            print("9")
            optimzer.zero_grad()
            print("10")
            loss.backward()
            print("11")
            optimzer.step()
            print("12")
            if iter_count%1000==0:
                print("13")
                print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
                print("14")
                torch.save(model.state_dict(),'.model.pth')
                print("15")
                os.replace('.model.pth','model.pth')
                print("16")
            print("17")
            iter_count+=1
            print("18")
        print("19")
    print("20")