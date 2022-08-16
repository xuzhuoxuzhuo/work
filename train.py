#################################################训练文件################################################################
'''
使用本文件进行训练
'''

from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model import PanopticFPN,BiFPN
import random
import torch
import torch.nn as nn
import numpy as np
from dataset import get_data_root,HorseDataset
from iou import get_biou,get_matrix,get_boundary,get_mious
###########################################参数######################################
UseGpu=0 #UseGpu=0,使用CPU训练，UseGpu=1,使用GPU训练
batchsize=4#Batchsize
epochs=100#epoch
learningrate=1e-4#学习率
seed=666 #随机数种子，确保算法可以复现，无需更改
#save='save'
#数据集路径
train_datadir = r'weizmann_horse_db\train_img' #原始图像训练集
train_gtdir = r'weizmann_horse_db\train_gt' #训练集标签
val_datadir = r'weizmann_horse_db\val_img' #原始图像测试集
val_gtdir = r'weizmann_horse_db\val_gt' #测试集标签
print('导入数据集...')
#设置随机种子，保证训练结果具有可复现性
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
################################导入数据集###########################################
#将所有训练集图像与标签存至列表
traindata_root,traingt_root=get_data_root(train_datadir,train_gtdir)
#生成训练集dataloader
TrainHorse_Dataset = HorseDataset(traindata_root,traingt_root,train=1)
TrainHorse_Dataloader = DataLoader(TrainHorse_Dataset,batch_size=batchsize,shuffle=True)
#将所有测试集图像与标签存至列表
valdata_root,valgt_root=get_data_root(val_datadir,val_gtdir)
#生成测试集dataloader
ValHorse_Dataset = HorseDataset(valdata_root,valgt_root,train=0)
ValHorse_Dataloader = DataLoader(ValHorse_Dataset,batch_size=1,shuffle=True)
print('导入成功')
#################################Train##############################################
model=BiFPN()  #实例化网络模型
#model=PanopticFPN()
Loss = nn.CrossEntropyLoss() #使用交叉熵损失
if UseGpu==1:
    model=model.cuda()
    Loss=Loss.cuda()
loss_history = []#保存训练集损失
test_loss_history = []#保存测试集损失
train_mious_history=[]#保存训练集mious
test_mious_history=[]#保存测试集mious
train_bious_history=[]#保存训练集mious
test_bious_history=[]#保存测试集mious
epochlist=[]
optimizer = torch.optim.Adam(model.parameters(),lr=learningrate)#Adam优化器
#开始训练
for epoch in range(0,epochs):
    print('Epoch:',epoch+1,'开始训练')
    #用于求样本数
    idx=0
    idx1=0
    #指标初始化
    train_mious=0
    test_mious = 0
    train_bious=0
    test_bious=0
    trainloss=0
    testloss=0
    for x,y in TrainHorse_Dataloader:
        idx1=idx1+1
        if UseGpu == 1:
            x=x.cuda()
            y=y.cuda()
        optimizer.zero_grad()#清空梯度
        output = model(x)#得到输出
        y_=y
        y=y.long()#计算交叉熵损失需要转化成long
        loss = Loss(output,y)#计算损失
        trainloss+=loss.item()
        loss.backward()#反向传播
        optimizer.step()#梯度更新
        idx_=idx+batchsize
        if idx_>279:
            idx_=279
        print('训练进度:',idx_,':279')
        #对一个batch的图像计算miou,biou
        for i in range(0,x.shape[0]):
            idx = idx + 1
            img = x[i]
            pre = output[i][1]#二分类问题只需要第二通道的结果就行，第二通道的像素值大于0.5便说明属于1类
            #save_image(img, save + '/' + str(idx) + 'img.png')
            #save_image(pre, save + '/' + str(idx)+ 'pre.png')
            '''
            将输出与gt从tensor转换成Numpy，并将输出四舍五入（大于0.5归为1类）
            '''
            pre = pre.cpu().detach().numpy()
            pre = np.around(pre)
            y=y_
            y = y[i].cpu()
            y = y.numpy()
            y = y.astype(int)
            '''
             计算miou,biou指标
            '''
            matrix = get_matrix(pre, y)
            miou = get_mious(matrix)
            biou = get_biou(pre, y)
            train_mious = train_mious + miou
            train_bious=train_bious+biou
    trainloss=trainloss/idx1
    train_mious=train_mious/idx_
    train_bious=train_bious/idx_
    idx=0
    print('Epoch:',epoch+1,'开始测试')
    '''
    测试，与训练相同，少了梯度更新操作不加赘述
    '''
    for x,y in ValHorse_Dataloader:
        if UseGpu == 1:
            x=x.cuda()
            y=y.cuda()
        idx=idx+1
        output = model(x)
        y_=y
        y=y.long()
        loss = Loss(output,y)
        testloss+=loss.item()
        print('测试进度:',idx,':48')
        img = x[0]
        pre = output[0][1]
        pre = pre.cpu().detach().numpy()
        pre = np.around(pre)
        y=y_
        y = y[0].cpu()
        y=y.numpy()
        y = y.astype(int)
        matrix = get_matrix(pre, y)
        miou = get_mious(matrix)
        biou=get_biou(pre,y)
        test_bious=test_bious+biou
        test_mious = test_mious + miou
    testloss=testloss/idx
    test_mious=test_mious/idx
    test_bious=test_bious/idx
    print('trainloss:',trainloss,'testloss',testloss)
    #储存loss miou biou 用于后续画图
    loss_history.append(trainloss)
    test_loss_history.append(testloss)
    train_mious_history.append(train_mious)
    test_mious_history.append(test_mious)
    train_bious_history.append(train_bious)
    test_bious_history.append(test_bious)
    epochlist.append(epoch + 1)
###################################保存结果#################################################
print('训练结束，保存模型')
torch.save(model.state_dict(), 'model3.pth')
'''保存loss，miou,biou'''
plt.figure()
plt.title('trainloss')
plt.plot(epochlist,loss_history)
plt.savefig('trainloss.png',dpi=300)

plt.figure()
plt.title('testloss')
plt.plot(epochlist,test_loss_history)
plt.savefig('testloss.png',dpi=300)

plt.figure()
plt.title('trainmiou')
plt.plot(epochlist,train_mious_history)
plt.savefig('trainmiou.png',dpi=300)

plt.figure()
plt.title('testmiou')
plt.plot(epochlist,test_mious_history)
plt.savefig('testmiou.png',dpi=300)

plt.figure()
plt.title('trainbiou')
plt.plot(epochlist,train_bious_history)
plt.savefig('trainbiou.png',dpi=300)

plt.figure()
plt.title('testbiou')
plt.plot(epochlist,test_bious_history)
plt.savefig('testbiou.png',dpi=300)