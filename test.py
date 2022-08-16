#################################################测试文件################################################################
'''
使用本文件导入训练模型对测试集进行预测
'''
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image
from model import PanopticFPN,BiFPN
import torch
import cv2
import numpy as np
from dataset import HorseDataset,get_data_root
import random
from iou import get_biou,get_matrix,get_boundary,get_mious

def test():
    '''
    测试文件目录
    '''
    val_datadir = r'weizmann_horse_db/val_img'
    val_gtdir = r'weizmann_horse_db/val_gt'
    modelroot='model/model-resnet18-Bifpn.pth'
    model = BiFPN()  # 初始化模型
    #model = PanopticFPN()     #对于路径'model/model-resnet18-fpn.pth'
    '''
    保存测试文件结果的文件夹名
    '''
    save='save'
    #使用同训练文件的随机数种子
    seed = 666
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    batchsize=1#batchsize,不用改

    #导入模型
    model.load_state_dict(torch.load(modelroot,map_location=torch.device('cpu')),strict=False)
    #导入验证集
    valdata_root,valgt_root=get_data_root(val_datadir,val_gtdir)
    ValHorse_Dataset = HorseDataset(valdata_root,valgt_root,train=0)
    ValHorse_Dataloader = DataLoader(ValHorse_Dataset,batch_size=batchsize,shuffle=False)
    ##################################预测结果#########################################
    '''
    保存验证集预测图片结果输出miou,biou指标
    '''
    print('保存图片预测结果')
    batchsize=1#batchsize,不用改
    idx=0#计算有多少个样本
    mious=0#miou指标
    bious=0#biou指标
    for x, y in ValHorse_Dataloader:
        idx = idx + 1
        output = model(x)#获得模型输出
        print('保存进度:', idx * batchsize, ':48')
        img = x[0]
        pre = output[0][1] #二分类问题只需要第二通道的结果就行，第二通道的像素值大于0.5便说明属于1类
        save_image(img, save + '/' + str(idx) + 'img.png')#保存原图
        '''
        将输出与gt从tensor转换成Numpy，并将输出四舍五入（大于0.5归为1类）
        '''
        pre=pre.detach().numpy()
        pre=np.around(pre)
        y=y[0].numpy()
        y=y.astype(int)
        '''
        计算miou,biou指标
        '''
        matrix=get_matrix(pre,y)
        miou=get_mious(matrix)
        biou=get_biou(pre,y)
        bious=bious+biou
        mious=mious+miou
        print(miou)
        cv2.imwrite(save+'/'+str(idx)+'img_gt.png',y*255)#保存gt
        cv2.imwrite(save+'/'+str(idx)+'pre.png',pre*255)#保存预测图像
    mious=mious/idx
    bious=bious/idx
    print('mious:',mious,'bious:',bious)
if __name__ == '__main__':
    test()