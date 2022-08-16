#################################################数据集导入################################################################
'''
本文件中包含导入数据集的工具
'''
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import torch
import numpy as np
'''
输入：img与gt文件夹路径
输出：img与gt文件夹中所有图片的路径构成的列表
'''
def get_data_root(datadir,gtdir):
    img_root=[]#储存图像路径的列表
    gt_root=[]#储存标签的列表
    for root,_,PicName in os.walk(datadir): #提取datadir路径的根目录路径，以及里面包含的文件名
        for pic_name in PicName:
            img=os.path.join(root,pic_name) #组合根目录与文件名,构成文件的路径
            img_root.append(img)
    for root,_,GTName in os.walk(gtdir):#提取gtdir路径的根目录路径，以及里面包含的文件名
        for gt_name in GTName:
            gt=os.path.join(root,gt_name)#组合根目录与文件名,构成文件的路径
            gt_root.append(gt)
    return img_root,gt_root
'''
重写Dataset类
包括图像预处理与__getitem__读取数据
参数：
data_root img路径
gt_root gt路径
train：train=1,对训练集进行处理 train=0：对测试集进行处理
'''
class HorseDataset(Dataset):
    def __init__(self,data_root,gt_root,train):
        self.data=data_root
        self.gt=gt_root
        #对训练集图像预处理，包括resize与随机亮度，对比度，饱和度的数据增强
        if train==1:
            self.transformer=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.ToTensor()])
        else:
            # 测试集图像预处理，包括resize
            self.transformer = transforms.Compose([
                transforms.Resize((224, 224)),
                #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor()
        ])
            # 对标签的预处理，包括resize
        self.transformergt=transforms.Compose([
            transforms.Resize((224,224)),
        ])
        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):#读取一个样本的gt与img
        data_path=self.data[item]
        gt_path=self.gt[item]
        data = self.transformer(Image.open(data_path)) #图像预处理
        gt =torch.from_numpy(np.array(self.transformergt(Image.open(gt_path)))).type(torch.FloatTensor) #numpy转tensor，不要使用totensor，否则会改变标签值
        return data,gt
    pass