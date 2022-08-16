################################################模型文件############################################################
'''
本文件包括本次实验使用的所有网络模型
'''
from torch import nn
'''
Backbone:Resnet18,提取不同尺度特征图，用于后续FPN,BiFPN进行特征融合
输入：3*224*224的RGB图像
输出：四个尺度下的特征图
'''
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.relu= nn.ReLU()
        self.conv0 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=3,stride=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv1_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.bn2 = nn.BatchNorm2d(128)
        self.conv_64_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv2_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.bn3 = nn.BatchNorm2d(256)
        self.conv_128_256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)

        self.bn4 = nn.BatchNorm2d(512)
        self.conv_256_512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, stride=2)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv4_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)

        pass
    def forward(self,x): #前向传播
        #3*224*224->64*56*56
        x=self.conv0(x)
        x=self.bn0(x)
        x=self.maxpool0(x)
        #第一个残差块 得到64*56*56的特征图
        output1_1=self.conv1_1(x)
        output1_1=self.bn1(output1_1)
        output1_1=self.relu(output1_1)
        output1_1=self.conv1_2(output1_1)
        output1_1=self.bn1(output1_1)
        output1_1=self.relu(output1_1)
        output1_1=output1_1+x#残差结构

        output1_2=self.conv1_3(output1_1)
        output1_2=self.bn1(output1_2)
        output1_2=self.relu(output1_2)
        output1_2=self.conv1_4(output1_2)
        output1_2=self.bn1(output1_2)
        output1_2=self.relu(output1_2)
        output1_2=output1_2+output1_1

        # 第一个残差块 得到128*28*28的特征图
        output2_1=self.conv2_1(output1_2)
        output2_1=self.bn2(output2_1)
        output2_1=self.relu(output2_1)
        output2_1=self.conv2_2(output2_1)
        output2_1=self.bn2(output2_1)
        output2_1=self.relu(output2_1)
        output2_1=output2_1+self.bn2(self.conv_64_128(output1_2))
        output2_2=self.conv2_3(output2_1)
        output2_2=self.bn2(output2_2)
        output2_2=self.relu(output2_2)
        output2_2=self.conv2_4(output2_2)
        output2_2=self.bn2(output2_2)
        output2_2=self.relu(output2_2)
        output2_2=output2_2+output2_1

        # 第三个残差块 得到256*14*14的特征图
        output3_1=self.conv3_1(output2_2)
        output3_1=self.bn3(output3_1)
        output3_1=self.relu(output3_1)
        output3_1=self.conv3_2(output3_1)
        output3_1=self.bn3(output3_1)
        output3_1=self.relu(output3_1)
        output3_1=output3_1+self.bn3(self.conv_128_256(output2_2))
        output3_2=self.conv3_3(output3_1)
        output3_2=self.bn3(output3_2)
        output3_2=self.relu(output3_2)
        output3_2=self.conv3_4(output3_2)
        output3_2=self.bn3(output3_2)
        output3_2=self.relu(output3_2)
        output3_2=output3_2+output3_1

        # 第四个残差块 得到512*7*7的特征图
        output4_1=self.conv4_1(output3_2)
        output4_1=self.bn4(output4_1)
        output4_1=self.relu(output4_1)
        output4_1=self.conv4_2(output4_1)
        output4_1=self.bn4(output4_1)
        output4_1=self.relu(output4_1)
        output4_1=output4_1+self.bn4(self.conv_256_512(output3_2))
        output4_2=self.conv4_3(output4_1)
        output4_2=self.bn4(output4_2)
        output4_2=self.relu(output4_2)
        output4_2=self.conv4_4(output4_2)
        output4_2=self.bn4(output4_2)
        output4_2=self.relu(output4_2)
        output4_2=output4_2+output4_1

        return [output1_2,output2_2,output3_2,output4_2] #输出四个尺度下的特征图
    pass

'''
head:FPN网络,利用Backbone提取的特征图进行特征融合，生成像素级输出
输入：四个尺度下的特征图
输出：2*224*224的预测结果 其中第一个通道是像素属于标签0的概率，第二个通道是像素属于标签1的概率
'''
class Fpn(nn.Module):
    def __init__(self):
        super(Fpn, self).__init__()
        self.relu= nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv512to64 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=1,padding=0,stride=1)
        self.conv256to64 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv128to64 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv64to64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.convfinal = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1, stride=1)
        #双线性插值上采样
        self.upsample1 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
    def forward(self,input):
        feature4 = self.conv512to64(input[3])#1/32 64*7*7
        feature3 = self.conv256to64(input[2])#1/16 64*14*14
        feature2 = self.conv128to64(input[1])#1/8 64*28*28
        feature1 = self.conv64to64(input[0])#1/4 64*56*56

        #FPN自顶向下特征融合
        feature3=self.conv1(feature3+self.upsample(feature4))
        feature2 = self.conv1(feature2 + self.upsample(feature3))
        feature1 = self.conv1(feature1 + self.upsample(feature2))
        #进行conv,upsanple操作，将7*7，14*14，28*28大小的特征图上采样至56*56
        pyramid1 = self.conv2(feature1)

        pyramid2 = self.conv1(feature2)
        pyramid2 = self.upsample(pyramid2)
        pyramid2 = self.conv2(pyramid2)

        pyramid3 = self.conv1(feature3)
        pyramid3 = self.upsample(pyramid3)
        pyramid3 = self.conv1(pyramid3)
        pyramid3 = self.upsample(pyramid3)
        pyramid3 = self.conv2(pyramid3)

        pyramid4 = self.conv1(feature4)
        pyramid4 = self.upsample(pyramid4)
        pyramid4 = self.conv1(pyramid4)
        pyramid4 = self.upsample(pyramid4)
        pyramid4 = self.conv1(pyramid4)
        pyramid4 = self.upsample(pyramid4)
        pyramid4 = self.conv2(pyramid4)

        #所有特征图融合，并四倍上采样至224*224
        sum = self.conv3(pyramid4+pyramid3+pyramid2+pyramid1)
        sum =self.upsample1(sum)
        sum = self.upsample4(sum)
        sum = self.conv4(sum)
        #softmax输出2*224*224预测结果
        out = self.softmax(self.convfinal(sum))
        return out

'''
head:Bifpn网络,利用Backbone提取的特征图进行特征融合，生成像素级输出
输入：四个尺度下的特征图
输出：2*224*224的预测结果 其中第一个通道是像素属于标签0的概率，第二个通道是像素属于标签1的概率
'''
class Bifpn(nn.Module):
    def __init__(self):
        super(Bifpn, self).__init__()
        self.relu= nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv512to64 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=1,padding=0,stride=1)
        self.conv256to64 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv128to64 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv64to64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.convfinal = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1, stride=1)
        # 双线性插值上采样
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)

    def forward(self,input):

        feature4 = self.conv512to64(input[3])#1/32
        feature3 = self.conv256to64(input[2])#1/16
        feature2 = self.conv128to64(input[1])#1/8
        feature1 = self.conv64to64(input[0])#1/4

        feature3_1=self.conv1(feature3+self.upsample(feature4))
        feature2_1 = self.conv1(feature2 + self.upsample(feature3))
        feature1_1 = self.conv1(feature1 + self.upsample(feature2))
        #以上部分同FPN网络

        #加入PAN架构自底向上，同时与初始特征图再进行融合
        feature1=feature1_1
        feature2 = self.conv1(feature2_1+nn.functional.interpolate(feature1,scale_factor=0.5,recompute_scale_factor=True)+feature2)
        feature3 = self.conv1(feature3_1+nn.functional.interpolate(feature2,scale_factor=0.5,recompute_scale_factor=True)+feature3)
        feature4 = self.conv1(nn.functional.interpolate(feature3,scale_factor=0.5,recompute_scale_factor=True) + feature4)


        #以下部分同FPN网络
        pyramid1 = self.conv2(feature1)

        pyramid2 = self.conv1(feature2)
        pyramid2 = self.upsample(pyramid2)
        pyramid2 = self.conv2(pyramid2)

        pyramid3 = self.conv1(feature3)
        pyramid3 = self.upsample(pyramid3)
        pyramid3 = self.conv1(pyramid3)
        pyramid3 = self.upsample(pyramid3)
        pyramid3 = self.conv2(pyramid3)

        pyramid4 = self.conv1(feature4)
        pyramid4 = self.upsample(pyramid4)
        pyramid4 = self.conv1(pyramid4)
        pyramid4 = self.upsample(pyramid4)
        pyramid4 = self.conv1(pyramid4)
        pyramid4 = self.upsample(pyramid4)
        pyramid4 = self.conv2(pyramid4)

        sum = self.conv3(pyramid4+pyramid3+pyramid2+pyramid1)
        sum = self.upsample1(sum)
        sum = self.upsample4(sum)
        sum = self.conv4(sum)
        out = self.softmax(self.convfinal(sum))
        return out

'''
PanopticFPN网络
Backbone:Resnet18
Neck:FPN网络
输入：3*224*224RGB图像
输出：2*224*224的预测结果 其中第一个通道是像素属于标签0的概率，第二个通道是像素属于标签1的概率
'''
class PanopticFPN(nn.Module):
    def __init__(self):
        super(PanopticFPN, self).__init__()
        self.backbone = Resnet18()
        self.fpn = Fpn()
        pass
    def forward(self,x):
        x=self.backbone(x)
        x=self.fpn(x)
        return x
        pass
'''
BiFPN网络
Backbone:Resnet18
Neck:BiFPN网络
输入：3*224*224RGB图像
输出：2*224*224的预测结果 其中第一个通道是像素属于标签0的概率，第二个通道是像素属于标签1的概率
'''
class BiFPN(nn.Module):
    def __init__(self):
        super(BiFPN, self).__init__()
        self.backbone = Resnet18()
        self.bifpn = Bifpn()
        pass
    def forward(self,x):
        x=self.backbone(x)
        x=self.bifpn(x)
        return x
        pass
