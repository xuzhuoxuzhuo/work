## Panoptic FPN

*本项目在 weizmann_horse数据集上完成语义分割任务，对《Panoptic Feature Pyramid Networks》中的ResNet-101-FPN网络进行简要复现-ResNet-18-FPN,并结合论文的思想搭建ResNet-18-BiFPN网络。




## 数据集
* weizmann_horse数据集，由327张horse组成。
            
![avatar](/markdownuse_pic/图片1.png)

* 数据集存放格式如下
```
weizmann_horse_db
├──── train_img
│    ├────  279 imgs
├──── train_gt
│    ├────  279 gt
├──── test_img
│    ├────  48 imgs 
├──── test_gt
│    ├────  48 gt
```
## Installation

   本人的代码已经在Python 3.9上使用CPU进行了测试。配置环境必需软件包请参阅requirements.txt。
   
   如需使用GPU训练，请确保您的python,torch,torchvison与cuda版本能够兼容。
   
   
## 推理预测

请运行以下命令对测试集的图像进行推理。
```
python test.py
```
   
请注意：

* test.py中数据集路径设置正确（本人数据集路径已经设置好，应该不用修改可以运行）

* 推理使用的网络模型见model文件夹中（默认使用的是model-resnet18-Bifpn，可在test.py第21-23行进行更换，直接运行test.py文件得到Miou=0.91,Biou=0.60的结果。)

* 运行代码后会在save文件夹中生成运行结果。
![avatar](/markdownuse_pic/图片2.png)

## 训练

请运行以下命令对训练集进行训练，训练的参数在train.py中进行设置。
```
python train.py
```
  
 model文件夹中已存放训练好的模型权重文件，训练参数为batchsize=4，epochs=100，learningrate=1e-4。
 
 请注意：
 
 * train.py中数据集路径设置正确（本人数据集路径已经设置好，应该不用修改可以运行）
 * 默认使用的网络模型为ResNet-18-BiFPN网络。可在train.py中第48-49行进行更换。
