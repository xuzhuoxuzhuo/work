##############################################################计算iou指标##################################################
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
'''
计算混淆矩阵
输入：预测结果，真实标签
输出：混淆矩阵
'''
def get_matrix(out,gt):
    out=out.reshape((224*224))
    gt = gt.reshape((224 * 224))
    matrix=confusion_matrix(gt,out)

    return matrix
'''
计算miou
输入：混淆矩阵
输出：miou值
'''
def get_mious(matrix):
    TP=matrix[1][1]
    FN=matrix[1][0]
    FP=matrix[0][1]
    TN=matrix[0][0]
    miou = (TP/(TP+FP+FN) + TN/(TN+FN+FP))/2 #0类1类的iou然后求平均
    return miou
'''
获取图像边界
输入：图像
输出：图像边缘
'''
def get_boundary(input):
    padding = cv2.copyMakeBorder(input, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0).astype(float) #224*224->226*226 补一圈0
    kernel = np.ones((11,11), np.uint8) #腐蚀核
    erode = cv2.erode(padding,kernel,6) #腐蚀
    erode_img = erode[1:input.shape[0] + 1, 1:input.shape[1]+ 1] #226*226->224*224
    #cv2.imshow('1',input-erode_img)
    #cv2.waitKey(0)
    return input-erode_img #原图与腐蚀图像作差得到图像边缘
'''
计算biou
输入：预测结果，真实标签
输出：biou
'''
def get_biou(out,gt):
    out=get_boundary(out)
    gt=get_boundary(gt)
    #这里得到了out与Gt的二值化boundary，计算交并比
    matrix=get_matrix(out,gt)
    TP=matrix[1][1]
    FN=matrix[1][0]
    FP=matrix[0][1]
    TN=matrix[0][0]
    biou = TP/(TP+FP+FN)
    #print(TP,FP,FN,biou)
    #print(biou)
    return biou
