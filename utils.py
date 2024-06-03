import numpy as np
import skimage
from skimage import io, measure
import random
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from preclassify import del2, srad, dicomp, FCM, hcluster
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from collections import  Counter


def image_normalize(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def image_padding(data,r):
    if len(data.shape)==3:
        data_new=np.lib.pad(data,((r,r),(r,r),(0,0)),'constant',constant_values=0)
        return data_new
    if len(data.shape)==2:
        data_new=np.lib.pad(data,r,'constant',constant_values=0)
        return data_new


#生成自然数数组并打乱
def arr(length):
  arr=np.arange(length-1)
  #print(arr)
  random.shuffle(arr)
  #print(arr)
  return arr


# 在每个像素周围提取 patch ，然后创建成符合 pytorch 处理的格式
# 把代码改为：生成一个mask，其中哪些被选中的标记为1，没被选中的标记为0.

def createTrainingCubes(X, y, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = image_padding(X, margin)
    # 把类别 uncertainty 的像素忽略
    ele_num1 = np.sum(y == 1)
    ele_num2 = np.sum(y == 2)
    patchesData_1 = np.zeros((ele_num1, patch_size, patch_size, X.shape[2]))
    patchesLabels_1 = np.zeros(ele_num1)

    patchesData_2 = np.zeros((ele_num2, patch_size, patch_size, X.shape[2]))
    patchesLabels_2 = np.zeros(ele_num2)

    patchIndex_1 = 0
    patchIndex_2 = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            # remove uncertainty pixels
            if y[r - margin, c - margin] == 1:
                patch_1 = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData_1[patchIndex_1, :, :, :] = patch_1
                patchesLabels_1[patchIndex_1] = y[r - margin, c - margin]
                patchIndex_1 = patchIndex_1 + 1
            elif y[r - margin, c - margin] == 2:
                patch_2 = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData_2[patchIndex_2, :, :, :] = patch_2
                patchesLabels_2[patchIndex_2] = y[r - margin, c - margin]
                patchIndex_2 = patchIndex_2 + 1
    patchesLabels_1 = patchesLabels_1 - 1
    patchesLabels_2 = patchesLabels_2 - 1

    # 调用arr函数打乱数组
    arr_1 = arr(len(patchesData_1))
    arr_2 = arr(len(patchesData_2))

    train_len = 10000  # 设置训练集样本数
    pdata = np.zeros((train_len, patch_size, patch_size, X.shape[2]))
    plabels = np.zeros(train_len)

    for i in range(7000):
        pdata[i, :, :, :] = patchesData_1[arr_1[i], :, :, :]
        plabels[i] = patchesLabels_1[arr_1[i]]
    for j in range(7000, train_len):
        pdata[j, :, :, :] = patchesData_2[arr_2[j - 7000], :, :, :]
        plabels[j] = patchesLabels_2[arr_2[j - 7000]]

    return pdata, plabels


def createTestingCubes(X, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = image_padding(X, margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], patch_size, patch_size, X.shape[2]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patchesData


#定义一个类，分别保存所有的数据
#生成高可信度训练集
#uncertain 训练集
#以及最后的hard sample 训练集
class createCubes:
    def __init__(self, X, label, patch_size):
        self.img = X
        self.label = label
        self.patch_size = patch_size
        self.h, self.w = label.shape[0], label.shape[1]
        self.all_data_len = label.shape[0] * label.shape[1]

        margin = int((self.patch_size - 1) / 2)
        zeroPaddedX = image_padding(X, margin)
        patchesData = np.zeros((X.shape[0] * X.shape[1], patch_size, patch_size, X.shape[2]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchIndex = patchIndex + 1

        self.label_vector = self.label.reshape(1,-1).squeeze()
        self.patchesData = patchesData

    '''
    def createTestCubes(self):      
        #这种定义方式，会报错，因为其中包含了等于1.5的标签
        #下面改为全部都是0和1的标签
        return self.patchesData, self.label_vector
    '''

    def createTestCubes(self):
        label_1 = np.array(np.where(self.label_vector == 1)).squeeze()
        label_2 = np.array(np.where(self.label_vector == 2)).squeeze()
        pdata = np.zeros((len(label_1) + len(label_2), self.patch_size, self.patch_size, self.img.shape[2]))
        plabels = np.zeros(len(label_1) + len(label_2))

        pdata[:len(label_1)] = self.patchesData[label_1]
        plabels[:len(label_1)] = 0

        pdata[len(label_1):] = self.patchesData[label_2]
        plabels[len(label_1):] = 1

        return pdata, plabels

    def createHCTrainingCubes(self,unchange_num= 7000, change_num = 3000):
        #创建高可信度的训练数据
        self.unchange_hctrain_len = unchange_num
        self.change_hctrain_len = change_num

        label_1 = np.array(np.where(self.label_vector == 1)).squeeze()
        label_2 = np.array(np.where(self.label_vector == 2)).squeeze()

        label_1_index = np.random.choice(label_1, size=unchange_num, replace=False)
        label_2_index = np.random.choice(label_2, size=change_num, replace=False)
        self.unchange_hctrain_index = label_1_index
        self.change_hctrain_index = label_2_index

        pdata = np.zeros((unchange_num + change_num, self.patch_size, self.patch_size, self.img.shape[2]))
        plabels = np.zeros(unchange_num + change_num)

        '''
        # 打乱数组 # new: 不需要乱序，dataloader加载的时候选择乱序加载即可
        shuffle_ix_label_1 = np.random.permutation(np.arange(len(label_1_index)))
        shuffle_ix_label_2 = np.random.permutation(np.arange(len(label_2_index)))

        #数据需要shuffle
        pdata[:unchange_num] = self.patchesData[label_1_index[shuffle_ix_label_1]]
        plabels[:unchange_num] = 0

        pdata[unchange_num:] = self.patchesData[label_2_index[shuffle_ix_label_2]]
        plabels[unchange_num:] = 1
        '''
        pdata[:unchange_num] = self.patchesData[label_1_index]
        plabels[:unchange_num] = 0

        pdata[unchange_num:] = self.patchesData[label_2_index]
        plabels[unchange_num:] = 1

        return pdata, plabels

    def createUncertainTrainingCubes(self):
        self.label_uncertain_index = np.where(self.label_vector == 1.5)
        self.uncertain_train_len = len(self.label_uncertain_index)

        #pdata = np.zeros((self.uncertain_train_len, self.patch_size, self.patch_size, self.img.shape[2]))
        pdata = np.array(self.patchesData[self.label_uncertain_index])

        return pdata

    def createHardSampleCubes(self, index, newHCtrainCube=True):
        #传入的index必须是一维的，或者在使用前改为一维的
        self.hard_smaple_index = index

        #phard = np.zeros((len(self.hard_smaple_index), self.patch_size, self.patch_size, self.img.shape[2]))
        phard  = self.patchesData[self.hard_smaple_index]
        p_hard_label = np.zeros(len(self.hard_smaple_index))

        if not newHCtrainCube:
            return phard, p_hard_label

        if newHCtrainCube:
            #hc_to_unertain = np.intersect1d(self.hard_smaple_index, self.change_hctrain_index)
            change_hctrain_index_new = list(set(self.change_hctrain_index) - set(self.hard_smaple_index))
            self.change_hctrain_index = change_hctrain_index_new

            pdata = np.zeros((len(self.unchange_hctrain_index) + len(self.change_hctrain_index) +
                              len(self.hard_smaple_index),
                              self.patch_size, self.patch_size, self.img.shape[2]))

            plabels = np.zeros((len(self.unchange_hctrain_index) + len(self.change_hctrain_index) + len(self.hard_smaple_index)))

            pdata[:len(self.unchange_hctrain_index)] = self.patchesData[self.unchange_hctrain_index]
            plabels[:len(self.unchange_hctrain_index)] = 0

            pdata[len(self.unchange_hctrain_index):(len(self.unchange_hctrain_index) + len(self.change_hctrain_index))] = self.patchesData[self.change_hctrain_index]
            plabels[len(self.unchange_hctrain_index):(len(self.unchange_hctrain_index) + len(self.change_hctrain_index))] = 1

            pdata[(len(self.unchange_hctrain_index) + len(self.change_hctrain_index)):] = self.patchesData[self.hard_smaple_index]
            plabels[(len(self.unchange_hctrain_index) + len(self.change_hctrain_index)):] = 0

            return pdata, plabels


#  Inputs:  gtImg  = ground truth image
#           tstImg = change map
#  Outputs: FA  = False alarms
#           MA  = Missed alarms
#           OE  = Overall error
#           PCC = Overall accuracy




def evaluate(gtImg, tstImg):
    gtImg[np.where(gtImg > 128)] = 255
    gtImg[np.where(gtImg < 128)] = 0
    tstImg[np.where(tstImg > 128)] = 255
    tstImg[np.where(tstImg < 128)] = 0
    [ylen, xlen] = gtImg.shape
    FA = 0.0
    MA = 0.0
    label_0 = np.sum(gtImg == 0)
    label_1 = np.sum(gtImg == 255)
    print(label_0)
    print(label_1)

    for j in range(0, ylen):
        for i in range(0, xlen):
            if gtImg[j, i] == 0 and tstImg[j, i] != 0:
                FA = FA + 1
            if gtImg[j, i] != 0 and tstImg[j, i] == 0:
                MA = MA + 1

    OE = FA + MA
    PCC = 1 - OE / (ylen * xlen)
    PRE = ((label_1 + FA - MA) * label_1 + (label_0 + MA - FA) * label_0) / ((ylen * xlen) * (ylen * xlen))
    KC = (PCC - PRE) / (1 - PRE)
    print(' Change detection results ==>')
    print(' ... ... FP:  ', FA)
    print(' ... ... FN:  ', MA)
    print(' ... ... OE:  ', OE)
    print(' ... ... PCC: ', format(PCC * 100, '.2f'))
    print(' ... ... KC: ', format(KC * 100, '.2f'))


def postprocess(res, connectiivity_thresh=20):
    res_new = res.copy()
    res = measure.label(res, connectivity=2)
    num = res.max()
    count = 0
    for i in range(1, num + 1):
        idy, idx = np.where(res == i)
        if len(idy) <= connectiivity_thresh:
            res_new[idy, idx] = 0
            count += 1
    print("in postprocess count is {}".format(count))
    return res_new

#应该区域联通区域的性质！！！


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




class createCubes_iter:
    def __init__(self, X, label, patch_size, args):
        #label_initial 中包含三类标签 1, 1.5, 2  (未变化、uncertain、变化)
        self.args = args
        self.img = X
        self.label_initial_2d = label
        self.patch_size = patch_size
        self.h, self.w = label.shape[0], label.shape[1]
        self.all_data_len = label.shape[0] * label.shape[1]

        margin = int((self.patch_size - 1) / 2)
        zeroPaddedX = image_padding(X, margin)
        patchesData = np.zeros((X.shape[0] * X.shape[1], patch_size, patch_size, X.shape[2]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchIndex = patchIndex + 1

        self.label_initial_1d = self.label_initial_2d.reshape(1,-1).squeeze()
        self.label_current_1d = self.label_initial_1d
        self.patchesData = patchesData

    '''
    def createTestCubes(self):      
        #这种定义方式，会报错，因为其中包含了等于1.5的标签, 计算loss时候会出现不匹配情况。
        #下面改为全部都是0和1的标签
        return self.patchesData, self.label_vector
    '''

    def createTestCubes(self):
        label_1 = np.array(np.where(self.label_current_1d == 1)).squeeze()
        label_2 = np.array(np.where(self.label_current_1d == 2)).squeeze()
        pdata = np.zeros((len(label_1) + len(label_2), self.patch_size, self.patch_size, self.img.shape[2]))
        plabels = np.zeros(len(label_1) + len(label_2))

        pdata[:len(label_1)] = self.patchesData[label_1]
        plabels[:len(label_1)] = 0

        pdata[len(label_1):] = self.patchesData[label_2]
        plabels[len(label_1):] = 1

        return pdata, plabels

    def createHCTrainingCubes(self,unchange_num= 7000, change_num = 3000):
        #创建高可信度的训练数据
        self.unchange_hctrain_len = unchange_num
        self.change_hctrain_len = change_num

        label_1 = np.array(np.where(self.label_current_1d == 1)).squeeze()
        label_2 = np.array(np.where(self.label_current_1d == 2)).squeeze()

        label_1_index = np.random.choice(label_1, size=unchange_num, replace=False)
        label_2_index = np.random.choice(label_2, size=change_num, replace=False)

        '''
        self.unchange_hctrain_index = label_1_index
        self.change_hctrain_index = label_2_index
        '''

        pdata = np.zeros((unchange_num + change_num, self.patch_size, self.patch_size, self.img.shape[2]))
        plabels = np.zeros(unchange_num + change_num)

        '''
        # 打乱数组 # new: 不需要乱序，dataloader加载的时候选择乱序加载即可
        shuffle_ix_label_1 = np.random.permutation(np.arange(len(label_1_index)))
        shuffle_ix_label_2 = np.random.permutation(np.arange(len(label_2_index)))

        #数据需要shuffle
        pdata[:unchange_num] = self.patchesData[label_1_index[shuffle_ix_label_1]]
        plabels[:unchange_num] = 0

        pdata[unchange_num:] = self.patchesData[label_2_index[shuffle_ix_label_2]]
        plabels[unchange_num:] = 1
        '''
        pdata[:unchange_num] = self.patchesData[label_1_index]
        plabels[:unchange_num] = 0

        pdata[unchange_num:] = self.patchesData[label_2_index]
        plabels[unchange_num:] = 1

        return pdata, plabels

    def createUncertainTrainingCubes(self):
        #需要传入使用的是哪个阶段的uncertain label
        self.label_uncertain_index = np.where(self.label_current_1d == 1.5)
        self.uncertain_train_len = len(self.label_current_1d)

        #pdata = np.zeros((self.uncertain_train_len, self.patch_size, self.patch_size, self.img.shape[2]))
        pdata = np.array(self.patchesData[self.label_uncertain_index])

        return pdata

    def createHardSampleCubes(self, hard_index, new_label, newHCtrainCube=True):
        #hard_index,根据res中的连通性计算出来的可能的FP的部分，即预测为1、应该维0、并且重置标签为0的点的坐标
        #传入的hard_index必须是一维的，或者在使用前改为一维的
        #new_label，使用网络重新预测的change和unchanged的标签， 原有的uncertain的标签不变，输入也是1维的。
        #根据new label计算 新的uncertain标签，newlabel中和原来的k-means 预测的HC标签不一致的，加入到uncertain中

        self.hard_smaple_index = hard_index

        #phard = np.zeros((len(self.hard_smaple_index), self.patch_size, self.patch_size, self.img.shape[2]))
        phard  = self.patchesData[self.hard_smaple_index]
        p_hard_label = np.zeros(len(self.hard_smaple_index))

        self.label_2nd_1d = self.label_current_1d  # 原始标签
        self.label_2nd_1d[hard_index] = 1  # hard sample 对应的标签 置为0，在cubes中对应的值为1.


        if newHCtrainCube:
            diff_index = np.array(np.where(new_label != self.label_current_1d)).squeeze() # 网络预测和k-means预测HC不同的像素位置
            print("length of hard sample is {}".format(len(diff_index)))
            #self.label_2nd_1d = self.label_initial_1d #原始标签
            self.label_2nd_1d[diff_index] = 1.5 # 把原始标签中 网络预测和k-means预测HC不同的像素位置 置为uncertain
            #self.label_2nd_1d 方便选取uncertain

            #重新计算训练数据集，加入hard sample，删除掉了网络预测和k-means预测HC不同的像素
            pdata = np.zeros((self.args.unchange_num + self.args.change_num, self.patch_size, self.patch_size, self.img.shape[2]))
            plabels = np.zeros(self.args.unchange_num + self.args.unchange_num) #初始全0

            len_hard = len(hard_index)
            #pdata[:len_hard] = phard

            label_1 = np.array(np.where(self.label_2nd_1d == 1)).squeeze()
            label_1_m = np.setdiff1d(label_1, hard_index)

            if len_hard < self.args.unchange_num:
                pdata[:len_hard] = phard
                label_1_index = np.random.choice(label_1_m, size=self.args.unchange_num - len_hard, replace=False)
                pdata[len_hard:self.args.unchange_num] = self.patchesData[label_1_index]
            else:                #如果超出了
                label_1_index = np.random.choice(self.hard_smaple_index, size=self.args.unchange_num, replace=False)
                pdata[:self.args.unchange_num] = phard[label_1_index]

            label_2 = np.array(np.where(self.label_2nd_1d == 2)).squeeze()
            if len(label_2) > self.args.change_num:
                label_2_index = np.random.choice(label_2, size=self.args.change_num, replace=False)
                pdata[self.args.unchange_num:] = self.patchesData[label_2_index]
                plabels[self.args.unchange_num:] = 1
                change_num = self.args.change_num
            else:
                pdata[self.args.unchange_num:len(label_2)] = self.patchesData[label_2]
                plabels[self.args.unchange_num:len(label_2)] = 1
                change_num = len(label_2)

            self.label_current_1d = self.label_2nd_1d # 原始标签
            return pdata[:self.args.unchange_num+change_num], plabels[:self.args.unchange_num+change_num]

        else:
            return phard, p_hard_label


