import argparse
import os
import pickle
import time

#import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from utils import *

from model.SimpleNet import SimpleNet
from preclassify import cluster_2types

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

#parser.add_argument('--data', metavar='DIR', default='.\\data\\Yellow_River', help='path to image')
#parser.add_argument('--data', metavar='DIR', default='.\\data\\Yellow_River', help='path to image')
#parser.add_argument('--data', metavar='DIR', default='.\\data\\Sulzberger1', help='path to image')
parser.add_argument('--data', metavar='DIR', default='.\\data\\ottawa', help='path to image')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['lenet','alexnet', 'vgg16', 'DDNet','simplenet'], default='simplenet',
                    help='CNN architecture (default: lenet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')

parser.add_argument('--nmb_cluster', '--k', type=int, default=15,
                    help='number of cluster for k-means (default: 2)')



""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self,x_train, y_train):
        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)
        self.y_data = torch.LongTensor(y_train)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签

        # x=torch.FloatTensor(data_rotate(self.x_data[index].cpu().numpy()))
        # y=torch.FloatTensor(gasuss_noise(self.y_data[index]))
        # x=torch.FloatTensor(datarotate(self.x_data[index]))
        # return x,self.y_data[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


def new_train_loader(mdata, mlabel, patch_size=7):
    x_train, y_train = createTrainingCubes(mdata, mlabel, patch_size)
    x_train = x_train.transpose(0, 3, 1, 2)
    print('... x train shape: ', x_train.shape) #(10000, 3, 7, 7)
    print('... y train shape: ', y_train.shape) #(74273, 3, 7, 7)

    # 创建 trainloader 和 testloader
    trainset = TrainDS(x_train, y_train)
    # train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=0)

    return train_loader


def main():
    global args
    args = parser.parse_args()
    img_name = args.data.split("\\")[-1]
    print(img_name)
    print(args.arch)

    im1_path  = args.data + '_1.bmp'
    im2_path  = args.data + '_2.bmp'
    imgt_path = args.data + '_gt.bmp'

    # important parameter
    patch_size = 7

    if 'Yellow_River' in im1_path :
        # read image, and then tranform to float32
        im1 = io.imread(im1_path)[:,:].astype(np.float32)
        im2 = io.imread(im2_path)[:,:].astype(np.float32)
        print("im1.shape is {}".format(im1.shape)) #(289, 257)
        im_gt = io.imread(imgt_path)[:, :].astype(np.float32)
        print("im_gt.shape is {}".format(im_gt.shape))  # (289, 257)
    else:
        # read image, and then tranform to float32
        im1 = io.imread(im1_path)[:,:,0].astype(np.float32)
        im2 = io.imread(im2_path)[:,:,0].astype(np.float32)
        print("im1.shape is {}".format(im1.shape)) #(289, 257)
        im_gt = io.imread(imgt_path)[:,:,0].astype(np.float32)
        print("im_gt.shape is {}".format(im_gt.shape)) #(289, 257)

    im_di = dicomp(im1, im2)
    print("im_di.shape is {}".format(im_di.shape)) #(289, 257)
    ylen, xlen = im_di.shape
    pix_vec = im_di.reshape([ylen*xlen, 1])

    # hiearchical FCM clustering
    # in the preclassification map,
    # pixels with high probability to be unchanged are labeled as 1
    # pixels with high probability to be changed are labeled as 2
    # pixels with uncertainty are labeled as 1.5
    #preclassify_lab = hcluster(pix_vec, im_di)
    preclassify_lab = cluster_2types(pix_vec, im_di)

    print("preclassify_lab.shape is {}".format(preclassify_lab.shape))
    print('... ... hiearchical clustering finished !!!')

    mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
    mdata[:,:,0] = im1
    mdata[:,:,1] = im2
    mdata[:,:,2] = im_di
    mlabel = preclassify_lab

    train_loader = new_train_loader(mdata, mlabel, patch_size=patch_size)

    x_test = createTestingCubes(mdata, patch_size)
    x_test = x_test.transpose(0, 3, 1, 2)
    print('... x test shape: ', x_test.shape)


    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    istrain = True
    # 网络放到GPU上
    net = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()


    if os.path.exists(img_name +'_' + args.arch +'kmeans_model.pth'):
        ## 读取模型
        model_load = net
        state_dict = torch.load(img_name +'_' + args.arch +'kmeans_model.pth')
        model_load.load_state_dict(state_dict['model'])
    else:
        # 开始训练
        total_loss = 0
        for epoch in range(149):
            if epoch % 50 == 0:
                preclassify_lab = cluster_2types(pix_vec, im_di)
                mlabel = preclassify_lab
                train_loader = new_train_loader(mdata, mlabel, patch_size=patch_size)

            for i, (inputs, labels) in enumerate(train_loader):

                inputs = inputs.to(device)
                labels = labels.to(device)
                # 优化器梯度归零
                optimizer.zero_grad()
                # 正向传播 +　反向传播 + 优化
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('[Epoch: %d]  [loss avg: %.4f]  [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))
        print('Finished Training')

        model = net
        torch.save({'model': model.state_dict()}, img_name +'_' + args.arch +'kmeans_model.pth')

    # 逐像素预测类别
    istrain=False
    net.eval()
    outputs = np.zeros((ylen, xlen))
    glo_fin=torch.Tensor([]).cuda()
    dct_fin=torch.Tensor([]).cuda()
    for i in range(ylen):
        for j in range(xlen):
            img_patch = x_test[i*xlen+j, :, :, :]
            img_patch = img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2])
            img_patch = torch.FloatTensor(img_patch).to(device)
            prediction = net(img_patch)

            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i, j] = prediction+1
        if (i+1) % 50 == 0:
            print('... ... row', i+1, ' handling ... ...')

    outputs = outputs-1

    #plt.imshow(outputs, 'gray')
    #plt.imsave(".\\save\\" + img_name +'_outputs.jpeg', outputs)
    plt.imsave(".\\save\\" + img_name + '_' + args.arch +'_kmeans_outputs.jpeg', outputs, cmap='gray')

    res = outputs*255
    res = postprocess(res)
    evaluate(im_gt, res)
    #plt.imshow(res, 'gray')
    plt.imsave(".\\save\\" + img_name + '_' + args.arch + '_kmeans_res.jpeg', res, cmap='gray')

if __name__ == '__main__':
    main()


'''
yellow_river
 Change detection results ==>
 ... ... FP:   1368
 ... ... FN:   1901
 ... ... OE:   3269
 ... ... PCC:  95.60
 ... ... KC:  95.89
'''
'''yellow_river 多次加载数据 epoch=100
 Change detection results ==>
 ... ... FP:   948
 ... ... FN:   2449
 ... ... OE:   3397
 ... ... PCC:  95.43
 ... ... KC:  95.69
'''

'''
ottawa
85451
16049
 Change detection results ==>
 ... ... FP:   686
 ... ... FN:   987
 ... ... OE:   1673
 ... ... PCC:  98.35
 ... ... KC:  98.50
'''

'''ottawa cluster =2 多次加载数据 epoch=100 
 Change detection results ==>
 ... ... FP:   335.0
 ... ... FN:   1285.0
 ... ... OE:   1620.0
 ... ... PCC:  98.40
 ... ... KC:  93.86
'''
''''ottawa cluster =2 多次加载数据 epoch=149 
 ... ... FP:   714.0
 ... ... FN:   1021.0
 ... ... OE:   1735.0
 ... ... PCC:  98.29
 ... ... KC:  93.53
'''

'''
Sulzberger1 hcluster
52926
12610
 Change detection results ==>
 ... ... FP:   931
 ... ... FN:   236
 ... ... OE:   1167
 ... ... PCC:  98.22
 ... ... KC:  98.65
'''

'''Sulzberger1 cluster_2types
52926
12610
 Change detection results ==>
 ... ... FP:   1107
 ... ... FN:   168
 ... ... OE:   1275
 ... ... PCC:  98.05
 ... ... KC:  98.53
'''

'''
52926 Sulzberger1 hcluster 60epoch
12610
 Change detection results ==>
 ... ... FP:   824
 ... ... FN:   195
 ... ... OE:   1019
 ... ... PCC:  98.45
 ... ... KC:  98.82
'''

''' Sulzberger1 多次加载数据 epoch=60
52926
12610
 Change detection results ==>
 ... ... FP:   1003
 ... ... FN:   154
 ... ... OE:   1157
 ... ... PCC:  98.23
 ... ... KC:  98.66
'''

'''Sulzberger1 多次加载数据 epoch=100
52926
12610
 Change detection results ==>
 ... ... FP:   1085
 ... ... FN:   119
 ... ... OE:   1204
 ... ... PCC:  98.16
 ... ... KC:  98.61
'''

