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
import torch.nn.functional as F

import matplotlib.pyplot as plt

from utils import *
from preclassify import *

from model.DDNet import DDNet
from model.SimpleNet import SimpleNet

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

#parser.add_argument('data', metavar='DIR', default='.\\data\\Yellow_River', help='path to image')
parser.add_argument('--data', metavar='DIR', default='.\\data\\Sulzberger1', help='path to image')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['SimpleNet', 'DDNet'], default='SimpleNet',
                    help='CNN architecture (default: lenet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=5,
                    help='number of cluster for k-means (default: 5)')

parser.add_argument('--patch_size', '--p', type=int, default=7,
                    help='patch size, defualt is 7 ')

parser.add_argument('--batch_size', '--b', type=int, default=256,
                    help='batch size, defualt is 128 ')

parser.add_argument('--unchange_num',  type=int, default=7000,
                    help='unchanged pixel number for HC DATASET , defualt is 1000 ')

parser.add_argument('--change_num',  type=int, default=3000,
                    help='changed pixel number for HC DATASET , defualt is 1000 ')


parser.add_argument('--T1', type=int, default=100,
                    help='the low threshold used for adjust weight of the unlabled data traing, defualt is 100 ')

parser.add_argument('--T2', type=int, default=400,
                    help='the second threshhold for adjust weight of the unlabled data traing, defualt is 700 ')

parser.add_argument('--af', type=float, default=3.0,
                    help='the weight of the unlabled data traing, defualt is 3.0 ')

parser.add_argument('--hc_epoch', type=int, default=50,
                    help='the epoches of highly confident sample traing, defualt is 50 ')

parser.add_argument('--un_epoch', type=int, default=50,
                    help='the epoches of uncertain sample traing, defualt is 50 ')

parser.add_argument('--unlabel_interval', type=int, default=5,
                    help='the step interval of unlabel sample trainging for call hc sample traing, defualt is 50 ')


""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self,x_train, y_train=None):
        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)
        if y_train is not None:
            self.y_data = torch.LongTensor(y_train)
        else:
            self.y_data = None

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        if self.y_data is not None:
            return self.x_data[index], self.y_data[index]
        else:
            return self.x_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len

criterion = nn.CrossEntropyLoss().cuda()

#eval_criterion = F.nll_loss().cuda() #nll_loss() missing 2 required positional arguments: 'input' and 'target'
def evaluate_model(model, test_loader, test_len):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda()
            output = model(data)
            predicted = torch.max(output,1)[1]
            correct += (predicted == labels.cuda()).sum()
            #loss += F.nll_loss(output, labels.cuda()).item()
            loss += criterion(output, labels.cuda()).item()

    return (float(correct)/test_len) *100, (loss/len(test_loader))


#train_len = len(train = torch.utils.data.TensorDataset(x_train, y_train))
def train_supervised(model, train_loader, test_loader, epochs, train_len, test_len):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # 使用DDNet时，当lr为0.1或0.01时都会出现train loss为Nan的情况
    # 使用simplenet lr为0.1时没有问题。

    EPOCHS = epochs
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        running_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            output = model(X_batch)
            #labeled_loss = F.nll_loss(output, y_batch)
            labeled_loss = criterion(output, y_batch)

            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()

        if epoch % 10 == 0:
            test_acc, test_loss = evaluate_model(model, test_loader, test_len)
            print('Epoch: {} : Train Loss : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '
                  .format(epoch,running_loss / (10 * train_len),test_acc,test_loss))
            model.train()

'''
T1 = 100
T2 = 400
af = 3
'''

def alpha_weight(step, T1, T2, af):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af


acc_scores = []
unlabel = []
pseudo_label = []

alpha_log = []
test_acc_log = []
test_loss_log = []

#
def semisup_train(model, train_loader, unlabeled_loader, test_loader, test_len, args, epochs=50, batch_interval=50):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    EPOCHS = epochs
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 100

    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):
            # Forward Pass to get the pseudo labels
            #x_unlabeled = x_unlabeled[0].cuda()
            x_unlabeled = x_unlabeled.cuda()
            model.eval()
            output_unlabeled = model(x_unlabeled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            model.train()

            """ ONLY FOR VISUALIZATION"""
            if (batch_idx < 3) and (epoch % 10 == 0):
                unlabel.append(x_unlabeled.cpu())
                pseudo_label.append(pseudo_labeled.cpu())
            """ ********************** """

            # Now calculate the unlabeled loss using the pseudo label
            output = model(x_unlabeled)
            #unlabeled_loss = alpha_weight(step) * F.nll_loss(output, pseudo_labeled)
            unlabeled_loss = alpha_weight(step, args.T1, args.T2, args.af) * criterion(output, pseudo_labeled)

            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()

            # For every 50 batches train one epoch on labeled data
            if batch_idx % batch_interval == 0:

                # Normal training procedure
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                    output = model(X_batch)
                    #labeled_loss = F.nll_loss(output, y_batch)
                    labeled_loss = criterion(output, y_batch)

                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()

                # Now we increment step by 1
                step += 1

        test_acc, test_loss = evaluate_model(model, test_loader, test_len)
        print('Epoch: {} : Alpha Weight : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch,
                                                                                                   alpha_weight(step, args.T1, args.T2, args.af),
                                                                                                   test_acc, test_loss))

        """ LOGGING VALUES """
        alpha_log.append(alpha_weight(step, args.T1, args.T2, args.af))
        test_acc_log.append(test_acc / 100)
        test_loss_log.append(test_loss)
        """ ************** """
        model.train()


def main():
    global args
    args = parser.parse_args()

    img_name = args.data.split("\\")[-1]
    args.img_name = img_name

    im1_path  = args.data + '_1.bmp'
    im2_path  = args.data + '_2.bmp'
    imgt_path = args.data + '_gt.bmp'

    # important parameter
    patch_size = args.patch_size  # default is 7

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
    preclassify_lab = hcluster_new(pix_vec, im_di, args.nmb_cluster)

    print("preclassify_lab.shape is {}".format(preclassify_lab.shape))
    print('... ... hiearchical clustering finished !!!')


    mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
    mdata[:,:,0] = im1
    mdata[:,:,1] = im2
    mdata[:,:,2] = im_di
    mlabel = preclassify_lab

    #x_train, y_train = createTrainingCubes(mdata, mlabel, patch_size)
    cubes = createCubes(mdata, mlabel, patch_size)
    x_train, y_train = cubes.createHCTrainingCubes(unchange_num=args.unchange_num, change_num=args.change_num)
    x_train = x_train.transpose(0, 3, 1, 2)
    print('... x train shape: ', x_train.shape) #Sulzberger1 (10000, 3, 7, 7)
    print('... y train shape: ', y_train.shape) #Sulzberger1 (74273, 3, 7, 7)

    #x_test = createTestingCubes(mdata, patch_size)
    x_test, y_test = cubes.createTestCubes()
    x_test = x_test.transpose(0, 3, 1, 2)
    print('... x test shape: ', x_test.shape)
    print('... y test shape: ', y_test.shape)

    unlabel_data = cubes.createUncertainTrainingCubes()
    unlabel_data = unlabel_data.transpose(0, 3, 1, 2)
    print('... unlabel_data shape: ', unlabel_data.shape) # Sulzberger1 (4292, 3, 7, 7)

    # 创建 trainloader 和 testloader
    trainset = TrainDS(x_train, y_train)
    # train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = TrainDS(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size*2, shuffle=True, num_workers=0)

    unlabel_dataset = TrainDS(unlabel_data)
    unlabel_loader = torch.utils.data.DataLoader(dataset=unlabel_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    istrain = True
    # 网络放到GPU上
    net = SimpleNet().to(device)
    #net = DDNet().to(device)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()

    train_supervised(net, train_loader, test_loader, args.hc_epoch, len(trainset), len(testset))
    torch.save({'model': net.state_dict()}, img_name +'_' + args.arch+ '_' + str(args.patch_size) + '_'+ str(args.hc_epoch)
               + '_' +'hc_model.pth')

    semisup_train(net, train_loader, unlabel_loader, test_loader,len(testset), args, epochs= args.un_epoch, batch_interval=args.unlabel_interval)
    torch.save({'model': net.state_dict()},
               img_name + '_' + args.arch + '_'+ str(args.patch_size)+ '_' + str(args.un_epoch) + '_'+ str(args.unlabel_interval)
                   + '_' + 'unlabel_model.pth')


    # 逐像素预测类别

    x_test_all = createTestingCubes(mdata, patch_size)
    x_test_all = x_test_all.transpose(0, 3, 1, 2)
    print('... x test shape: ', x_test_all.shape)

    istrain=False
    net.eval()
    outputs = np.zeros((ylen, xlen))
    #glo_fin=torch.Tensor([]).cuda()
    #dct_fin=torch.Tensor([]).cuda()
    for i in range(ylen):
        for j in range(xlen):
            if preclassify_lab[i, j] != 1.5 :
                outputs[i, j] = preclassify_lab[i, j]
            else:
                img_patch = x_test_all[i*xlen+j, :, :, :]
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
    plt.imsave(".\\save\\" + img_name + '_' + args.arch + '_'+ str(args.patch_size)+ '_' + str(args.un_epoch) + '_'
               + str(args.unlabel_interval)+ '_outputs.jpeg', outputs, cmap='gray')

    res = outputs*255
    res = postprocess(res)
    evaluate(im_gt, res)
    #plt.imshow(res, 'gray')
    plt.imsave(".\\save\\"+ img_name + '_' + args.arch + '_'+ str(args.patch_size)+ '_' + str(args.un_epoch) + '_'
               + str(args.unlabel_interval)+  '_res.jpeg', res, cmap='gray')



    outputs_all = np.zeros((ylen, xlen))
    for i in range(ylen):
        for j in range(xlen):
            img_patch = x_test_all[i*xlen+j, :, :, :]
            img_patch = img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2])
            img_patch = torch.FloatTensor(img_patch).to(device)
            prediction = net(img_patch)

            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs_all[i, j] = prediction+1
        if (i+1) % 50 == 0:
            print('... ... row', i+1, ' handling ... ...')

    outputs_all = outputs_all-1

    plt.imsave(".\\save\\" + img_name + '_' + args.arch + '_'+ str(args.patch_size)+ '_' + str(args.un_epoch) + '_'
               + str(args.unlabel_interval)+ '_outputs_all.jpeg', outputs_all, cmap='gray')

    res_all = outputs_all*255
    res_all = postprocess(res_all)
    evaluate(im_gt, res_all)
    #plt.imshow(res, 'gray')
    plt.imsave(".\\save\\"+ img_name + '_' + args.arch + '_'+ str(args.patch_size)+ '_' + str(args.un_epoch) + '_'
               + str(args.unlabel_interval)+  '_res_all.jpeg', res_all, cmap='gray')


if __name__ == '__main__':
    main()


