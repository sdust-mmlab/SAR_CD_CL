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
import cv2


class MRC(nn.Module):
    def __init__(self, inchannel):
        super(MRC, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, 15, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(15)

        self.conv2_1 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(5)

        self.conv2_2 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(5)

        self.conv2_3 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_3 = nn.BatchNorm2d(5)

    def forward(self, x):
        ori_out = F.relu(self.bn1(self.conv1(x)))

        shape = (x.shape[0], 5, 7, 7)

        all_zero3_3 = torch.zeros(size=shape).cuda()
        all_zero1_3 = torch.zeros(size=(x.shape[0], 5, 3, 7)).cuda()
        all_zero3_1 = torch.zeros(size=(x.shape[0], 5, 7, 3)).cuda()

        all_zero3_3[:, :, :, :] = ori_out[:, 0:5, :, :]
        all_zero1_3[:, :, :, :] = ori_out[:, 5:10, 2:5, :]
        all_zero3_1[:, :, :, :] = ori_out[:, 10:15, :, 2:5]

        square = F.relu(self.bn2_1(self.conv2_1(all_zero3_3)))
        horizontal = F.relu(self.bn2_2(self.conv2_2(all_zero1_3)))
        vertical = F.relu(self.bn2_3(self.conv2_3(all_zero3_1)))
        horizontal_final = torch.zeros(size=(x.shape[0], 5, 7, 7)).cuda()
        vertical_final = torch.zeros(size=(x.shape[0], 5, 7, 7)).cuda()
        horizontal_final[:, :, 2:5, :] = horizontal[:, :, :, :]
        vertical_final[:, :, :, 2:5] = vertical[:, :, :, :]

        glo = square + horizontal_final + vertical_final
        # glo= F.relu(self.bn3(self.conv3(glo)))

        return glo


def DCT(x):
    out = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=True)
    # print(out.shape)
    # dct_out_1 =torch.Tensor([cv2.dct(x[i,0,:,:].detach().cpu().numpy()) \
    #                          for i in range(x.shape[0])])
    dct_out_1 = torch.Tensor([cv2.dct(np.float32(out[i, 0, :, :].detach().cpu().numpy())) \
                              for i in range(x.shape[0])])
    dct_out_2 = torch.Tensor([cv2.dct(np.float32(out[i, 1, :, :].detach().cpu().numpy())) \
                              for i in range(x.shape[0])])
    dct_out_3 = torch.Tensor([cv2.dct(np.float32(out[i, 2, :, :].detach().cpu().numpy())) \
                              for i in range(x.shape[0])])
    dct_out = torch.zeros(size=(x.shape[0], 3, 8, 8))
    dct_out[:, 0, :, :] = dct_out_1
    dct_out[:, 1, :, :] = dct_out_2
    dct_out[:, 2, :, :] = dct_out_3
    dct_out = dct_out.cuda()  # 放回cuda
    out = dct_out.view(x.shape[0], 3, 64)
    # out=torch.cat((out,out),2)
    out = F.glu(out, dim=-1)
    dct_out = out.view(x.shape[0], 1, 96)
    return dct_out


class DDNet(nn.Module):
    def __init__(self):
        super(DDNet, self).__init__()
        self.mrc1 = MRC(3)
        self.mrc2 = MRC(5)
        self.mrc3 = MRC(5)
        self.mrc4 = MRC(5)

        self.linear1 = nn.Linear(341, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):  # x.shape [B,3,7,7]
        m_1 = self.mrc1(x) #m1.shape [B,5,7,7]
        m_2 = self.mrc2(m_1) #m2.shape [B,5,7,7]
        m_3 = self.mrc3(m_2) #m3.shape [B,5,7,7]
        m_4 = self.mrc4(m_3) #m4.shape [B,5,7,7]
        # glo= F.relu(self.bn(self.conv(m_4)))
        glo = m_4.view(x.shape[0], 1, 245) #glo.shape torch.Size([128, 1, 245])

        dct_out = DCT(x) #dct_out.shape torch.Size([128, 1, 96])

        out = torch.cat((glo, dct_out), 2) #out.shape torch.Size([128, 1, 341])
        out = out.view(out.size(0), -1) #out.shape torch.Size([128, 341])
        # print(out.shape)
        out_1 = self.linear1(out) #out_1.shape torch.Size([128, 10])
        out = self.linear2(out_1) #out.shape torch.Size([128, 2])

        return out