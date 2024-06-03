import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNet, self).__init__()

        self.step1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True))
        self.step2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True))
        self.step3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True))

        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.projector1 = nn.Linear(in_features=256 , out_features=128)
        self.projector2 = nn.Linear(in_features=128,
                                    out_features=num_classes)

    def forward(self, x):

        x = self.step1(x)#[128,64,5,5]
        x = self.step2(x) #[128,128,3,3]
        x = self.step3(x) #[128,256,1,1]

        output = self.dropout(x) #[128,256,1,1]

        output = output.view(x.shape[0], -1)

        output = self.relu(self.projector1(output))
        output = self.projector2(output)
        return output


if __name__ == '__main__':
    x = torch.rand(128, 3, 7, 7)
    num_classes = 2
    net = SimpleNet(num_classes)
    output = net(x)
    print(output.shape)