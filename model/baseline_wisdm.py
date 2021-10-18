import torch.nn as nn
from block.utils_block import *
import torch.nn.functional as F

# channel = [128, 256, 512, 13824]
# channel = [96, 192, 387, 10449]
# channel = [64, 128, 256, 6912]
# channel = [32, 64, 128, 3456]
# channel = [16, 32, 64, 1728]

class ConvNet_2d_wisdm(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[64, 128, 256, 6912]):
        super(ConvNet_2d_wisdm, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], 3, 2, 1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], 3, 2, 1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], 3, 2, 1),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(channel[3], 6)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = F.max_pool2d(x, (6,1))  #hcl试的6,1

        x = self.flatten(x)
        x = self.classifier(x)
        return x

# channel=[16, 32, 64, 1600]
# channel=[32, 64, 128, 3200]
# channel=[64, 128, 256, 6400]
# channel=[96, 192, 387, 9600]
# channel=[128, 256, 512, 12800]

class Cnn_wisdm(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[128, 256, 512, 13824]):
        super(Cnn_wisdm, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], 3, 2, 1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], 3, 2, 1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
        )
        self.pooling1 = nn.MaxPool2d((2, 1), (1, 1), (1, 0))

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], 3, 2, 1),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
        )
        self.pooling2 = nn.MaxPool2d((2, 1), (1, 1), (1, 0))

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(channel[3], 6)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pooling1(x)
        x = self.layer3(x)
        x = self.pooling2(x)

        x = self.flatten(x)
        x = self.classifier(x)


        return x

if __name__ == "__main__":
    net = Cnn_wisdm()
    X = torch.rand(1, 1, 200, 3)
    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)


