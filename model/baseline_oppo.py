import torch.nn as nn
from block.utils_block import *
from block.awn import SliceableLinear
#
# channel_list = {1: [128, 256, 512, 53760],
#                 0.75: [96, 192, 384, 4320],
#                 0.5: [64, 128, 256, 26880],
#                 0.25: [32, 64, 128, 13440],
#                 0.125: [16, 32, 64, 6720]}

#
# [128, 256, 512, 76800]) [64, 128, 256, 38400]

class ConvNet_2d_oppo(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[16, 32, 64, 6720]):  # 53760
        super(ConvNet_2d_oppo, self).__init__()

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
            nn.Linear(channel[3], 18)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.flatten(x)

        x = self.classifier(x)

        return x

class Cnn_oppo(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[128, 256, 512, 53760]):
        super(Cnn_oppo, self).__init__()

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
            nn.Linear(channel[3], 18)  #  143360
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
    net = Cnn_oppo()
    X = torch.rand(1, 1, 40, 113)  # pamap2
    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)
