import torch.nn as nn
from block.utils_block import *
from block.awn import SliceableLinear

channel_list = {1: [128, 256, 512, 143360],
                0.75: [96, 192, 384, 107520],
                0.5: [64, 128, 256, 71680],
                0.25: [32, 64, 128, 35840],
                0.125: [16, 32, 64, 17920]}  # [128, 256, 512, 23040]

class ConvNet_2d_pamap2(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[128, 256, 512, 143360]):
        super(ConvNet_2d_pamap2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(channel[3], 12)  #  143360
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x

# channel=[16, 32, 64, 12800]
# channel=[32, 64, 128, 25600]
# channel=[64, 128, 256, 51200]
# channel = [96, 192, 384, 76800]
# channel=[128, 256, 512, 102400]
class Cnn_pamap2(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[128, 256, 512, 143360]):
        super(Cnn_pamap2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
        )
        self.pooling1 = nn.MaxPool2d((2, 1), (1, 1), (1, 0))

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
        )
        self.pooling2 = nn.MaxPool2d((2, 1), (1, 1), (1, 0))

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(channel[3], 12)  #  143360
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
    net = Cnn_pamap2()
    X = torch.rand(1, 1, 171, 40)  # pamap2
    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)

