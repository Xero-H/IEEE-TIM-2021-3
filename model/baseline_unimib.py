import torch.nn as nn
from block.utils_block import *
from block.awn import SliceableLinear

# channel = [128, 256, 512, 27648]
# channel = [96, 192, 387, 20736]
# channel = [64, 128, 256, 13824]
# channel = [32, 64, 128, 6912]
# channel = [16, 32, 64, 3456]

class ConvNet_2d_unimib(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[64, 128, 256, 13824]):
        super(ConvNet_2d_unimib, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(channel[3], 17)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.flatten(x)
        x = self.classifier(x)

        return x


class ConvNet_2d_unimib33(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[128, 256, 512, 27648]):
        super(ConvNet_2d_unimib33, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
        )
        self.pooling1 = nn.MaxPool2d((2, 1), (1, 1), (1, 0))

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
        )
        self.pooling2 = nn.MaxPool2d((2, 1), (1, 1), (1, 0))

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(channel[3], 17)
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



class ConvNet_2d_unimib88(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self, channel=[64, 128, 256, 9984]):
        super(ConvNet_2d_unimib88, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], (12, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], (12, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], (12, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            # SliceableLinear(13056, 17, fixed_out=True)
            nn.Linear(channel[3], 17)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        return x

if __name__ == "__main__":
    net = ConvNet_2d_unimib33()
    X = torch.rand(1, 1, 151, 3)  # pamap2
    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)

