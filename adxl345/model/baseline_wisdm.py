import torch.nn as nn
from block.utils_block import *



class ConvNet_2d_wisdm(nn.Module):
    """
    2d版卷积，网络绘制方法
    """
    def __init__(self):
        super(ConvNet_2d_wisdm, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, 2, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d((4, 1), (2, 1), (1, 0))
        # )

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 64, (8, 1), (2, 1), 0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(64, 128, (8, 1), (2, 1), 0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(128, 256, (8, 1), (2, 1), 0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 512, (8, 1), (2, 1), 0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )

        # self.avgpool = nn.AvgPool2d((4, 1))

        # self.conv1 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1  # (1, 0)
        # )
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(
        #     in_channels=64,
        #     out_channels=128,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1  # (1, 0)
        # )
        # self.bn2 = nn.BatchNorm2d(128)
        # # self.pool2 = nn.MaxPool2d(kernel_size=(2, 1),
        # #                           stride=(1, 1),
        # #                           padding=(1, 0))
        # self.conv3 = nn.Conv2d(
        #     in_channels=128,
        #     out_channels=256,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1  # (1, 0)
        # )
        # self.bn3 = nn.BatchNorm2d(256)
        # # self.pool3 = nn.MaxPool2d(kernel_size=(2, 1),
        # #                           stride=(1, 1),
        # #                           padding=(1, 0))
        # self.conv4 = nn.Conv2d(
        #     in_channels=256,
        #     out_channels=512,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )
        # self.bn4 = nn.BatchNorm2d(512)
        #
        # self.relu = nn.ReLU6(inplace=True)

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(13824, 6)
            # SliceableLinear(14592, 6, fixed_out=True)
        )

        # self.fc1 = nn.Linear(14592, 1024)  # 4608 52000
        # # self.bn1 = nn.BatchNorm1d(100)
        # self.dropout1 = nn.Dropout(p=0.3)
        # self.fc2 = nn.Linear(1024, 128)
        # # self.bn2 = nn.BatchNorm1d(100)
        # self.dropout2 = nn.Dropout(p=0.3)
        # self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
        # # x = self.pool3(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)

        x = self.flatten(x)
        x = self.classifier(x)

        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)

        return x


if __name__ == "__main__":
    net = ConvNet_2d_wisdm()
    X = torch.rand(1, 1, 200, 3)
    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)


