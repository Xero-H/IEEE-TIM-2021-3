import torch.nn as nn
import torch

from block.awn import SliceableLinear
from block.awn import MaskTriangularConv2d
from block.awn import SwitchableSharedBatchNorm2d
from block.awn import AWNet

from block.utils_block import *


class AWLeNet5_wisdm(AWNet):
    def __init__(self, num_classes=6,
                 init_width_mult=1.0, slices=[1.0], divisor=1, min_channels=1):
        super(AWLeNet5_wisdm, self).__init__()

        self.set_width_mult(1.0)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        n = self._slice(128, init_width_mult)
        inC = 1
        # outL = 12

        log_slices = [0.25, 0.5, 0.75, 1.0]
        # self.features = nn.Sequential(
        #     MaskTriangularConv2d(inC, n, (8, 1), (2, 1), 0, fixed_in=True),
        #     SwitchableSharedBatchNorm2d(n, slices),
        #     nn.ReLU(inplace=True),
        #     MaskTriangularConv2d(n, 2*n, (8, 1), (2, 1), 0),
        #     SwitchableSharedBatchNorm2d(2*n, slices),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d((2, 1), (1, 1), (1, 0)),
        #     MaskTriangularConv2d(2*n, 4*n, (8, 1), (2, 1), 0),
        #     SwitchableSharedBatchNorm2d(4*n, slices),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d((2, 1), (1, 1), (1, 0)),
        # )
        self.features = nn.Sequential(
            MaskTriangularConv2d(inC, n, 3, 2, 1, fixed_in=True),
            SwitchableSharedBatchNorm2d(n, slices),
            nn.ReLU(inplace=True),
            MaskTriangularConv2d(n, 2 * n, 3, 2, 1),
            SwitchableSharedBatchNorm2d(2 * n, slices),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0)),
            MaskTriangularConv2d(2 * n, 4 * n, 3, 2, 1),
            SwitchableSharedBatchNorm2d(4 * n, slices),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0)),
        )

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            SliceableLinear(n*108, num_classes, fixed_out=True)  # 16128 4032
                                                                # (321) 6912 1728
        )

    def forward(self, x):
        print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.classifier(x)
        return x


def get_model(args):

    return AWLeNet5_wisdm(num_classes=args.num_classes,
                          init_width_mult=args.model_init_width_mult,
                          slices=args.model_width_mults)


if __name__ == "__main__":
    net = AWLeNet5_wisdm()
    X = torch.rand(1, 1, 200, 3)
    # named_children?????????????????????????????????(named_modules????????????????????????,???????????????????????????)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)