import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch
import torch.nn.functional as F
from block.awn import SliceableConv2d, SliceableLinear
from block.awn import MaskTriangularConv2d
from block.awn import SwitchableSharedBatchNorm2d
from block.awn import AWNet

from block.utils_block import *


def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
            # print('this similarity matrix x shape',x.shape)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)

    return R


class conv_loss_block(AWNet, nn.Module):
    def __init__(self, inC, outC, init_width_mult=1.0, slices=[1.0], divisor=1, min_channels=1, pooling_flag=False):
        super(conv_loss_block, self).__init__()
        self.decode_ys = []
        self.bns_decode_ys = []

        self.set_width_mult(1.0)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        n = self._slice(128, init_width_mult)
        self.inC = inC
        self.outC = outC
        self.pooling_flag = pooling_flag

        if self.pooling_flag:
            self.encoder = nn.Sequential(
                MaskTriangularConv2d(inC, outC, (6, 1), (2, 1), (1, 0), fixed_in=True),
                SwitchableSharedBatchNorm2d(n, slices),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1), (1, 1), (1, 0))
            )
        else:
            self.encoder = nn.Sequential(
                MaskTriangularConv2d(inC, outC, (6, 1), (2, 1), (1, 0), fixed_in=True),
                SwitchableSharedBatchNorm2d(num_features=128, slices=slices),
                nn.ReLU(inplace=True),
            )

        self.pool = nn.MaxPool2d((2, 1), (1, 1), (1, 0))

        for i in range(3):
            decode_y = nn.Linear(n, 6)
            setattr(self, 'decode_y%i' % i, decode_y)
            self._set_init(decode_y)
            self.decode_ys.append(decode_y)

        self.conv_loss = MaskTriangularConv2d(n,  n, (2, 1), (1, 0), fixed_in=True)  # ###

        if True:
            self.bn = SwitchableSharedBatchNorm2d(n, slices)
            nn.init.constant_(self.bn.sliced_weight, 1)
            nn.init.constant_(self.bn.sliced_bias, 0)

        self.nonlin = nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, amsgrad=False)

        self.clear_stats()

    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)
        init.constant_(layer.bias, 0.2)

    def clear_stats(self):
        self.loss_sim = 0.0
        self.loss_pred = 0.0
        self.correct = 0
        self.examples = 0

    def set_learning_rate(self, lr):
        self.lr = lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot, loop, is_training):

        h = self.encoder(x)

        h_return = h
        h_shape = h.shape

        h_return = self.dropout(h_return)

        h_loss = self.conv_loss(h_return)

        Rh = similarity_matrix(h_loss)

        # caculate unsupervised loss
        Rx = similarity_matrix(x).detach()
        loss_unsup = F.mse_loss(Rh, Rx)

        h_pool = h_return

        y_hat_local = self.decode_ys[loop](h_pool.view(h_pool.size(0), -1))
        loss_pred = (1 - 0.99) * F.cross_entropy(y_hat_local, y.detach().long())

        Ry = similarity_matrix(y_onehot).detach()
        loss_sim = 0.99 * F.mse_loss(Rh, Ry)

        loss_sup = loss_pred + loss_sim

        loss = loss_sup * 1 + loss_unsup * 0

        if is_training:
            loss.backward(retain_graph=False)

        if is_training:
            self.optimizer.step()
            self.optimizer.zero_grad()
            h_return.detach_()
        loss = loss.item()

        return h_return, loss


class convnet(AWNet):
    def __init__(self, init_width_mult=1.0, num_classes=17, slices=[1.0], divisor=1, min_channels=1):
        super(convnet, self).__init__()
        self.set_width_mult(1.0)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        n = self._slice(128, init_width_mult)

        self.bn = []
        self.layer1 = conv_loss_block(1, n, pooling_flag=False)
        self.layer2 = conv_loss_block(n, 2*n, pooling_flag=True)
        self.layer3 = conv_loss_block(2*n, 4*n, pooling_flag=True)


        self.layer_out = SliceableLinear(n*216, num_classes, fixed_out=True)
        self.layer_out.sliced_weight.data.zero_()

        bn = nn.SwitchableSharedBatchNorm2d(n, slices),
        setattr(self, 'pre_bn', bn)
        self.bn.append(bn)

    def parameters(self):
        return self.layer_out.parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers): ######################
            layer.set_learning_rate(lr)

    def optim_step(self):
        for i, layer in enumerate(self.layers):
            # print('下一步优化')
            layer.optim_step()

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):######################
            # print('初始化optim')
            layer.optim_zero_grad()

    def forward(self, x, y, y_onehot, is_training):

        total_loss = 0.0
        x = x.type(torch.cuda.FloatTensor)
        x = self.bn[0](x)
        x, loss = self.layer1(x, y, y_onehot, 0, is_training)
        total_loss += loss
        x, loss = self.layer2(x, y, y_onehot, 1, is_training)
        total_loss += loss
        x, loss = self.layer3(x, y, y_onehot, 2, is_training)
        total_loss += loss
        x = x.contiguous().view(x.size(0), -1)
        x = self.layer_out(x)

        return x, total_loss

class AWLeNet5_unimib(AWNet):
    def __init__(self, num_classes=17,
                 init_width_mult=1.0, slices=[1.0], divisor=1, min_channels=1):
        super(AWLeNet5_unimib, self).__init__()

        self.set_width_mult(1.0)
        self.set_divisor(divisor)
        self.set_min_channels(min_channels)

        n = self._slice(128, init_width_mult)
        inC = 1
        # outL = 12

        log_slices = [0.25, 0.5, 0.75, 1.0]
        self.features = nn.Sequential(
            MaskTriangularConv2d(inC, n, (6, 1), (2, 1), (1, 0), fixed_in=True),
            SwitchableSharedBatchNorm2d(n, slices),
            nn.ReLU(inplace=True),
            MaskTriangularConv2d(n, 2*n, (6, 1), (2, 1), (1, 0)),
            SwitchableSharedBatchNorm2d(2*n, slices),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0)),
            MaskTriangularConv2d(2*n, 4*n, (6, 1), (2, 1), (1, 0)),
            SwitchableSharedBatchNorm2d(4*n, slices),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0)),

            # MaskTriangularConv2d(4 * n, 4 * n, (6, 1), (2, 1), (1, 0)),
            # SwitchableSharedBatchNorm2d(4 * n, slices),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d((2, 1), (1, 1), (1, 0)),
            # MaskTriangularConv2d(4 * n, 4 * n, (6, 1), (2, 1), (1, 0)),
            # SwitchableSharedBatchNorm2d(4 * n, slices),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d((2, 1), (1, 1), (1, 0)),
        )
        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            SliceableLinear(n*216, num_classes, fixed_out=True)  # 384 768 1536 3072
        )

    def forward(self, x):
        x = self.features(x)
        # print("features(x)", x.shape)
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        # print("flatten(x)", x.shape)
        x = self.classifier(x)
        return x


def get_model(args):

    return AWLeNet5_unimib(num_classes=args.num_classes,
                           init_width_mult=args.model_init_width_mult,
                           slices=args.model_width_mults)


if __name__ == "__main__":
    net = AWLeNet5_unimib()
    X = torch.rand(1, 1, 151, 3)
    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)

