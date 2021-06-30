from torch import nn
import torch.nn.functional as F
import torch


class Config:
    def __init__(self):
        self.model_name = 'ResNet'

        self.log_path = './run/log/' + self.model_name
        self.save_path = './run/saved_dict/' + self.model_name + '.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.resize = 96
        self.batch_size = 256
        self.lr = 0.001
        self.num_epochs = 5
        self.lr_scheduler_gamma = 0.9  # 学习率指数衰减的gamma
        self.dev_data = False  # 是否有验证集
        self.val_batch = 100  # 每多少轮输出在验证集上的效果
        self.require_improvement = 1000  # 验证集loss超过1000batch没下降，结束训练

        self.class_list = list(str(s) for s in range(0, 10))


# class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
#     def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
#         super(Residual, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         return F.relu(Y + X)
#
#
# class GlobalAvgPool2d(nn.Module):
#     # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
#     def __init__(self):
#         super(GlobalAvgPool2d, self).__init__()
#
#     def forward(self, x):
#         return F.avg_pool2d(x, kernel_size=x.size()[2:])
#
#
# class FlattenLayer(nn.Module):
#     def __init__(self):
#         super(FlattenLayer, self).__init__()
#
#     def forward(self, x):  # x shape: (batch, *, *, ...)
#         return x.view(x.shape[0], -1)
#
#
# def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
#     if first_block:
#         assert in_channels == out_channels  # 第一个模块的通道数和输入通道一致
#     blk = []
#     for i in range(num_residuals):
#         if i == 0 and not first_block:
#             blk.append(Residual)
#         else:
#             blk.append(Residual(out_channels, out_channels))
#     return nn.Sequential(*blk)
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
#         self.net.add_module('resnet_block2', resnet_block(64, 128, 2))
#         self.net.add_module('resnet_block3', resnet_block(128, 256, 2))
#         self.net.add_module('resnet_block3', resnet_block(256, 512, 2))
#         self.net.add_module('global_avg_pool', GlobalAvgPool2d())  # # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
#         self.net.add_module('fc', nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

class BasicBlock(nn.Module):
    expansion = 1  # 每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 50,101,152
class Bottleneck(nn.Module):
    expansion = 4  # 4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,  # 输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class Model(nn.Module):

    def __init__(self, block=BasicBlock, blocks_num=[3, 4, 6, 3], num_classes=10, include_top=True):  # block残差结构 include_top为了之后搭建更加复杂的网络
        super(Model, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
