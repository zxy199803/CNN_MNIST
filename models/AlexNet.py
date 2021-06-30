import torch
from torch import nn


class Config:
    def __init__(self):
        self.model_name = 'AlexNet'

        self.log_path = './run/log/' + self.model_name
        self.save_path = './run/saved_dict/' + self.model_name + '.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.resize = 224
        self.batch_size = 128
        self.lr = 0.001
        self.num_epochs = 7
        self.lr_scheduler_gamma = 0.9  # 学习率指数衰减的gamma
        self.dev_data = False  # 是否有验证集
        self.val_batch = 100  # 每多少轮输出在验证集上的效果
        self.require_improvement = 1000  # 验证集loss超过1000batch没下降，结束训练

        self.class_list = list(str(s) for s in range(0, 10))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),  # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # kernel_size, stride
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        featrue = self.conv(img)
        output = self.fc(featrue.view(img.shape[0], -1))
        return output
