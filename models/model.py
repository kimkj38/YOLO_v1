import torch
import torch.nn as nn
import numpy as np


class BaseNetwork(nn.Module):
    def __init__(self, pretrained_weights):
        super(BaseNetwork, self).__init__()
        self.pretrained_weights = pretrained_weights

        # modules
        self.conv_0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv_1 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_2 = nn.Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_7 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_8 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_9 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_10 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_11 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_12 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_13 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_14 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_15 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_16 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_17 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_18 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_19 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.leaky = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.maxpool(self.leaky(self.conv_0(x)))
        x = self.maxpool(self.leaky(self.conv_1(x)))
        x = self.leaky(self.conv_2(x))
        x = self.leaky(self.conv_3(x))
        x = self.leaky(self.conv_4(x))
        x = self.maxpool(self.leaky(self.conv_5(x)))
        x = self.leaky(self.conv_6(x))
        x = self.leaky(self.conv_7(x))
        x = self.leaky(self.conv_8(x))
        x = self.leaky(self.conv_9(x))
        x = self.leaky(self.conv_10(x))
        x = self.leaky(self.conv_11(x))
        x = self.leaky(self.conv_12(x))
        x = self.leaky(self.conv_13(x))
        x = self.leaky(self.conv_14(x))
        x = self.maxpool(self.leaky(self.conv_15(x)))
        x = self.leaky(self.conv_16(x))
        x = self.leaky(self.conv_17(x))
        x = self.leaky(self.conv_18(x))
        x = self.leaky(self.conv_19(x))

        return x

    def load_pretrained_weights(self):
        fp = open(self.pretrained_weights, 'rb')
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=4)
        weights = np.fromfile(fp, dtype=np.float32)

        conv_list = [self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4,
                     self.conv_5, self.conv_6, self.conv_7, self.conv_8, self.conv_9,
                     self.conv_10, self.conv_11, self.conv_12, self.conv_13, self.conv_14,
                     self.conv_15, self.conv_16, self.conv_17, self.conv_18, self.conv_19]

        ptr = 0
        for layer in conv_list:
            num_biases = layer.bias.numel()
            conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
            ptr = ptr + num_biases
            conv_biases = conv_biases.view_as(layer.bias.data)
            layer.bias.data.copy_(conv_biases)

            num_weights = layer.weight.numel()
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights
            conv_weights = conv_weights.view_as(layer.weight.data)
            layer.weight.data.copy_(conv_weights)


# YOLO v1 pytorch
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.base = BaseNetwork('extraction.weights')

        self.conv_20 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_21 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_22 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conn_0 = nn.Linear(7*7*1024, 4096)
        self.conn_1 = nn.Linear(4096, 7*7*30)

        self.leaky = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.base(x)
        x = self.leaky(self.conv_20(x))
        x = self.leaky(self.conv_21(x))
        x = self.leaky(self.conv_22(x))
        x = self.leaky(self.conv_23(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.leaky(self.conn_0(x)))
        x = self.leaky(self.conn_1(x))
        x = self.sigmoid(x)

        return x.view(-1, 7, 7, 30)

    def init_network(self):
        print('Initializing weights...')
        self.base.load_pretrained_weights()
        ## TODO Add an initializer
