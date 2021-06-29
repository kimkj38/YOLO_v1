import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dataset import VOCdataset
import torchvision.transforms as transforms
import numpy as np


# YOLO v1 pytorch
class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()

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
        self.conv_20 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_21 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_22 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conn_0 = nn.Linear(7*7*1024, 4096)
        self.conn_1 = nn.Linear(4096, 7*7*30)

        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(2,2)
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
        x = self.leaky(self.conv_20(x))
        x = self.leaky(self.conv_21(x))
        x = self.leaky(self.conv_22(x))
        x = self.leaky(self.conv_23(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.leaky(self.conn_0(x)))
        x = self.leaky(self.conn_1(x))
        x = torch.squeeze(x)

        return x




def load_weights(file, net):
    fp = open(file, 'rb')
    #The first 5 values are header information
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number
    # 4. Images seen by the network (during training)
    header = np.fromfile(fp, dtype = np.int32, count = 4)
    weights = np.fromfile(fp, dtype = np.float32)

    conv_list = [net.conv_0, net.conv_1, net.conv_2, net.conv_3, net.conv_4,
                 net.conv_5, net.conv_6, net.conv_7, net.conv_8, net.conv_9,
                 net.conv_10, net.conv_11, net.conv_12, net.conv_13, net.conv_14,
                 net.conv_15, net.conv_16, net.conv_17, net.conv_18, net.conv_19,
                 net.conv_20, net.conv_21, net.conv_22, net.conv_23]

    conn_list = [net.conn_0, net.conn_1]

    ptr = 0
    for layer in conv_list:
        num_biases = layer.bias.numel()
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases
        conv_biases = conv_biases.view_as(layer.bias.data)
        layer.bias.data.copy_(conv_biases)

        num_weights = layer.weight.numel()
        conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        ptr = ptr + num_weights
        conv_weights = conv_weights.view_as(layer.weight.data)
        layer.weight.data.copy_(conv_weights)

    for layer in conn_list:
        num_biases = layer.bias.numel()
        conn_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases
        conn_biases = conn_biases.view_as(layer.bias.data)
        layer.bias.data.copy_(conn_biases)

        num_weights = layer.weight.numel()
        conn_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        ptr = ptr + num_weights
        conn_weights = conn_weights.view_as(layer.weight.data.T)
        conn_weights = conn_weights.T
        layer.weight.data.copy_(conn_weights)

print_freq = 5
lr = 0.001
weight_decay = 5.0e-4
num_epochs = 100


#model
darknet = DarkNet()

#Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(darknet.parameters(), lr = lr, weight_decay = weight_decay)

#Load Dataset
train_dataset =  VOCdataset(transforms=transforms.ToTensor(), mode='train')
test_dataset =  VOCdataset(transforms=transforms.ToTensor(), mode='test')

val_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)

val_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=32)

#Training Loop
best_val_loss = np.inf

for epoch in range(num_epochs):
    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))

    # Training.
    darknet.train()
    total_loss = 0.0
    total_batch = 0

    for idx, (x, labels) in enumerate(train_loader):
        batch_size_this_iter = x.size(0)

        preds = darknet(x)
        loss = criterion(preds, labels)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 현재 loss 출력
        if idx % print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
                  % (epoch, num_epochs, idx, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))


        # Validation.
        darknet.eval()
        val_loss = 0.0
        total_batch = 0

        for idx, (x, labels) in enumerate(val_loader):
            # Load data as a batch.
            batch_size_this_iter = x.size(0)

            with torch.no_grad():
                preds = darknet(x)
            loss = criterion(preds, labels)
            loss_this_iter = loss.item()
            val_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter
        val_loss /= float(total_batch)

        # Print.
        print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
              % (epoch + 1, num_epochs, val_loss, best_val_loss))
