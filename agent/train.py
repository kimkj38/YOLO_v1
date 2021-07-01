import os
import torch
from models import YOLOv1
from loss import DetectionLoss
from datasets import get_data
import math
# from torchsummary import summary


def train():
    print_freq = 5
    init_lr = 0.001
    base_lr = 0.01
    momentum = 0.9
    weight_decay = 5.0e-4
    num_epochs = 135
    dataset_dir = os.path.join(os.getcwd(), 'datasets')

    # model
    yolo = YOLOv1()
    yolo.init_network()
    yolo.cuda()

    # check params
    # summary(yolo, input_size=(3, 448, 448))
    # torch.save(yolo.state_dict(), 'model.pth')

    # Loss, Optimizer
    criterion = DetectionLoss()
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduling.
    def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
        if epoch == 0:
            lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
        elif epoch == 1:
            lr = base_lr
        elif epoch == 75:
            lr = 0.001
        elif epoch == 105:
            lr = 0.0001
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # Load Dataset
    train_loader, test_loader = get_data(dataset_dir=dataset_dir)

    for epoch in range(num_epochs):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, num_epochs))

        # Training.
        yolo.train()
        total_loss = 0.0
        total_batch = 0

        for idx, (x, labels) in enumerate(train_loader):
            x = x.to('cuda')
            labels = labels.to('cuda')
            batch_size_this_iter = x.size(0)

            update_lr(optimizer, epoch, float(idx) / float(len(train_loader) - 1))
            lr = get_lr(optimizer)

            preds = yolo(x)
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
            # yolo.eval()
            # val_loss = 0.0
            # total_batch = 0
            #
            # for v_idx, (x_v, labels_v) in enumerate(val_loader):
            #     # Load data as a batch.
            #     batch_size_this_iter = x.size(0)
            #
            #     with torch.no_grad():
            #         preds = yolo(x)
            #     loss = criterion(preds, labels)
            #     loss_this_iter = loss.item()
            #     val_loss += loss_this_iter * batch_size_this_iter
            #     total_batch += batch_size_this_iter
            # val_loss /= float(total_batch)
            #
            # # Print.
            # print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
            #       % (epoch + 1, num_epochs, val_loss, best_val_loss))
