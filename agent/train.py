import os
import torch
from model import YOLOv1
from loss import DetectionLoss
from datasets import get_data
# from torchsummary import summary


def train():
    print_freq = 5
    lr = 0.001
    weight_decay = 5.0e-4
    num_epochs = 100
    dataset_dir = os.path.join(os.getcwd(), 'datasets')

    # model
    yolo = YOLOv1()
    yolo.init_network()

    # check params
    # summary(yolo, input_size=(3, 448, 448))
    # torch.save(yolo.state_dict(), 'model.pth')

    # Loss, Optimizer
    criterion = DetectionLoss()
    optimizer = torch.optim.Adam(yolo.parameters(), lr=lr, weight_decay=weight_decay)

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
            batch_size_this_iter = x.size(0)

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
