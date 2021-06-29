import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from datasets.dataset import VOCdataset
import torchvision.transforms as transforms
import numpy as np

print_freq = 5
lr = 0.001
weight_decay = 5.0e-4
num_epochs = 100
batch_size = 32
n_features = 1000
S = 7
B = 2
C = 20


#model
model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, n_features),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(),
        nn.Linear(n_features, (B*5+C) * S * S),
        nn.Softmax()
    )

#Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

#Load Dataset
train_dataset =  VOCdataset(transforms=transforms.ToTensor(), mode='train')
val_dataset =  VOCdataset(transforms=transforms.ToTensor(), mode='test')

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                         shuffle=False,
                         batch_size=32)


#Training Loop
best_val_loss = np.inf

for epoch in range(num_epochs):
    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))

    # Training.
    model.train()
    total_loss = 0.0
    total_batch = 0

    for idx, (x, labels) in enumerate(train_loader):
        batch_size_this_iter = x.size(0)

        preds = model(x)
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
        model.eval()
        val_loss = 0.0
        total_batch = 0

        for idx, (x, labels) in enumerate(val_loader):
            # Load data as a batch.
            batch_size_this_iter = x.size(0)

            with torch.no_grad():
                preds = model(x)
            loss = criterion(preds, labels)
            loss_this_iter = loss.item()
            val_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter
        val_loss /= float(total_batch)

        # Print.
        print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
              % (epoch + 1, num_epochs, val_loss, best_val_loss))

#save model
save_folder = './results'
torch.save(model.state_dict(), save_folder)


