import torch
from model import YOLOv1
from torchsummary import summary

# model
yolo = YOLOv1()
yolo.init_network()

# check params
summary(yolo, input_size=(3, 448, 448))
torch.save(yolo.state_dict(), 'model.pth')
