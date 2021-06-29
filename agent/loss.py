import torch
import torch.nn as nn
from utils.box_utils import intersection_over_union


class DetectionLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(DetectionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, targets):
        # Calculate IoU
        iou_b1 = intersection_over_union(predictions[..., 0:4], targets[..., 0:4])
        iou_b2 = intersection_over_union(predictions[..., 5:9], targets[..., 5:9])
        ious = torch.cat([iou_b1, iou_b2], dim=-1)
        iou_max, bestbox = torch.max(ious, dim=-1)  # [batch_size, S, S]
        bestbox = bestbox.unsqueeze(-1)
        exists_box = targets[..., 4].unsqueeze(-1)  # [batch_size, S, S, 1]

        # Localization Loss
        box_predictions = exists_box * (
              (
                  bestbox * predictions[..., 5:10]
                  + (1 - bestbox) * predictions[..., 0:5]
              )
          )

        box_targets = exists_box * targets[..., 0:5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
              torch.abs(box_predictions[..., 2:4] + 1e-6)
          )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
          
        box_loss = self.mse(
              torch.flatten(box_predictions, end_dim=-2),
              torch.flatten(box_targets, end_dim=-2)
          )

        # Confidence Loss
        # For Object
        pred_box = (bestbox * predictions[..., 4:5] + (1 - bestbox) * predictions[..., 9:10])
        
        object_loss = self.mse(
              torch.flatten(exists_box * pred_box),
              torch.flatten(exists_box * targets[..., 4:5])
          )
        
        # For No Object
        no_object_loss = self.mse(
              torch.flatten((1 - exists_box) * predictions[..., 4:5], start_dim=1),
              torch.flatten((1 - exists_box) * targets[..., 4:5], start_dim=1)
          )

        no_object_loss += self.mse(
              torch.flatten((1 - exists_box) * predictions[..., 9:10], start_dim=1),
              torch.flatten((1 - exists_box) * targets[..., 4:5], start_dim=1)
          )

        # Class Loss
        class_loss = self.mse(
              torch.flatten(exists_box * predictions[..., 10:], end_dim=-2),
              torch.flatten(exists_box * targets[..., 10:], end_dim=-2)
          )

        loss = (
              self.lambda_coord * box_loss 
              + object_loss 
              + self.lambda_noobj * no_object_loss 
              + class_loss 
          ) 

        return loss
