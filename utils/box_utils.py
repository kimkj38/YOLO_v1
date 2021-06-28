import torch
from collections import Counter


def convert_predictions(predictions, batch_size, S):
    """
    predictions: (batch_size, S, S, 30)
    return: (batch_size, S, S, 6) -> [confidence, box_score, x, y, w, h]
    """
    bboxes1 = predictions[..., 0:5]  # (batch_size, S, S, 5)
    bboxes2 = predictions[..., 5:10]
    bbox_scores = torch.cat((predictions[..., 4].unsqueeze(-1), predictions[..., 9].unsqueeze(-1)), dim=-1)  # (batch_size, S, S, 2)
    best_bbox = bbox_scores.argmax(-1).unsqueeze(-1)  # (batch_size, S, S, 1)
    best_bboxes = bboxes1 * (1 - best_bbox) + best_bbox * bboxes2  # (batch_size, S, S, 5)
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)  # (batch_size, S, S, 1)
    x = 1 / S * (best_bboxes[..., 0:1] + cell_indices)
    y = 1 / S * (best_bboxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))  ## TODO Why 1/S ??
    w_h = 1 / S * best_bboxes[..., 2:4]
    absolute_bboxes = torch.cat((x, y, w_h), dim=-1)

    predicted_class = predictions[..., 10:].argmax(-1).unsqueeze(-1)
    best_bbox_scores = torch.max(predictions[..., 4], predictions[..., 9]).unsqueeze(-1)  ## TODO using best_bbox as a mask

    return torch.cat((predicted_class, best_bbox_scores, absolute_bboxes), dim=-1)


def predictions_to_boxes(predictions):
    batch_size, S, _, _ = predictions.shape
    converted_pred = convert_predictions(predictions, batch_size, S).reshape(batch_size, S * S, -1)  # (batch_size, S * S, 6)
    all_bboxes = []

    for b in range(batch_size):
        bboxes = []
        for g in range(S * S):
            bboxes.append([x.item() for x in converted_pred[b, g, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def intersection_over_union(pred_boxes, label_boxes):
    bbox1_x1 = pred_boxes[..., 0:1] - pred_boxes[..., 2:3] / 2
    bbox1_x2 = pred_boxes[..., 0:1] + pred_boxes[..., 2:3] / 2
    bbox1_y1 = pred_boxes[..., 1:2] - pred_boxes[..., 3:4] / 2
    bbox1_y2 = pred_boxes[..., 1:2] + pred_boxes[..., 3:4] / 2

    bbox2_x1 = label_boxes[..., 0:1] - label_boxes[..., 2:3] / 2
    bbox2_x2 = label_boxes[..., 0:1] + label_boxes[..., 2:3] / 2
    bbox2_y1 = label_boxes[..., 1:2] - label_boxes[..., 3:4] / 2
    bbox2_y2 = label_boxes[..., 1:2] + label_boxes[..., 3:4] / 2

    x1 = torch.max(bbox1_x1, bbox2_x1)
    x2 = torch.min(bbox1_x2, bbox2_x2)
    y1 = torch.max(bbox1_y1, bbox2_y1)
    y2 = torch.min(bbox1_y2, bbox2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    bbox1_area = abs((bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1))
    bbox2_area = abs((bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1))

    return intersection / (bbox1_area + bbox2_area - intersection + 1e-6)

## TODO update later
def mean_average_precision(pred_boxes, label_boxes, num_classes=20, eps=1e-6):
    for c in range(num_classes):
        predictions = []
        ground_truths = []
        for pred in pred_boxes:
            if pred[1] == c:
                predictions.append(pred)
        for label in label_boxes:
            if label[1] == c:
                ground_truths.append(label)
