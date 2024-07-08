
import os
from typing import Any, List, Tuple
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms



def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union

def evaluate_model(model: Any, data_loader: DataLoader, label_dir: str) -> Tuple[float, float, float]:
    """
    Evaluate the model using ground truth labels.
    """
    model.eval()
    all_iou = []
    with torch.no_grad():
        for imgs, labels in data_loader:
            if isinstance(imgs, dict):
                for img_name, img_tensor in imgs.items():
                    imgs = img_tensor # Adjust this line based on your dataset
            imgs = imgs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            results = model(imgs)
            for i, result in enumerate(results.xyxy):
                if len(result) > 0:
                    # Take the most obvious person (highest confidence score)
                    best_result = result[0]
                    pred_box = best_result[:4].cpu().numpy()
                    
                    # Load ground truth boxes
                    class_id, cx, cy, w, h = labels
                    if class_id == 0:  # Only consider person class
                        gx, gy, gw, gh = cx * 416, cy * 416, w * 416, h * 416
                        gt_box = [gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2]
                        iou = calculate_iou(pred_box, gt_box)
                        all_iou.append(iou)
    
    mean_iou = np.mean(all_iou)
    return mean_iou