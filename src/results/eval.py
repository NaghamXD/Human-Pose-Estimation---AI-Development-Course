
import os
from typing import Any, List, Tuple
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms



def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes in XYWH format.
    """
    # Convert XYWH to XYXY format for intersection calculation
    x1 = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
    y1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    x2 = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
    y2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of each box
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_pckh(ground_truths: np.ndarray, predictions: np.ndarray, threshold=0.5):
    correct_keypoints = 0
    total_keypoints = 0

    for pred, gt in zip(predictions, ground_truths):
        head_size = np.linalg.norm(gt[1] - gt[2])
        pckh_threshold = threshold * head_size

        for p, g in zip(pred, gt):
            if np.linalg.norm(p - g) <= pckh_threshold:
                correct_keypoints += 1
            total_keypoints += 1

    pckh = correct_keypoints / total_keypoints
    return pckh

def evaluate_model(ground_truth_bbox: np.ndarray, predicted_bbox: np.ndarray, ground_truths: np.ndarray, predictions: np.ndarray) -> float:
    """
    Evaluate the model using ground truth labels.
    """
    ious = []
    for gt_box, pred_box in zip(ground_truth_bbox, predicted_bbox):
        iou = calculate_iou(gt_box, pred_box)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    pckh = calculate_pckh(predictions, ground_truths)

    results = {
        "Metric": ["PCKh@0.5", "Mean IoU"],
        "Score": [pckh, mean_iou]
    }

    df = pd.DataFrame(results)
    return df