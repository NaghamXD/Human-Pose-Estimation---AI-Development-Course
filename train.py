import argparse
import yaml
import os
from typing import Set
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


from src.conf import settings
from src.results.eval import evaluate_model
from src.common.cli import LabelConfig
from modelutils import (KeypointDataset, 
                   DetectDataset)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_yolov5(config_path: str, batch_size: int = 32, epochs: int = 50, model_path: str = 'yolov5_best.pt') -> None:
    # Load data configuration
    with open(config_path, 'r') as file:
        data_config = yaml.safe_load(file)

    train_dir = data_config['train']
    val_dir = data_config['val']
    num_classes = data_config['nc']
    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    val_img_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')

    # Create Datasets and DataLoaders
    train_dataset = DetectDataset(train_img_dir, train_label_dir, transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
)
    val_dataset = DetectDataset(val_img_dir, val_label_dir, transform=transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
)
            
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Only detect person class

    best_iou = 0.0

    # Training and Validation
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            # Ensure imgs is a tensor
            if isinstance(imgs, dict):
                for _, img_tensor in imgs.items():
                    imgs = img_tensor # Adjust this line based on your dataset
                    
            imgs = imgs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            results = model(imgs)
        logger.info(f'Epoch {epoch+1}/{epochs} training completed.')

        # Validation
        mean_iou = evaluate_model(model, val_loader, val_label_dir)
        logger.info(f'Epoch {epoch+1}/{epochs} validation Mean IoU: {mean_iou:.4f}')

        # Save the best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), model_path)
            logger.info(f'New best model saved with Mean IoU: {mean_iou:.4f}')
            
            
def train_hrnet(args: argparse.Namespace,
                labels_config: LabelConfig) -> None:
    
    
    dataset=KeypointDataset(settings.RAW_DATA_DIR,
                            settings.LABELS_CONFIG_PATH, 
                            transform=transform)
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * settings.TRAIN_RATIO)
    val_size = int(total_size * settings.VAL_RATIO)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train = DataLoader(train_dataset, 
                              batch_size=settings.BATCH_SIZE, 
                              shuffle=True, num_workers=4)
    val = DataLoader(val_dataset, 
                            batch_size=settings.BATCH_SIZE, 
                            shuffle=False, num_workers=4)
    test = DataLoader(test_dataset, 
                             batch_size=settings.BATCH_SIZE, 
                             shuffle=False, num_workers=4)
    
    
    num_outputs = labels_config.num_labels
