import os
import json
import logging
from typing import Any, Dict, List, Set, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        """
        Custom dataset for loading images and their corresponding annotations.
        
        Args:
            images_dir (str): Directory with all the images.
            labels_dir (str): Directory with all the label files (.txt).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        self.image_names = [name for name in os.listdir(images_dir) if name.endswith('.jpg') or name.endswith('.png')]
        self.label_names = [name for name in os.listdir(labels_dir) if name.endswith('.txt')]

        self.annotations = self.load_annotations()
        logger.info(f"Loaded {len(self.image_names)} images and {len(self.label_names)} labels.")


    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) : #-> Tuple[Dict[str, Image.Image], torch.Tensor[float]]
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise e        

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Extract annotation using the image name (without extension) to match with the annotation key
        annotation = torch.tensor(self.annotations.get(img_name, [])[0], dtype=torch.float32) 

        # if img_name in ['im00183.png','im04386.png','im00359.png', 'im00070.png','im01740.png', 'im00637.png','im02227.png','im00098.png']:
        #     logger.info(f"Processing image: {img_name}")
        logger.info(f"Processing image: {img_name}")

        return image, annotation

    def load_annotations(self) -> Dict[str, List[List[float]]]:
            annotations = {}
            for label_file in self.label_names:
                label_path = os.path.join(self.labels_dir, label_file)
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    img_name = os.path.splitext(label_file)[0] + '.png'
                    
                    if len(lines) != 0:             
                        # Process non-empty annotations
                        for line in lines:
                            parts = line.strip().split()
                            class_id = int(parts[0])
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            bbox = [class_id, center_x, center_y, width, height]
                            if img_name not in annotations:
                                annotations[img_name] = []
                            annotations[img_name].append(bbox)
                    else:
                        logger.warning(f"File {label_path} Does not have annotation!!")
            return annotations


    
class KeypointDataset(Dataset):
    def __init__(self, images_dir: str, annotations: Dict[str, Dict[str, List[int]]], transform=None):
        self.images_dir = images_dir
        self.annotations = annotations
        self.transform = transform
        self.image_names = list(annotations.keys())
        self.label_names = list(next(iter(annotations.values())).keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx) -> tuple[torch.Tensor, Any]:
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = self.annotations[image_name]
        targets = []
        for label in self.label_names:
            targets.extend(labels[label])
        targets = torch.tensor(targets, dtype=torch.float32)

        return (image, targets)
