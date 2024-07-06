
import json
import yaml
import logging
import os
import sys
import random
import shutil
from scipy.io import loadmat
import numpy as np
import pandas as pd
from collections import UserDict
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
from typing import List, Set



from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.conf import settings


class LabelConfig(UserDict):
    def __init__(self, data):
        super().__init__(data)

    @property
    def num_images(self) -> int:
        return len(self.data)

    @property
    def num_labels(self) -> int:
        if self.data:
            first_key = next(iter(self.data))
            return len(self.data[first_key])
        return 0

    @property
    def label_names(self) -> List[str]:
        if self.data:
            first_key = next(iter(self.data))
            return list(self.data[first_key].keys())
        return []

    def image_names(self) -> List[str]:
        return list(self.data.keys())
    
    def image_sizes(self, directory) -> List[Set[int]]:
        image_sizes = []
        image_names = list(self.data.keys())
        for name in image_names:
            img_path = os.path.join(directory, name)
            with Image.open(img_path) as img:
                size_str = (img.width, img.height)
                image_sizes.append(size_str)
        return image_sizes



def load_labels_config(path) -> LabelConfig:
    with open(path, "r", encoding="utf-8") as handle:
        labels_config = LabelConfig(json.load(handle))
    
    return labels_config

def create_yolo_data(custom_yolo_annotation_path, output_path, raw_data_dir, train_ratio=settings.TRAIN_RATIO) -> None:
    '''
        Custom Dataset from RoboFlow:
            1. Download YOLOv5 PyTorch TXT as .zip file, open it and add custom_yolo_annotation_path as its path
            .txt file should contain [class_id center_x center_y width height]
            2. create an output_path:
            output_path/
            ├── train/
            │   ├── images/   # Directory containing training images
            │   └── labels/   # Directory containing training labels (RoboFlow annotations)
            ├── val/
            │   ├── images/   # Directory containing validation images
            │   └── labels/   # Directory containing validation labels (RoboFlow annotations)
            ├── test/
            │   └── images/   # Directory containing test images
            └── yaml file(s)  # YAML file(s) specifying paths to images and labels:
                names:
                0: person
                nc: 1
                path: /Users/rashajaber/Human-Pose-Estimation---AI-Development-Course/hr-lspet/output_path
                test: test/images
                train: train/images
                val: val/images
            3. From raw_data_dir it will move the annotated original files to their new destination

    '''
    # Create output directories if they don't exist
    train_img_path = os.path.join(output_path, 'data', 'train', 'images')
    train_label_path = os.path.join(output_path, 'data','train', 'labels')
    val_img_path = os.path.join(output_path, 'data','val', 'images')
    val_label_path = os.path.join(output_path, 'data','val', 'labels')
    inference_path = os.path.join(output_path, 'data','test', 'images')
    
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)
    os.makedirs(inference_path, exist_ok=True)
    
    # Get list of annotation files
    all_images = set(os.listdir(raw_data_dir))
    annotation_files = os.listdir(custom_yolo_annotation_path)
    random.shuffle(annotation_files)
    
    # Determine number of files for train, val, and inference
    num_files = len(annotation_files)
    num_train = int(train_ratio * num_files)
    
    # Split files into train, val, and inference sets
    train_files = annotation_files[:num_train]
    val_files = annotation_files[num_train:]

    clean_annotation_files = []
    # Move files to respective directories
    for filename in train_files:
        img_name = f"{filename.split('_')[0]}.png"
        clean_annotation_files.append(img_name)
        dest_filename = f"{filename.split('_')[0]}.txt"
        shutil.copy(os.path.join(raw_data_dir, img_name), train_img_path)
        shutil.copy(os.path.join(custom_yolo_annotation_path, filename), os.path.join(train_label_path, dest_filename))
    
    for filename in val_files:
        img_name = f"{filename.split('_')[0]}.png"
        clean_annotation_files.append(img_name)
        dest_filename = f"{filename.split('_')[0]}.txt"        
        shutil.copy(os.path.join(raw_data_dir, img_name), val_img_path)
        shutil.copy(os.path.join(custom_yolo_annotation_path, filename), os.path.join(val_label_path, dest_filename))
    
    
    
    inference_files = all_images - set(clean_annotation_files)

    for filename in inference_files:
        shutil.copy(os.path.join(raw_data_dir, filename), inference_path)
    
    # Create data.yaml
    data_yaml_path = os.path.join(output_path, 'config/data.yaml')
    with open(data_yaml_path, 'w') as yamlfile:
        yaml.safe_dump({
            'names': {0: 'person'},
            'nc': 1,
            'train': os.path.join(output_path, 'data','train'),
            'val': os.path.join(output_path, 'data','val'),
            'test': os.path.join(output_path, 'data','test')
        }, yamlfile, default_flow_style=False)
        
    # make sure there is no such file as '.DS_Store'
    
    

def create_hrnet_json_labels(img_dir: str, mat_path: str, output_json_path:str) -> None:
    # Load joint data from joints.mat
    try:
        joints_data  = loadmat(mat_path)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        raise

    key = [k for k in joints_data.keys() if k[0] != '_'][0] #joints
    joint_annotations = joints_data[key]
    
    
    # Get Images filenames:
    image_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    num_images = len(image_filenames)

    
    for img_idx in range(num_images):
        keypoints = []
        for kp_idx in range(joint_annotations.shape[0]):
            x, y, v = joint_annotations[kp_idx, :, img_idx]
            keypoints.extend([int(x), int(y), int(v)])
        
        image_name = image_filenames[img_idx]
        annotation = {image_name: {"bbox": [], "keypoints": keypoints}}
        
        # Save each annotation to a separate JSON file
        json_filename = os.path.join(output_json_path, f"{image_name.split('.')[0]}.json")
        with open(json_filename, 'w') as json_file:
            json.dump(annotation, json_file, indent=4)

    print(f"Annotations saved to {output_json_path} directory")
