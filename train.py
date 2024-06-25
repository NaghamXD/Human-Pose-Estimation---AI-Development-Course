import argparse
from typing import Set

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


from src.conf import settings
from src.common.cli import LabelConfig
from utils import KeypointDataset


def transform(img_size: Set[int]):
    if img_size:
        trans = transforms.Compose([
            transforms.Resize(img_size),# img_size = (224, 224)
            transforms.ToTensor(),
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
    return trans

def train_yolov5(args: argparse.Namespace,
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
    # num_workers: how many core can you use in parellel when uploading the images
    train= DataLoader(train_dataset, 
                              batch_size=settings.BATCH_SIZE, 
                              shuffle=True, num_workers=4)
    val= DataLoader(val_dataset, 
                            batch_size=settings.BATCH_SIZE, 
                            shuffle=False, num_workers=4)
    test = DataLoader(test_dataset, 
                             batch_size=settings.BATCH_SIZE, 
                             shuffle=False, num_workers=4)
    
    
    num_outputs = labels_config.num_labels
    # eval metrics:
    
    
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
