import os
import json
from typing import Any, Dict, List, Set

import torch
from torch.utils.data import Dataset

from PIL import Image


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

