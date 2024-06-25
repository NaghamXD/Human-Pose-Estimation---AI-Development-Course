
import json
import logging
import os
import re 

from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
from typing import List, Set

import numpy as np
import pandas as pd
from   scipy.io   import  loadmat
from collections import UserDict
from functools import cached_property


from src.conf import settings

from PIL import Image


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



def load_labels_config() -> LabelConfig:
    with open(settings.LABELS_CONFIG_PATH, "r", encoding="utf-8") as handle:
        labels_config = LabelConfig(json.load(handle))
    
    return labels_config




