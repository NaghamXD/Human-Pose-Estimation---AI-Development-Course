import os
import cv2
import logging
import pandas as pd
import json
from typing import Union

from PIL import Image

import torch
from ultralytics import YOLO

from src.common import cli
from utils import normalize_xywh
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_yolo(image_path: str, model_path: str, save_path: str) -> Union[list, str]:
    image_name = os.path.basename(os.path.normpath(image_path))
    
    # Load your trained model
    model = YOLO(model_path)
    model.classes = [0]  # Set classes to detect (e.g., person)

    img = cv2.imread(image_path)
    
    if img is None:
        logger.error(f'Failed to read image: {image_path}')
        return

    results = model(img, conf=0.25)
    # results[0].show()
    logger.info(f"Image: {image_name}")
    
    if results[0].boxes.xyxy.numel() > 0:
        confs = results[0].boxes.conf
        high_conf_idx = torch.argmax(confs)
        conf = float(confs[high_conf_idx])

        xyxy = results[0].boxes.xyxy[high_conf_idx].cpu().numpy()
        xywh = cli.xyxy_to_xywh(xyxy)
        
        
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        x_center, y_center, width, height = normalize_xywh(xywh, image_width, image_height)

        
        # Save bounding boxes to json file
        
        json_data = {"bbox": [x_center, y_center, width, height]}
        json_file_path = os.path.join(save_path, f"{image_name.split('.')[0]}.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        df = pd.DataFrame({
                'x_center': [x_center],
                'y_center': [y_center],
                'width': [width],
                'height': [height],
                'confidence': conf
            })
            
        #Contextual Constraints:
        # if conf < 0.5 :
        #     logger.info(f"Low Accuracy Image: {image_name}")
        #     results[0].show()
            
        return json_data['bbox']
    else:
        results[0].show()
        logger.warning(f'File {image_name} does not have a bbox predicted')
        return image_name
