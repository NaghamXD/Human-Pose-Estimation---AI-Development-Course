import argparse
import yaml
import os
from typing import Set
from pathlib import Path
import logging


from ultralytics import YOLO

from src.conf import settings



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_yolo(dataconfig_path: str, 
               modelconfig_path:str, 
               batch_size: int , 
               epochs: int , 
               model_weights: str = 'yolov8n-pose.pt') -> None:
    
    imgSize     = 640
    numWorkers  = 0
    ampMode     = False
    save_dir = os.path.join(settings.YOLO_OUTPUT_DATA_DIR, 'results')
    
    # modelYolo = YOLO(modelconfig_path)
    modelYolo = YOLO(modelconfig_path)
    modelYolo = modelYolo.load(model_weights)
    
    for epoch in range(epochs):
        modelYolo.train(data = dataconfig_path, 
                        epochs = 1, 
                        batch = batch_size, 
                        imgsz = imgSize, 
                        workers = numWorkers, 
                        project='runs/train',
                        name = 'yolov8-pose', 
                        amp = ampMode,
                        save_period=10 if epoch % 10 == 0 else None,
                        save_dir=save_dir)
        # Perform validation
        results = modelYolo.val(
                        data=dataconfig_path,
                        batch_size=batch_size,
                        imgsz=640,
                        project='runs/val',
                        name='yolov8-pose',
        )
        # Print validation results for the current epoch
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Validation Results - mAP: {results['metrics/mAP50']}")

    # Save final model weights
    modelYolo.save(f'weights/yolov8-pose_final.pt')



      
                                                
