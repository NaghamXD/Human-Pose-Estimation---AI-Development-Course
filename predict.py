
import os
import cv2
import yaml
import logging
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_yolov5(config_path: str, model_path: str, save_path: str) -> None:
    # Load data configuration
    with open(config_path, 'r') as file:
        data_config = yaml.safe_load(file)

    test_dir = data_config['test']
    test_img_dir = os.path.join(test_dir, 'images')
    os.makedirs(save_path, exist_ok=True)
    output_img_path=os.path.join(save_path, 'images')
    output_txt_path=os.path.join(save_path, 'labels')
    
    
    # Define transformations using torchvision.transforms.Compose
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust size as needed
        transforms.ToTensor(),
    ])

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model
    model.classes = [0]  # Set classes to detect (e.g., person)

    img_list = os.listdir(test_img_dir)
    num_imgs = len(img_list)
    # Prediction
    with torch.no_grad():
        for i in range(num_imgs):
            img_path = os.path.join(test_img_dir, img_list[i])
            img = cv2.imread(img_path)

            if img is None:
                print(f'Failed to read image: {img_path}')
                continue

            results = model(img)
            results.show()
            logger.info(f"Image: {img_list[i]}")
            xyxy = results.xyxy[0].cpu().numpy()[0]
            print(results.pandas().xyxy[0])
            # Convert the bounding box coordinates to integers
            x1, y1, x2, y2 =  int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            try:
                # Draw the bounding box on the image (use appropriate color and thickness)
                color = (0, 255, 0)  # Green color
                thickness = 2  # Thickness of the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                save_img_path = os.path.join(output_img_path, img_list[i])
                cv2.imwrite(save_img_path, img)
            except cv2.error as e:
                print("Error: ", e)