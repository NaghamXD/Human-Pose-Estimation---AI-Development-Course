
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
    # model = torch.hub.load('path/to/yolov5', 'custom', path=model_path, source='local', force_reload = True)  # local repo
    model.classes = [0]  # Set classes to detect (e.g., person)


    # Prediction
    with torch.no_grad():
        for img_name in os.listdir(test_img_dir):
            img_path = os.path.join(test_img_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f'Failed to read image: {img_path}')
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            img_pil = Image.fromarray(img_rgb)
            img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
            results = model(img_tensor)
           # Process each detected object
            for pred in results[0]:  # iterate through predictions
                xyxy = pred[:4].cpu().numpy()  # extract bounding box coordinates
                conf = float(pred[4])  # confidence score
                cls_conf = float(pred[5])  # class confidence
                cls = int(pred[5])  # class index (assuming it's the same as class confidence)

                # Save labels to text file
                if output_txt_path:
                    x_center = (xyxy[2] + xyxy[0]) / 2
                    y_center = (xyxy[3] + xyxy[1]) / 2
                    width = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    xywh = [x_center, y_center, width, height]  # x_center, y_center, width, height
                    with open(os.path.join(output_txt_path, f'{img_name[:-4]}.txt'), 'a') as f:
                        f.write(f'{cls} {" ".join([str(x) for x in xywh])}\n')

                # Draw bounding boxes on the image
                if output_img_path:
                     cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)


            # Save annotated image
            save_img_path = os.path.join(output_img_path, img_name)
            cv2.imwrite(save_img_path, img)
            print(f'Processed image: {img_name}')