
import os
import cv2
import yaml
import logging
import pandas as pd
import json
from collections import defaultdict
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

    # test_dir = data_config['test']
    test_img_dir = data_config['test'] #os.path.join(test_dir, 'images')
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
    
    low_accuracy_images = []

    # Prediction
    with torch.no_grad():
        for i in range(num_imgs):
            img_path = os.path.join(test_img_dir, img_list[i])
            img = cv2.imread(img_path)
            # img = Image.open(img_path).convert('RGB')  # Open image with PIL and convert to RGB


            if img is None:
                print(f'Failed to read image: {img_path}')
                continue

            # Apply transformation
            # img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension

            results = model(img)
            # results.show()
            logger.info(f"Image: {img_list[i]}")
            
            xyxy_all_tensors = results.xyxy[0]
            
            if xyxy_all_tensors.shape[0]:
                high_conf_idx = torch.argmax(xyxy_all_tensors[:, 4])
                xyxy = xyxy_all_tensors[high_conf_idx].cpu().numpy()
                print(results.pandas().xyxy[0])

                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]) 
                conf =  float(xyxy[4])

                # Save bounding boxes to existing json file
                json_file_path = os.path.join(save_path, 'labels', f"{img_list[i].split('.')[0]}.json")
                # Check if the file exists
                if not os.path.exists(json_file_path):
                    print(f"File {json_file_path} does not exist.")
                    return
                
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                
                data[img_list[i]]['bbox'] = [x1, y1, x2, y2]
                
                # Save the updated JSON back to the file
                with open(json_file_path, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                
                #Contextual Constraints:
                if conf < 0.8:
                    results.show()
                    low_accuracy_images.append([img_list[i],1, conf])
            else:
                logger.warning(f'File {img_list[i]} does not have a bbox predicted')
                low_accuracy_images.append([img_list[i],0, 0])
                # results.show()

            
        # Save low accuracy images to CSV
        if low_accuracy_images:
            file_name = 'yolo_low_accuracy_images.csv'
            df = pd.DataFrame(low_accuracy_images, columns=['file_name', 'bbox_exist', 'conf'])
            csv_filepath = os.path.join(save_path, file_name)
            df.to_csv(csv_filepath, index=False)
        logger.info(f"Saved {len(low_accuracy_images)} of low accuracy images into file {file_name}")
                
