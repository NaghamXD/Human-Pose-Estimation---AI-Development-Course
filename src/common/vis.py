import cv2
import os
import matplotlib.pyplot as plt


def visualize_bboxes(image_path: str, bbox_file_path: str) -> None:
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f'Failed to read image: {image_path}')
        return

    # Read the bounding box file
    with open(bbox_file_path, 'r') as f:
        lines = f.readlines()

    # Draw bounding boxes on the image
    for line in lines:
        x1, y1, x2, y2 = map(float, line.strip().split())
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0)  # Green color
        thickness = 2  # Thickness of the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
    # Convert BGR to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_rgb)
    plt.title('Bounding Boxes')
    plt.axis('off')  # Hide the axis
    plt.show()