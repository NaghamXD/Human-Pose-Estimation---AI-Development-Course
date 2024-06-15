import os
from pathlib import Path
import logging


# Logging String
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

PROJECT_NAME = 'HRNet_HPE'

list_of_files= [
    "src/conf/__init__.py",
    "src/conf/labels.json",
    "src/conf/settings.py",
    
    "src/common/cli.py",
    "src/commom/vis.py",
    
    "src/data/augmentations.py",
    "src/data/inference.py",
    
    "src/net/base.py",
    "src/net/hrnet.py",
    
    "src/results/gui.py",
    "src/results/eval.py",
    
    "research/trials_nagham.ipynb",
    "research/trials_amin.ipynb",
    "research/trials_rasha.ipynb"

    
    "utils.py",
    "train.py",
    "predict.py",
    "test.py"
    
    "requirements.txt"
]


for file in list_of_files:
    file_path = Path(file)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating Directory : {file_dir} for the file: {file_name}")
        
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, 'w') as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")
        

# Commands: python3 template.py