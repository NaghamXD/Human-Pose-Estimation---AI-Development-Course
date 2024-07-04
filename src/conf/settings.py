import logging.config
import os
from pathlib import Path
from typing import Any


# Data Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(f"BASE_DIR is set to: {BASE_DIR}")
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR.joinpath("hr-lspet")))
RAW_DATA_DIR = DATA_DIR.joinpath('raw')
YOLO_OUTPUT_DATA_DIR = DATA_DIR.joinpath('yolo_output')
PROCESSED_DATA_DIR = DATA_DIR.joinpath('hrnet_input')

SOURCE_DIR = BASE_DIR.joinpath("src")
CONF_DIR = SOURCE_DIR.joinpath('conf')


LABELS_PATH = DATA_DIR.joinpath("joints.mat")
LABELS_CONFIG_PATH = CONF_DIR.joinpath('labels.json')
YOLO_LABELS_CONFIG_PATH = CONF_DIR.joinpath('bbox_labels.json')

PROCESSED_RAW_DATA_DIR = PROCESSED_DATA_DIR.joinpath('raw')
TRAINING_PATH = PROCESSED_DATA_DIR.joinpath("train")
VALIDATION_PATH = PROCESSED_DATA_DIR.joinpath('val')
TESTING_PATH = PROCESSED_DATA_DIR.joinpath("test")

# Features Path 
# FEATURES_DATA_DIR = DATA_DIR.joinpath("features")
# TRAINING_FEATURES_PATH = FEATURES_DATA_DIR.joinpath("training")
# TESTING_FEATURES_PATH = FEATURES_DATA_DIR.joinpath("testing")

# Results Paths
RESULTS_DIR = BASE_DIR.joinpath("results")



# For data-split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# for data training
BATCH_SIZE = 32
EPOCHS = 50

