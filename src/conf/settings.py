import logging.config
import os
from pathlib import Path
from typing import Any


# Data Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR.joinpath("hr-lspet")))
RAW_DATA_DIR = DATA_DIR.joinpath('raw')
LABELS_PATH = DATA_DIR.joinpath("joints.mat")


SOURCE_DIR = BASE_DIR.joinpath("src")
CONF_DIR = SOURCE_DIR.joinpath('conf')


LABELS_CONFIG_PATH = CONF_DIR.joinpath('labels.json')
TRAINING_PATH = DATA_DIR.joinpath("train")
VALIDATION_PATH = DATA_DIR.joinpath('val')
TESTING_PATH = DATA_DIR.joinpath("test")

# Features Path 
# FEATURES_DATA_DIR = DATA_DIR.joinpath("features")
# TRAINING_FEATURES_PATH = FEATURES_DATA_DIR.joinpath("training")
# TESTING_FEATURES_PATH = FEATURES_DATA_DIR.joinpath("testing")

# Results Paths
# RESULTS_DIR = BASE_DIR.joinpath("results")



# For data-split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# for data training
BATCH_SIZE = 32

