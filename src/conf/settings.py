import logging.config
import os
from pathlib import Path
from typing import Any


# Data Paths

BASE_DIR = Path(os.environ.get("BASE_DIR", Path(os.path.relpath(__file__)).parent.parent.parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR.joinpath("hpe_data")))




SOURCE_DIR = BASE_DIR.joinpath("src")
CONF_DIR = SOURCE_DIR.joinpath('conf')

LABELS_DIR = CONF_DIR.joinpath('MPHB-label-mat')
LABELS_PATH = LABELS_DIR.joinpath("MPHB-label.mat")
TRAINING_LABELS_PATH = LABELS_DIR.joinpath("train+val.mat")
TESTING_LABELS_PATH = LABELS_DIR.joinpath("test.mat")

# Features Path 
# FEATURES_DATA_DIR = DATA_DIR.joinpath("features")
# TRAINING_FEATURES_PATH = FEATURES_DATA_DIR.joinpath("training")
# TESTING_FEATURES_PATH = FEATURES_DATA_DIR.joinpath("testing")

# Results Paths
# RESULTS_DIR = BASE_DIR.joinpath("results")