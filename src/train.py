"""Script to start training the CycleGAN"""

import os

import torch
from train_utils import initiate_pipeline

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
DATA_DIR = os.path.join(PROJ_ROOT, "data")

config = {
    "dataset_name": "horse2zebra",
    "image_dim": 256,
    "image_channels": 3,
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 1,
    "lambda_cycle": 10,
    "lambda_identity": 0.1,
    "condition_step": 1,
    "save_checkpoint": False,
    "save_images": True,
    "epochs": 150,
    "load_checkpoint": False,
    "checkpoint_file_name": "checkpoint.pth",
    "data_dir": DATA_DIR
}

def main():
    initiate_pipeline(config)

if __name__ == "__main__":
    main()
