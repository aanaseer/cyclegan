"""Script to start training the CycleGAN"""

import os

import torch
from train_utils import initiate_pipeline

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
SAVE_DIR = os.path.join(PROJ_ROOT, "outputs")

config = {
    "dataset_name": "bit2moji2simpsons",
    "image_dim": 256,
    "image_channels": 3,
    "learning_rate": 0.002,
    "beta1": 0.5,
    "beta2": 0.999,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 1,
    "lambda_cycle": 10,
    "lambda_identity": 0.0001,
    "condition_step": 250,
    "save_checkpoint": True,
    "save_images": True,
    "epochs": 200,
    "load_checkpoint": False,
    "checkpoint_file_name": "checkpoint.pth",
    "proj_root": PROJ_ROOT,
    "data_dir": DATA_DIR,
    "output_save_dir": SAVE_DIR,
    "learning_decay": True
}


def main():
    initiate_pipeline(config)


if __name__ == "__main__":
    main()
