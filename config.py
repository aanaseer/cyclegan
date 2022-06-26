import os
import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.001
BETA_1 = 0.5
BETA_2 = 0.999
