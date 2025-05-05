import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from urllib import request
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
from functools import partial
import gc
import joblib

from SiamMae import *
from train_siam import *

root_path = '...'

from src.data.realestate.re10k_dataset import Re10k_dataset
dataset = Re10k_dataset(data_root="../dataset",mode="train")
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

# Model training

# Model, optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_mae_vit_tiny_path8 = sim_mae_vit_small_patch8_dec512d8b
model = sim_mae_vit_tiny_path8()
model = nn.DataParallel(model).to(device)

# Change in your branch
folder_logs = 'Siamese_folder/logs.txt'
folder_model = 'Siamese_folder'

num_epochs = 1000
model = train(model, train_loader, folder_logs, folder_model, num_epochs=num_epochs, lr=1e-4)