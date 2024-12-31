import math
import random
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非互動式 backend
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys, datetime, glob, importlib
sys.path.insert(0, ".")
import argparse

import torch
import torchvision

from SiamMae import *

def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x+1.0)*127.5
    x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def normalize_numpy(array):
    # 找到 array 的最小值和最大值
    min_val = np.min(array)
    max_val = np.max(array)

    # 如果最小值等於最大值，返回全0陣列，避免除以0的情況
    if min_val == max_val:
        return np.zeros_like(array)

    # 歸一化到 [0, 1]
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--len", type=int, default=4, help="len of prediction")
parser.add_argument('--gpu', default= '4', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from src.data.realestate.re10k_dataset import Re10k_dataset
dataset_abs = Re10k_dataset(data_root="../dataset",mode="test",infer_len=args.len)

test_loader_abs = torch.utils.data.DataLoader(
        dataset_abs,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last = True
    )

model_path = 'Siamese_folder/mask095_epoch_170.pt'
model = sim_mae_vit_small_patch8_dec512d8b()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

target_folder = f"Siamese_folder/result"
os.makedirs(target_folder, exist_ok=True)


mask_ratio = 0.5
with torch.no_grad():
    cnt = 0
    pred_images= []
    datas0 = []
    datas1 = []
    for batch_idx, (data) in enumerate(test_loader_abs):
        # print(data[0]['rgbs'].shape)
        data = data[0]['rgbs'].to(device) #* bcthw
        datas0.append(data[0,:,0,...])
        datas1.append(data[0,:,3,...])
        # img0 = as_png(data[0, :, 0, ...].permute(1,2,0))
        # img1 = as_png(data[0, :, 1, ...].permute(1,2,0))

        loss, pred = model.forward(data,mask_ratio=mask_ratio)
        pred_image = np.transpose(model.module.unpatchify(pred).cpu().detach().numpy(), (0, 2, 3, 1))
        pred_image = normalize_numpy(pred_image)
        pred_images.append(pred_image)

        # pred_image = normalize_numpy(pred_image[0])
        # pred_image = as_png(pred_image)

        # img0.save(f"{target_folder}/{batch_idx}_gt0.png")
        # img1.save(f"{target_folder}/{batch_idx}_gt1.png")
        # pred_image.save(f"{target_folder}/{batch_idx}_pred.png")

        cnt += 1
        print(f"cnt = {cnt}")
        if cnt >= 8:
            break

    datas0 = torch.stack(datas0,dim=0)
    datas1 = torch.stack(datas1,dim=0)

    x, mask, ids_restore = model.module.forward_encoder(datas1, mask_ratio = 1-mask_ratio)

    mask = torch.reshape(mask,(-1,32,32))
    resized_mask = torch.kron(mask, torch.ones((8, 8)).to('cuda'))

    data0_image = np.transpose(datas0.cpu().detach().numpy(), (0, 2, 3, 1)) #* (8,256,256,3)
    data0_image = normalize_numpy(data0_image)
    data1_image = np.transpose(datas1.cpu().detach().numpy(), (0, 2, 3, 1)) #* (8,256,256,3)
    data1_image = normalize_numpy(data1_image)

    masked_image = resized_mask[ :, :, :, np.newaxis].cpu() * data1_image[:]

    print(masked_image.shape)

    fig, axs = plt.subplots(3, 8, figsize=(15, 7))

    for i in range(8):
        axs[0, i].imshow(data0_image[i])
        axs[0, i].axis('off')
        axs[0, 0].set_title('Raw frames', fontsize=20)

    for i in range(8):
        axs[1, i].imshow(masked_image[i])
        axs[1, i].axis('off')
        axs[1, 0].set_title('Masked frames', fontsize=20)
    axs[1, 0].axis('off')

    for i in range(8):
        axs[2, i].imshow(pred_images[i][0])
        axs[2, i].axis('off')
        axs[2, 0].set_title('Reconstructed frames', fontsize=20)
    axs[2, 0].axis('off')


    plt.tight_layout()
    plt.savefig(f"{target_folder}/result.png")
    # plt.show()


