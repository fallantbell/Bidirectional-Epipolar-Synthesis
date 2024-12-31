import numpy as np
import math
import random
from PIL import Image
import argparse
import torch
from torchvision import transforms
import sys, datetime, glob, importlib
sys.path.insert(0, ".")

import os

from src.metric.metrics import perceptual_sim, psnr, ssim_metric
from src.metric.pretrained_networks import PNet

exp_name = "exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error"
exp_setting = "evaluate_frame_21_video_1000_ckpt_100000"
target_folder = f"experiments/realestate/{exp_name}/{exp_setting}"

exp_name_2 = "exp_fixed_for_epipolar_maskcam_sepsoft-3_4gpu"
exp_setting_2 = "evaluate_frame_6_video_250_ckpt_150000"
target_folder_2 = f"experiments/realestate/{exp_name_2}/{exp_setting_2}"

parser = argparse.ArgumentParser(description="training codes")
parser.add_argument('--gpu', default= '4', type=str)
parser.add_argument("--seed", type=int, default=2333, help="")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# fix the seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

vgg16 = PNet().to("cuda")
vgg16.eval()
vgg16.cuda()

n_values_percsim = []
n_values_ssim = []
n_values_psnr = []

transform = transforms.Compose([
    transforms.ToTensor(),          # 轉換為張量，並將像素值縮放到 [0, 1]
])

video_list = []

for videos in sorted(os.listdir(target_folder)):
    path = f'{target_folder}/{videos}'
    if len(os.listdir(path))==0 :
        continue
    video_list.append(videos)
    values_percsim = []
    values_ssim = []
    values_psnr = []
    for i in range(1, 6): 
        #* (b,c,h,w) (0,1)
        image1 = Image.open(f'{target_folder}/{videos}/gt_{str(i).zfill(2)}.png').convert('RGB')
        t_img = transform(image1).unsqueeze(0).to("cuda")
        image2 = Image.open(f'{target_folder}/{videos}/predict_{str(i).zfill(2)}.png').convert('RGB')
        p_img = transform(image2).unsqueeze(0).to("cuda")

        # t_img = (batch["rgbs"][:, :, i, ...] + 1)/2
        # p_img = (generate_video[i] + 1)/2
        t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()

        perc_sim = t_perc_sim
        ssim_sim = ssim_metric(p_img, t_img).item()
        psnr_sim = psnr(p_img, t_img).item()
        
        values_percsim.append(perc_sim)
        values_ssim.append(ssim_sim)
        values_psnr.append(psnr_sim)

    n_values_percsim.append(values_percsim)
    n_values_ssim.append(values_ssim)
    n_values_psnr.append(values_psnr)
    if len(video_list)>=250:
        break

# print(f'{np.mean(n_values_percsim[0]):.4f}')
# print(f'{np.mean(n_values_ssim[0]):.2f}')
# print(f'{np.mean(n_values_psnr[0]):.2f}')

total_percsim = []
total_ssim = []
total_psnr = []

with open(os.path.join("eval", f"{exp_name}.txt"), 'w') as f:
    for i in range(len(n_values_percsim)):
        video_name = video_list[i]
        result_string = f"#{video_name}, percsim: {np.mean(n_values_percsim[i]):.04f}, ssim: {np.mean(n_values_ssim[i]):.02f}, psnr: {np.mean(n_values_psnr[i]):.02f}"
        f.write(result_string)
        f.write('\n')
        for j in range(len(n_values_percsim[i])):
            percsim = n_values_percsim[i][j]
            ssim = n_values_ssim[i][j]
            psnrr = n_values_psnr[i][j]

            total_percsim.append(percsim)
            total_ssim.append(ssim)
            total_psnr.append(psnrr)
            
    f.write("Total, percsim: (%.03f, %.02f), ssim: (%.02f, %.02f), psnr: (%.03f, %.02f)" 
            % (np.mean(total_percsim), np.std(total_percsim), 
               np.mean(total_ssim), np.std(total_ssim),
               np.mean(total_psnr), np.std(total_psnr)))
    f.write('\n')
sys.exit()
# ---- 比較的

n_values_percsim = []
n_values_ssim = []
n_values_psnr = []

for j in range(len(video_list)):
    videos = video_list[j]
    values_percsim = []
    values_ssim = []
    values_psnr = []
    for i in range(1, 6): 
        #* (b,c,h,w) (0,1)
        image1 = Image.open(f'{target_folder_2}/{videos}/gt_{str(i).zfill(2)}.png').convert('RGB')
        t_img = transform(image1).unsqueeze(0).to("cuda")
        image2 = Image.open(f'{target_folder_2}/{videos}/predict_{str(i).zfill(2)}.png').convert('RGB')
        p_img = transform(image2).unsqueeze(0).to("cuda")

        # t_img = (batch["rgbs"][:, :, i, ...] + 1)/2
        # p_img = (generate_video[i] + 1)/2
        t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()

        perc_sim = t_perc_sim
        ssim_sim = ssim_metric(p_img, t_img).item()
        psnr_sim = psnr(p_img, t_img).item()
        
        values_percsim.append(perc_sim)
        values_ssim.append(ssim_sim)
        values_psnr.append(psnr_sim)

    n_values_percsim.append(values_percsim)
    n_values_ssim.append(values_ssim)
    n_values_psnr.append(values_psnr)

total_percsim = []
total_ssim = []
total_psnr = []

with open(os.path.join("eval", f"{exp_name_2}.txt"), 'w') as f:
    for i in range(len(n_values_percsim)):
        video_name = video_list[i]
        result_string = f"#{video_name}, percsim: {np.mean(n_values_percsim[i]):.04f}, ssim: {np.mean(n_values_ssim[i]):.02f}, psnr: {np.mean(n_values_psnr[i]):.02f}"
        f.write(result_string)
        f.write('\n')
        for j in range(len(n_values_percsim[i])):
            percsim = n_values_percsim[i][j]
            ssim = n_values_ssim[i][j]
            psnrr = n_values_psnr[i][j]

            total_percsim.append(percsim)
            total_ssim.append(ssim)
            total_psnr.append(psnrr)
            
    f.write("Total, percsim: (%.03f, %.02f), ssim: (%.02f, %.02f), psnr: (%.03f, %.02f)" 
            % (np.mean(total_percsim), np.std(total_percsim), 
               np.mean(total_ssim), np.std(total_ssim),
               np.mean(total_psnr), np.std(total_psnr)))
    f.write('\n')