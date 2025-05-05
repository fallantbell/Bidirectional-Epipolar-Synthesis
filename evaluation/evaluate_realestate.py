import math
import random
import os
cpu_num = 8  # Num of CPUs you want to use
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
import argparse

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非互動式 backend
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys, datetime, glob, importlib
sys.path.insert(0, ".")

import torch
import torchvision
torch.set_num_threads(cpu_num)

from omegaconf import OmegaConf
from einops import rearrange
from SiamMae import *

from torchsummary import summary
import time

# args
parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--base", type=str, default="realestate_16x16_adaptive",
                    help="experiments name")
parser.add_argument("--exp", type=str, default="exp_1_error",
                    help="experiments name")
parser.add_argument("--ckpt", type=str, default="last",
                    help="checkpoint name")
parser.add_argument("--siamese_ckpt", type=str, default="Siamese_folder/mask095_fulldata.pt",
                    help="siamese checkpoint")
parser.add_argument("--data-path", type=str, default="../dataset",
                    help="data path")
parser.add_argument("--len", type=int, default=4, help="len of prediction")
parser.add_argument('--gpu', default= '0', type=str)
parser.add_argument("--video_limit", type=int, default=20, help="# of video to test")
parser.add_argument("--gap", type=int, default=10, help="")
parser.add_argument("--seed", type=int, default=2333, help="")
parser.add_argument("--GT_start", action='store_true')
parser.add_argument("--mask_ratio", type=float, default=0.9, help="")
parser.add_argument("--mix_frame", type=int, default=10, help="")
parser.add_argument("--type",type=str, default='forward')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# config
config_path = "./configs/realestate/%s.yaml" % args.base
cpt_path = "./experiments/realestate/%s/model/%s.ckpt" % (args.exp, args.ckpt)
cpt_path = "experiments/exp_base_error/model/last.ckpt"
if args.exp[-5:]=="error":
    cpt_path = "./experiments/%s/model/%s.ckpt" % (args.exp, args.ckpt)
else:
    cpt_path = "./experiments/realestate/%s/model/%s.ckpt" % (args.exp, args.ckpt)

video_limit = args.video_limit
frame_limit = args.len

target_save_path = "./experiments/realestate/%s/evaluate_frame_%d_video_%d_ckpt_%s_mask%f/" % (args.exp, frame_limit, video_limit, args.ckpt,args.mask_ratio)
if args.GT_start:
    target_save_path = "./experiments/realestate/%s/evaluate_frame_%d_video_%d_GTstart/" % (args.exp, frame_limit, video_limit)
os.makedirs(target_save_path, exist_ok=True)

attend_test_folder = f"{args.type}_attn_test"
exp_test_folder = f"{attend_test_folder}/{args.exp}"
ckpt_test_folder = f"{exp_test_folder}/{args.ckpt}"
os.makedirs(attend_test_folder, exist_ok=True)
os.makedirs(exp_test_folder, exist_ok=True)
os.makedirs(ckpt_test_folder, exist_ok=True)

# metircs
from src.metric.metrics import perceptual_sim, psnr, ssim_metric
from src.metric.pretrained_networks import PNet

vgg16 = PNet().to("cuda")
vgg16.eval()
vgg16.cuda()

# load model
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# define relative pose
def compute_camera_pose(R_dst, t_dst, R_src, t_src):
    # first compute R_src_inv
    R_src_inv = R_src.transpose(-1,-2)
    
    R_rel = R_dst@R_src_inv
    t_rel = t_dst-R_rel@t_src
    
    return R_rel, t_rel

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
model.cuda()
model.load_state_dict(torch.load(cpt_path))
model.eval()

#* load siamese model
siamese_model_path = f'{args.siamese_ckpt}'
siamese_model = sim_mae_vit_small_patch8_dec512d8b()
siamese_model = nn.DataParallel(siamese_model).to("cuda")
siamese_model.load_state_dict(torch.load(siamese_model_path))
siamese_model.eval()

# load dataloader
from src.data.realestate.re10k_dataset import Re10k_dataset
dataset_abs = Re10k_dataset(data_root=f"{args.data_path}",mode="test",infer_len=args.len)

test_loader_abs = torch.utils.data.DataLoader(
        dataset_abs,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last = True
    )

def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x+1.0)*127.5
    x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def normalize_tensor(tensor):
    # 找到 tensor 的最小值和最大值
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # 如果最小值等於最大值，則返回全0張量，避免除以0的情況
    if min_val == max_val:
        return torch.zeros_like(tensor)

    # 歸一化到 [0, 1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def as_grayscale_image(tensor):
    # tensor shape: (1, h, w)
    img = tensor.cpu().numpy()  # 移除 batch 維度，轉為 numpy
    img = (img * 255).astype('uint8')  # 假設 tensor 是 [0,1] 範圍，轉換成 0-255
    resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(resized_img)

def evaluate_per_batch(temp_model, batch, total_time_len = 20, time_len = 1, show = False,siamese=False):
    video_clips = []
    video_clips.append(batch["rgbs"][:, :, 0, ...])

    # first generate one frame
    with torch.no_grad():
        for i in range(1):
            conditions = []
            R_src = batch["R_s"][0, i, ...]
            R_src_inv = R_src.transpose(-1,-2)
            t_src = batch["t_s"][0, i, ...]

            # create dict
            example = dict()
            example["K"] = batch["K"]
            example["K_inv"] = batch["K_inv"]

            #* 讀取新的src img, image encoding
            example["src_img"] = video_clips[-1]
            _, c_indices = temp_model.encode_to_c(example["src_img"])
            c_emb = temp_model.transformer.tok_emb(c_indices)
            conditions.append(c_emb)

            R_dst = batch["R_s"][0, i+1, ...]
            t_dst = batch["t_s"][0, i+1, ...]

            #* 計算前兩張image 的relative camera
            R_rel = R_dst@R_src_inv
            t_rel = t_dst-R_rel@t_src

            example["R_rel"] = R_rel.unsqueeze(0)
            example["t_rel"] = t_rel.unsqueeze(0)

            #* camera encoding
            embeddings_warp = temp_model.encode_to_e(example)
            conditions.append(embeddings_warp)

            # p1 不重要
            p1 = temp_model.encode_to_p(example)

            prototype = torch.cat(conditions, 1) #* (1,286,1024) img0+cam01 
            z_start_indices = c_indices[:, :0]

            #* 生成圖片與每個token 對應的epipolar ratio
            index_sample,bi_epi_ratio = temp_model.sample_latent(z_start_indices, prototype, [p1, None, None],
                                        steps=c_indices.shape[1],
                                        k_ori=batch["K_ori"],w2c=batch['w2c_seq'][:,i:i+3,...],
                                        temperature=1.0,
                                        sample=False,
                                        top_k=100,
                                        callback=lambda k: None,
                                        embeddings=None)

            #* shape (1 3 256 256)
            sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
            video_clips.append(sample_dec) # update video_clips list


            #* 因為siamese 是使用8x8 的mask,lor 則是使用16x16的token, 所以要做一層對應的轉換
            #* 然後siamese 是取前幾小的做保留其餘做mask, 但bi_epi_ratio 則是想保留前幾大的, 所以都取了倒數進行對應
            masked_token = torch.zeros((32, 32))
            for j in range(32):
                for k in range(32):
                    masked_token[j][k] = 1/bi_epi_ratio[0,(j//2)*16+(k//2)]
            masked_token = rearrange(masked_token,'h w -> (h w)')
            #* 從小到大排序, 數字越小代表生成的越好, 需要保留
            sorted_indices = torch.argsort(masked_token)
            small_index = sorted_indices[:int(1024*0.5)]

            masked_token = masked_token.unsqueeze(0)

    #! siamese

            if siamese==True:
                for k in range(5):
                    #* 拿src img 和 最新predict 的image 去做 siamese 
                    pred_images = []
                    for j in range(args.mix_frame):

                        #* 將保留的token 進行拆分分別做recon，以免每次都算到一樣的
                        sliced_mask_token = masked_token.clone()
                        shuffled_tensor = small_index[torch.randperm(small_index.size(0))]
                        #* 0代表要保留不做mask 的區塊
                        for l in range(int(1024*(1-args.mask_ratio))):
                            sliced_mask_token[0,shuffled_tensor[l]]=0

                        #* data shape (b c t h w) (1 3 2 256 256)
                        siamese_data = torch.stack([video_clips[-2],video_clips[-1]],dim=2)
                        #* 將mask 區域進行reconstruct
                        loss, pred = siamese_model.forward(siamese_data,mask_ratio=args.mask_ratio,mask_example=sliced_mask_token)
                        pred_image = siamese_model.module.unpatchify(pred)
                        pred_images.append(pred_image)


                    #* 平均多次計算的結果
                    pred_image = pred_images[0]
                    for j in range(1,args.mix_frame):
                        pred_image += pred_images[j]
                    pred_image/=args.mix_frame

                    #* 替換最後結果
                    video_clips[-1] = pred_image
                current_im = as_png(pred_image.permute(0,2,3,1)[0])

#! -----------------


    with torch.no_grad():
        for i in range(0, total_time_len-2, time_len):
            conditions = []

            #* cam 0
            R_src = batch["R_s"][0, i, ...]
            R_src_inv = R_src.transpose(-1,-2)
            t_src = batch["t_s"][0, i, ...]

            print(f"i={i}")

            # create dict
            example = dict()
            example["K"] = batch["K"]
            example["K_inv"] = batch["K_inv"]

            for t in range(time_len):
                #* 取得新的img0 並encode
                example["src_img"] = video_clips[-2]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i+t+1, ...]
                t_dst = batch["t_s"][0, i+t+1, ...]

                #* 計算img01 之間的relative camera
                R_rel = R_dst@R_src_inv
                t_rel = t_dst-R_rel@t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p1 不重要
                p1 = temp_model.encode_to_p(example)

                #* 取得新的img1 並encode
                example["src_img"] = video_clips[-1]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i+t+2, ...]
                t_dst = batch["t_s"][0, i+t+2, ...]

                #* 計算img02 之間的relative camera
                R_rel = R_dst@R_src_inv
                t_rel = t_dst-R_rel@t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p2 不重要
                p2 = temp_model.encode_to_p(example)
                
                R_rel, t_rel = compute_camera_pose(batch["R_s"][0, i+t+2, ...], batch["t_s"][0, i+t+2, ...], 
                                                   batch["R_s"][0, i+t+1, ...], batch["t_s"][0, i+t+1, ...])
                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                # p3 不重要
                p3 = temp_model.encode_to_p(example)

                #* img0 + cam01 + img1 + cam02
                prototype = torch.cat(conditions, 1)

                z_start_indices = c_indices[:, :0]

                #* 生成圖片與每個token 對應的epipolar ratio
                index_sample,bi_epi_ratio = temp_model.sample_latent(z_start_indices, prototype, [p1, p2, p3],
                                        steps=c_indices.shape[1],
                                        k_ori=batch["K_ori"],w2c=batch['w2c_seq'][:,i:i+3,...],
                                        temperature=1.0,
                                        sample=False,
                                        top_k=100,
                                        callback=lambda k: None,
                                        embeddings=None)

                print(f"sample shape = {index_sample.shape}")
                sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
                video_clips.append(sample_dec) # update video_clips list
                current_im = as_png(sample_dec.permute(0,2,3,1)[0])

                
                #* 因為siamese 是使用8x8 的mask,lor 則是使用16x16的token, 所以要做一層對應的轉換
                #* 然後siamese 是取前幾小的做保留其餘做mask, 但bi_epi_ratio 則是想保留前幾大的, 所以都取了倒數進行對應
                masked_token = torch.zeros((32, 32))
                for j in range(32):
                    for k in range(32):
                        masked_token[j][k] = 1/bi_epi_ratio[0,(j//2)*16+(k//2)]
                masked_token = rearrange(masked_token,'h w -> (h w)')
                #* 從小到大排序, 數字越小代表生成的越好, 需要保留
                sorted_indices = torch.argsort(masked_token)
                small_index = sorted_indices[:int(1024*0.5)]
                masked_token = masked_token.unsqueeze(0)

        #! siamese 
                if siamese==True:       
                    if i%2==0:
                        continue
                    print(f"do siamese for frame {len(video_clips)-1}...")

                    #* 拿src img 和 最新predict 的image 去做 siamese 
                    for k in range(5):
                        pred_images = []
                        for j in range(args.mix_frame):

                            #* 將保留的token 進行拆分分別做recon，以免每次都算到一樣的
                            sliced_mask_token = masked_token.clone()
                            shuffled_tensor = small_index[torch.randperm(small_index.size(0))]
                            for l in range(int(1024*(1-args.mask_ratio))):
                                sliced_mask_token[0,shuffled_tensor[l]]=0

                            #* data shape (b c t h w) (1 3 2 256 256)
                            siamese_data = torch.stack([video_clips[-2],video_clips[-1]],dim=2)
                            #* 將mask 區域進行reconstruct
                            loss, pred = siamese_model.forward(siamese_data,mask_ratio=args.mask_ratio,mask_example=sliced_mask_token)
                            pred_image = siamese_model.module.unpatchify(pred)
                            pred_images.append(pred_image)

                        #* 平均多次計算的結果
                        pred_image = pred_images[0]
                        for j in range(1,args.mix_frame):
                            pred_image += pred_images[j]

                        pred_image/=args.mix_frame

                        #* 替換最後結果
                        video_clips[-1] = pred_image
                    current_im = as_png(pred_image.permute(0,2,3,1)[0])


         #!  ----------------
                
    return video_clips

# first save the frame and then evaluate the saved frame 
n_values_percsim = []
n_values_ssim = []
n_values_psnr = []

pbar = tqdm(total=1000)
b_i = 0     #* 生成了幾個video
iteration = iter(test_loader_abs)
skip_num = 0    #* 從第幾個video開始生成
cnt = 0
while b_i < video_limit:    

    try:
        batch, index, inter_index = next(iteration)
        #* 跳過skip_num 個video
        if cnt<skip_num:
            cnt+=1
            pbar.update(1)
            continue
    except:
        continue
                
    pbar.update(1)
        
    values_percsim = []
    values_ssim = []
    values_psnr = []
        

    sub_dir = os.path.join(target_save_path, f'{"%03d" % (cnt+b_i)}-mix{args.mix_frame}-mask{args.mask_ratio}')
    os.makedirs(sub_dir, exist_ok=True)
            
    for key in batch.keys():
        batch[key] = batch[key].cuda()
    
    #* 每個video生成frame_limit 張圖片
    generate_video = evaluate_per_batch(model, batch, total_time_len = frame_limit, time_len = 1,siamese=True)
    
    #* 保存生成的圖片
    for i in range(1, len(generate_video)):
        gt_img = np.array(as_png(batch["rgbs"][0, :, i, ...].permute(1,2,0)))
        forecast_img = np.array(as_png(generate_video[i][0].permute(1,2,0)))
        
        cv2.imwrite(os.path.join(sub_dir, "predict_%02d.png" % i), forecast_img[:, :, [2,1,0]])
        cv2.imwrite(os.path.join(sub_dir, "gt_%02d.png" % i), gt_img[:, :, [2,1,0]])

    #* 計算生成的前5張與GT 的指標就好 (short term)
    for i in range(1, 6): 
        t_img = (batch["rgbs"][:, :, i, ...] + 1)/2
        p_img = (generate_video[i] + 1)/2
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
    
    b_i += 1
    
pbar.close()
    
total_percsim = []
total_ssim = []
total_psnr = []

with open(os.path.join(target_save_path, "eval_3.txt"), 'w') as f:
    #* 將所有video的計算結果寫入txt, 並計算平均結果
    for i in range(len(n_values_percsim)):

        f.write("#%d, percsim: %.04f, ssim: %.02f, psnr: %.02f" % (i, np.mean(n_values_percsim[i]), 
                                                                   np.mean(n_values_ssim[i]), np.mean(n_values_psnr[i])))
        f.write('\n')
        
        for j in range(len(n_values_percsim[i])):
            percsim = n_values_percsim[i][j]
            ssim = n_values_ssim[i][j]
            psnr = n_values_psnr[i][j]

            total_percsim.append(percsim)
            total_ssim.append(ssim)
            total_psnr.append(psnr)
            
    f.write("Total, percsim: (%.03f, %.02f), ssim: (%.02f, %.02f), psnr: (%.03f, %.02f)" 
            % (np.mean(total_percsim), np.std(total_percsim), 
               np.mean(total_ssim), np.std(total_ssim),
               np.mean(total_psnr), np.std(total_psnr)))
    f.write('\n')