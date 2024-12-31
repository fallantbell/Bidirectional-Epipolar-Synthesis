import math
import random
import os
cpu_num = 4  # Num of CPUs you want to use
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


# args
parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--base", type=str, default="realestate_16x16_adaptive",
                    help="experiments name")
parser.add_argument("--exp", type=str, default="exp_1_error",
                    help="experiments name")
parser.add_argument("--ckpt", type=str, default="last",
                    help="checkpoint name")
parser.add_argument("--data-path", type=str, default="/latent_opt_test/RealEstate10K_Downloader/",
                    help="data path")
parser.add_argument("--len", type=int, default=4, help="len of prediction")
parser.add_argument('--gpu', default= '0', type=str)
parser.add_argument("--video_limit", type=int, default=20, help="# of video to test")
parser.add_argument("--gap", type=int, default=10, help="")
parser.add_argument("--seed", type=int, default=2333, help="")
parser.add_argument("--GT_start", action='store_true')
parser.add_argument("--mask_ratio", type=float, default=0.9, help="")
parser.add_argument("--mix_frame", type=int, default=10, help="")
parser.add_argument("--cross", action='store_true')
parser.add_argument("--type",type=str, default='forward')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# fix the seed
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

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
siamese_model_path = 'Siamese_folder/mask095_fulldata_epoch_42.pt'
siamese_model = sim_mae_vit_small_patch8_dec512d8b()
siamese_model = nn.DataParallel(siamese_model).to("cuda")
siamese_model.load_state_dict(torch.load(siamese_model_path))
siamese_model.eval()

# load dataloader
# from src.data.realestate.realestate_sample import VideoDataset
from src.data.realestate.re10k_dataset import Re10k_dataset
sparse_dir = "%s/sparse/" % args.data_path
image_dir = "%s/dataset/" % args.data_path
# dataset_abs = VideoDataset(sparse_dir = sparse_dir, image_dir = image_dir, length = args.len, low = args.gap, high = args.gap, split = "test")
dataset_abs = Re10k_dataset(data_root="../dataset",mode="test",infer_len=args.len)

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
        conditions = []
        R_src = batch["R_s"][0, 0, ...]
        R_src_inv = R_src.transpose(-1,-2)
        t_src = batch["t_s"][0, 0, ...]

        # create dict
        example = dict()
        example["K"] = batch["K"]
        example["K_inv"] = batch["K_inv"]

        example["src_img"] = video_clips[-1]
        _, c_indices = temp_model.encode_to_c(example["src_img"])
        c_emb = temp_model.transformer.tok_emb(c_indices)
        conditions.append(c_emb)

        R_dst = batch["R_s"][0, 1, ...]
        t_dst = batch["t_s"][0, 1, ...]

        R_rel = R_dst@R_src_inv
        t_rel = t_dst-R_rel@t_src

        example["R_rel"] = R_rel.unsqueeze(0)
        example["t_rel"] = t_rel.unsqueeze(0)

        embeddings_warp = temp_model.encode_to_e(example)
        conditions.append(embeddings_warp)
        # p1
        p1 = temp_model.encode_to_p(example)

        prototype = torch.cat(conditions, 1) #* (1,286,1024) rgb1+camera 的embed
        z_start_indices = c_indices[:, :0]

        if args.cross==False:
            index_sample,bi_epi_ratio = temp_model.sample_latent(z_start_indices, prototype, [p1, None, None],
                                        steps=c_indices.shape[1],
                                        k_ori=batch["K_ori"],w2c=batch['w2c_seq'][:,0:3,...],
                                        temperature=1.0,
                                        sample=False,
                                        top_k=100,
                                        callback=lambda k: None,
                                        embeddings=None)

        if args.cross:
            index_sample = temp_model.sample_cross(prototype[:, -286:, :],prototype[:, :0, :],
                                                k_ori=batch["K_ori"],w2c=batch['w2c_seq'][:,0:2,...],
                                                top_k=100,)

        #* shape (1 3 256 256)
        sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
        video_clips.append(sample_dec) # update video_clips list
        current_im = as_png(sample_dec.permute(0,2,3,1)[0])
        # current_im.save(f'{ckpt_test_folder}/ori_0.png')

        #* 因為siamese 是使用8x8 的mask,lor 則是使用16x16的token, 所以要做一層對應的轉換
        #* 然後siamese 是取前幾小的做保留其餘做mask, 但bi_epi_ratio 則是想保留前幾大的, 所以都取了倒數進行對應
        masked_token = torch.zeros((32, 32))
        for j in range(32):
            for k in range(32):
                masked_token[j][k] = 1/bi_epi_ratio[0,(j//2)*16+(k//2)]
        masked_token = rearrange(masked_token,'h w -> (h w)')

        sorted_indices = torch.argsort(masked_token)
        small_index = sorted_indices[:int(1024*0.5)]
        # idx_im = np.ones((32, 32))
        # for j in range(len(small_index)):
        #     sm_idx = small_index[j]
        #     idx_im[sm_idx//32,sm_idx%32] = 0
        
        # recon_token = cv2.resize(idx_im, (256, 256), interpolation=cv2.INTER_NEAREST)
        # plt.imshow(recon_token, cmap="gray")
        # plt.axis("off")
        # plt.savefig(f'{ckpt_test_folder}/test32.png')

        masked_token = masked_token.unsqueeze(0)
        # masked_token = None
        # sys.exit()
#! siamese

        if siamese==True:
            # current_im.save(f"experiments/realestate/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/evaluate_frame_5_video_250_ckpt_100000/005-mix20-mask0.9/ori_0.png")

            for k in range(5):
                # #* 拿src img 和 最新predict 的image 去做 siamese 
                pred_images = []
                for j in range(args.mix_frame):

                    #* 將保留的token 進行拆分分別做recon，以免每次都算到一樣的
                    sliced_mask_token = masked_token.clone()
                    shuffled_tensor = small_index[torch.randperm(small_index.size(0))]
                    for l in range(int(1024*(1-args.mask_ratio))):
                        sliced_mask_token[0,shuffled_tensor[l]]=0
                    # for l in range(int(j*1024*0.05),int((j+1)*1024*0.05)):
                    #     sliced_mask_token[0,sorted_indices[l]]=0

                    #* data shape (b c t h w) (1 3 2 256 256)
                    siamese_data = torch.stack([video_clips[-2],video_clips[-1]],dim=2)
                    loss, pred = siamese_model.forward(siamese_data,mask_ratio=args.mask_ratio,mask_example=sliced_mask_token)
                    pred_image = siamese_model.module.unpatchify(pred)
                    # pred_image = normalize_tensor(pred_image)
                    pred_images.append(pred_image)

                    # if k==0 and j==0:
                    #     current_im = as_png(pred_image.permute(0,2,3,1)[0])
                    #     current_im.save(f'{ckpt_test_folder}/mask_recon.png')
                    #     sys.exit()

                #* 平均多次計算的結果
                pred_image = pred_images[0]
                for j in range(1,args.mix_frame):
                    pred_image += pred_images[j]
                pred_image/=args.mix_frame
                
                # current_im = as_png(pred_image.permute(0,2,3,1)[0])
                # current_im.save(f'{ckpt_test_folder}/mask_recon_{k}.png')
                # print(f"save recon {k} ...")

                #* 替換最後結果
                video_clips[-1] = pred_image
            current_im = as_png(pred_image.permute(0,2,3,1)[0])

#! -----------------

        # if show:
        #     plt.imshow(current_im)
        #     plt.show()

    #* 主要想測試看看epipolar 會不會因為第二張生成不好而受到很大的影響
    if args.GT_start == True:
        video_clips = []
        video_clips.append(batch["rgbs"][:, :, 0, ...])
        video_clips.append(batch["rgbs"][:, :, 1, ...])

    # then generate second
    with torch.no_grad():
        for i in range(0, total_time_len-2, time_len):
            conditions = []

            R_src = batch["R_s"][0, i, ...]
            R_src_inv = R_src.transpose(-1,-2)
            t_src = batch["t_s"][0, i, ...]

            print(f"i={i}")

            # create dict
            example = dict()
            example["K"] = batch["K"]
            example["K_inv"] = batch["K_inv"]

            for t in range(time_len):
                example["src_img"] = video_clips[-2]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i+t+1, ...]
                t_dst = batch["t_s"][0, i+t+1, ...]

                R_rel = R_dst@R_src_inv
                t_rel = t_dst-R_rel@t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                # print(f"data device = {R_rel.device}")
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p1
                p1 = temp_model.encode_to_p(example)

                example["src_img"] = video_clips[-1]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i+t+2, ...]
                t_dst = batch["t_s"][0, i+t+2, ...]

                R_rel = R_dst@R_src_inv
                t_rel = t_dst-R_rel@t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p2
                p2 = temp_model.encode_to_p(example)
                # p3
                R_rel, t_rel = compute_camera_pose(batch["R_s"][0, i+t+2, ...], batch["t_s"][0, i+t+2, ...], 
                                                   batch["R_s"][0, i+t+1, ...], batch["t_s"][0, i+t+1, ...])
                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                p3 = temp_model.encode_to_p(example)

                #* ------------------------------------------------------------------
                #* for cross
                #* 更改camera pose , 原本取最後是取到 cam 0 跟 cam 2 , 但是應該取的是 cam 1 跟 cam 2 

                if args.cross == True:
                    conditions = []

                    example["src_img"] = video_clips[-1]
                    _, c_indices = temp_model.encode_to_c(example["src_img"])
                    c_emb = temp_model.transformer.tok_emb(c_indices)
                    conditions.append(c_emb)

                    R_rel, t_rel = compute_camera_pose(batch["R_s"][0, i+t+2, ...], batch["t_s"][0, i+t+2, ...], 
                                                   batch["R_s"][0, i+t+1, ...], batch["t_s"][0, i+t+1, ...])
                    example["R_rel"] = R_rel.unsqueeze(0)
                    example["t_rel"] = t_rel.unsqueeze(0)
                    embeddings_warp = temp_model.encode_to_e(example)
                    conditions.append(embeddings_warp)

                #* ------------------------------------------------------------------

                prototype = torch.cat(conditions, 1)

                z_start_indices = c_indices[:, :0]

                if args.cross==False:
                    if show == True:
                        index_sample,return_attn_map,return_attn,idx,x_second,x_third,bi_epi_ratio = temp_model.sample_latent(z_start_indices, prototype, [p1, p2, p3],
                                                steps=c_indices.shape[1],
                                                k_ori=batch["K_ori"],w2c=batch['w2c_seq'][:,i:i+3,...],
                                                temperature=1.0,
                                                sample=False,
                                                top_k=100,
                                                callback=lambda k: None,
                                                embeddings=None,
                                                show=True)
                    else:
                        index_sample,bi_epi_ratio = temp_model.sample_latent(z_start_indices, prototype, [p1, p2, p3],
                                                steps=c_indices.shape[1],
                                                k_ori=batch["K_ori"],w2c=batch['w2c_seq'][:,i:i+3,...],
                                                temperature=1.0,
                                                sample=False,
                                                top_k=100,
                                                callback=lambda k: None,
                                                embeddings=None)
                if args.cross:
                    index_sample = temp_model.sample_cross(prototype[:, -286:, :],prototype[:, :0, :],
                                                       k_ori=batch["K_ori"],w2c=batch['w2c_seq'][:,i+1:i+3,...],
                                                       top_k=100,)

                print(f"sample shape = {index_sample.shape}")
                sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
                # sample_dec = normalize_tensor(sample_dec)
                video_clips.append(sample_dec) # update video_clips list
                current_im = as_png(sample_dec.permute(0,2,3,1)[0])
                # current_im.save(f'{ckpt_test_folder}/ori_{i+1}.png')

                
                #* 因為siamese 是使用8x8 的mask,lor 則是使用16x16的token, 所以要做一層對應的轉換
                #* 然後siamese 是取前幾小的做保留其餘做mask, 但bi_epi_ratio 則是想保留前幾大的, 所以都取了倒數進行對應
                masked_token = torch.zeros((32, 32))
                for j in range(32):
                    for k in range(32):
                        masked_token[j][k] = 1/bi_epi_ratio[0,(j//2)*16+(k//2)]
                masked_token = rearrange(masked_token,'h w -> (h w)')
                sorted_indices = torch.argsort(masked_token)
                small_index = sorted_indices[:int(1024*0.5)]
                masked_token = masked_token.unsqueeze(0)
                # masked_token = None

        #! siamese 
                if siamese==True:       
                    if i%2==0:
                        continue
                    print(f"do siamese for frame {len(video_clips)-1}...")
                    # current_im.save(f"experiments/realestate/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/evaluate_frame_5_video_250_ckpt_100000/005-mix20-mask0.9/ori_{i+1}.png")

                    #* 拿src img 和 最新predict 的image 去做 siamese 
                    for k in range(5):
                        pred_images = []
                        for j in range(args.mix_frame):

                            #* 將保留的token 進行拆分分別做recon，以免每次都算到一樣的
                            sliced_mask_token = masked_token.clone()
                            shuffled_tensor = small_index[torch.randperm(small_index.size(0))]
                            for l in range(int(1024*(1-args.mask_ratio))):
                                sliced_mask_token[0,shuffled_tensor[l]]=0
                            # for l in range(int(j*1024*0.05),int((j+1)*1024*0.05)):
                            #     sliced_mask_token[0,sorted_indices[l]]=0

                            #* data shape (b c t h w) (1 3 2 256 256)
                            siamese_data = torch.stack([video_clips[-2],video_clips[-1]],dim=2)
                            loss, pred = siamese_model.forward(siamese_data,mask_ratio=args.mask_ratio,mask_example=sliced_mask_token)
                            pred_image = siamese_model.module.unpatchify(pred)
                            # pred_image = normalize_tensor(pred_image)
                            pred_images.append(pred_image)

                        #* 平均多次計算的結果
                        pred_image = pred_images[0]
                        for j in range(1,args.mix_frame):
                            pred_image += pred_images[j]

                        pred_image/=args.mix_frame
                        # pred_image = torch.clamp(pred_image,0,1)

                        #* 替換最後結果
                        video_clips[-1] = pred_image
                    current_im = as_png(pred_image.permute(0,2,3,1)[0])

                    print(f"finish siamese for frame {len(video_clips)-1}...")

        #!  ----------------

                #* 取生成圖片中隨機的idx patch, 
                #* 看他在model中不同layer與 origin input attention map 的情況 
                if show:
                    current_im.save(f'{ckpt_test_folder}/normal.png')
                    x_second = temp_model.decode_to_img(x_second, [1, 256, 16, 16])
                    x_second = as_png(x_second.permute(0,2,3,1)[0])
                    x_second.save(f'{ckpt_test_folder}/second.png')
                    x_third = temp_model.decode_to_img(x_third, [1, 256, 16, 16])
                    x_third = as_png(x_third.permute(0,2,3,1)[0])
                    x_third.save(f'{ckpt_test_folder}/third.png')

                    print(f"bi ratio shape = {bi_epi_ratio.shape}")
                    # print(f"ratio max = {bi_epi_ratio.max()}")
                    # print(f"ratio min = {bi_epi_ratio.min()}")
                    bi_epi_ratio = bi_epi_ratio.squeeze()
                    #* 按照值大小升序排序排index
                    sorted_indices = torch.argsort(bi_epi_ratio)
                    print(f"worst index 5 = {sorted_indices[:5]}")
                    print(f"best index 5 = {sorted_indices[-5:]}")
                    #* attention 找最不好的token
                    small_index = sorted_indices[:int(256*0.8)]
                    idx_im = np.zeros((16, 16))
                    for j in range(len(small_index)):
                        sm_idx = small_index[j]
                        idx_im[sm_idx//16,sm_idx%16] = 1

                    recon_token = cv2.resize(idx_im, (256, 256), interpolation=cv2.INTER_NEAREST) 

                    for j in range(len(return_attn)):
                        print(f" j= {j}")
                        if j<15:
                            continue
                        attn_weight = torch.mean(return_attn_map[j], dim=1)
                        attn_map0 = attn_weight[0,571+idx,0:256]
                        attn_map0 = rearrange(attn_map0,"(h w) -> h w",h=16)
                        attn_map0 = normalize_tensor(attn_map0)
                        attn_map0 = as_grayscale_image(attn_map0)
                        attn_map1 = attn_weight[0,571+idx,286:542]
                        attn_map1 = rearrange(attn_map1,"(h w) -> h w",h=16)
                        attn_map1 = normalize_tensor(attn_map1)
                        attn_map1 = as_grayscale_image(attn_map1)

                        attn_weight = torch.mean(return_attn[j], dim=1)
                        attn_map4 = attn_weight[0,571+idx,0:256]
                        attn_map4 = rearrange(attn_map4,"(h w) -> h w",h=16)
                        attn_map4 = normalize_tensor(attn_map4)
                        attn_map4 = as_grayscale_image(attn_map4)
                        attn_map5 = attn_weight[0,571+idx,286:542]
                        attn_map5 = rearrange(attn_map5,"(h w) -> h w",h=16)
                        attn_map5 = normalize_tensor(attn_map5)
                        attn_map5 = as_grayscale_image(attn_map5)

                        # fig, axes = plt.subplots(3,3, figsize=(16, 16))
                        # ax1, ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9 = axes.flatten()

                        # ax1.set_title('prev0 img')
                        # prev0_im = as_png(video_clips[-3].permute(0,2,3,1)[0])
                        # ax1.imshow(attn_map0, cmap='gray', alpha=0.7)
                        # ax1.imshow(prev0_im, cmap='hot', alpha=0.3)

                        # ax2.set_title('prev1 img')
                        # prev1_im = as_png(video_clips[-2].permute(0,2,3,1)[0])
                        # ax2.imshow(attn_map1, cmap='gray', alpha=0.7)
                        # ax2.imshow(prev1_im, cmap='hot', alpha=0.3)

                        # ax3.set_title('cur img')
                        # idx_im = np.zeros((16, 16))
                        # idx_im[idx//16,idx%16] = 1
                        # idx_im = cv2.resize(idx_im, (256, 256), interpolation=cv2.INTER_NEAREST) 
                        # ax3.imshow(idx_im, cmap='gray', alpha=0.7)
                        # ax3.imshow(current_im, cmap='hot', alpha=0.3)

                        # ax4.set_title('prev0 after attn')
                        # prev0_im = as_png(video_clips[-3].permute(0,2,3,1)[0])
                        # ax4.imshow(attn_map4, cmap='gray', alpha=0.7)
                        # ax4.imshow(prev0_im, cmap='hot', alpha=0.3)

                        # ax5.set_title('prev1 after attn')
                        # prev1_im = as_png(video_clips[-3].permute(0,2,3,1)[0])
                        # ax5.imshow(attn_map5, cmap='gray', alpha=0.7)
                        # ax5.imshow(prev1_im, cmap='hot', alpha=0.3)

                        # ax6.set_title('epipolar02')
                        # epipolar02 = attn_weight[0,571+idx,0:256]
                        # epipolar02 = rearrange(epipolar02,"(h w) -> h w",h=16)
                        # epipolar02 = normalize_tensor(epipolar02)
                        # epipolar02 = as_grayscale_image(epipolar02)
                        # ax6.imshow(epipolar02,cmap='gray')

                        # ax7.set_title('epipolar12')
                        # epipolar12 = attn_weight[0,571+idx,286:542]
                        # epipolar12 = rearrange(epipolar12,"(h w) -> h w",h=16)
                        # epipolar12 = normalize_tensor(epipolar12)
                        # epipolar12 = as_grayscale_image(epipolar12)
                        # ax7.imshow(epipolar12,cmap='gray')

                        # ax8.set_title('cam0')
                        # cam0 = attn_weight[0,571+idx,256:286]
                        # cam0 = rearrange(cam0,"(h w) -> h w",h=6)
                        # cam0 = as_grayscale_image(cam0)
                        # ax8.imshow(cam0,cmap='gray')

                        # ax9.set_title('cam1')
                        # cam1 = attn_weight[0,571+idx,542:572]
                        # cam1 = rearrange(cam1,"(h w) -> h w",h=6)
                        # cam1 = as_grayscale_image(cam1)
                        # ax9.imshow(cam1,cmap='gray')

                        # ax7.axis('off')
                        # ax8.axis('off')
                        # ax9.axis('off')

                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        ax1, ax2,ax3 = axes.flatten()
                        # ax1.set_title('prev img')
                        # prev1_im = as_png(video_clips[-2].permute(0,2,3,1)[0])
                        # ax1.imshow(prev1_im, cmap='hot', alpha=1)

                        # ax2.set_title('cur im')
                        # ax2.imshow(current_im, cmap='hot', alpha=1)

                        ax1.set_title('mask token')
                        ax1.imshow(recon_token, cmap='gray', alpha=0.5)
                        ax1.imshow(current_im, cmap='hot', alpha=0.5)

                        ax2.set_title('prev img')
                        prev1_im = as_png(video_clips[-2].permute(0,2,3,1)[0])
                        ax2.imshow(attn_map1, cmap='gray', alpha=0.7)
                        ax2.imshow(prev1_im, cmap='hot', alpha=0.3)

                        ax3.set_title('cur img')
                        idx_im = np.zeros((16, 16))
                        idx_im[idx//16,idx%16] = 1
                        idx_im = cv2.resize(idx_im, (256, 256), interpolation=cv2.INTER_NEAREST) 
                        ax3.imshow(idx_im, cmap='gray', alpha=0.7)
                        ax3.imshow(current_im, cmap='hot', alpha=0.3)

                        plt.savefig(f'{ckpt_test_folder}/5_token29.png')
                    
                    print(f"done")
                    sys.exit()

                # if show:
                #     plt.imshow(current_im)
                #     plt.show()
                
    return video_clips

# first save the frame and then evaluate the saved frame 
n_values_percsim = []
n_values_ssim = []
n_values_psnr = []

pbar = tqdm(total=1000)
b_i = 0
iteration = iter(test_loader_abs)
skip_num = 100
cnt = 0
while b_i < video_limit:    

    try:
        batch, index, inter_index = next(iteration)
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
    
    generate_video = evaluate_per_batch(model, batch, total_time_len = frame_limit, time_len = 1,show=False,siamese=True)
        
    for i in range(1, len(generate_video)):
        gt_img = np.array(as_png(batch["rgbs"][0, :, i, ...].permute(1,2,0)))
        forecast_img = np.array(as_png(generate_video[i][0].permute(1,2,0)))
        
        cv2.imwrite(os.path.join(sub_dir, "predict_%02d.png" % i), forecast_img[:, :, [2,1,0]])
        cv2.imwrite(os.path.join(sub_dir, "gt_%02d.png" % i), gt_img[:, :, [2,1,0]])

    #* 計算生成的前5張與GT 的指標就好 (short term)
    #* 若是前兩張使用gt, 就順移一個
    add_one = 0
    if args.GT_start == True:
        add_one = 1

    for i in range(1+add_one, 6+add_one): 
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