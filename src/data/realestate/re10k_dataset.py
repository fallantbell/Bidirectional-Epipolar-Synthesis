import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import rearrange
import re

#! ------------------------------------------ 
#! LOR utils

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True
def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip
def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0

class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return to_tensor(clip)

    # def __repr__(self) -> str:
    #     return self.__class__.__name__

class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"
    

#! ------------------------------------------ 

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

def custom_sort(file_name):
    # 提取文件名中的數字部分
    num_part = re.findall(r'\d+', file_name)
    if num_part:
        return int(num_part[0])  # 將數字部分轉換為整數進行排序
    else:
        return file_name

class Re10k_dataset(Dataset):
    def __init__(self,data_root,mode,max_interval=5,midas_transform = None,infer_len = 20,do_latent = False):
        assert mode == 'train' or mode == 'test' or mode == 'finetune'

        self.mode = mode

        #* change to your own root
        self.inform_root = '{}/RealEstate10K/{}'.format(data_root, mode)
        self.image_root = '{}/realestate_4fps/{}'.format(data_root, mode)

        if self.mode == "finetune":
            self.inform_root = '{}/RealEstate10K/{}'.format(data_root, "train")
            self.image_root = '{}/realestate_4fps/{}'.format(data_root, "train")

        # self.transform = default_transform
        self.transform = transforms.Compose(
        [
            ToTensorVideo(),
            NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        # self.midas_transform = midas_transform

        self.max_interval = max_interval   #* 兩張圖片最大的間隔
        
        self.infer_len = infer_len  #* inference 時要生成的影片長度

        self.video_dirs = []
        self.image_dirs = []
        self.inform_dirs = []
        self.total_img = 0

        #* 原始圖像大小
        H = 360
        W = 640

        #* 256 x 256 縮放版本
        H = 256
        W = 455

        self.H = H
        self.W = W

        self.square_crop = True     #* 是否有做 center crop 

        xscale = W / min(H, W)      #* crop 之後 cx cy 的縮放比例
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale

        #* 取10000 video 做training
        num = 0
        for video_dir in sorted(os.listdir(self.image_root)):
            self.video_dirs.append(video_dir)
            num+=1
            if num>=10000:
                break

        print(f"video num: {len(self.video_dirs)}")
        print(f"load {mode} data finish")
        print(f"-------------------------------------------")


    def __len__(self):
        return len(self.video_dirs) 
    
    def get_image(self,index):

        #* 選哪一個video
        video_idx = index

        #* 讀取video 每個frame 的檔名
        frame_namelist = []
        video_dir = self.video_dirs[video_idx]
        npz_file_path = f"{self.image_root}/{video_dir}/data.npz"
        if os.path.isfile(npz_file_path) == False:
            return None, None, None, None, False
        npz_file = np.load(npz_file_path)

        for file_name in sorted(npz_file.files, key=custom_sort):
            frame_namelist.append(file_name)

        if len(frame_namelist) <= self.max_interval:
            return None, None, None, None, False
        
        if self.mode=="test" and len(frame_namelist) < self.infer_len:   #* inference 時影片長度小於要生成的長度
            return None, None, None, None, False

        #* 圖片張數
        if self.mode=="train":
            interval_len = 3
        if self.mode=="finetune":
            interval_len = 5
        if self.mode=="test":
            interval_len = 20

        #* 隨機取origin frame
        frame_idx = np.random.randint(len(frame_namelist)-interval_len)

        #* 取得圖片
        image_seq = []
        if self.mode=="train":
            frame_idxs = [frame_idx, frame_idx+1,frame_idx+2]
        if self.mode=="finetune":
            frame_idxs = [frame_idx, frame_idx+1,frame_idx+2,frame_idx+3,frame_idx+4]
        if self.mode == "test":     #* 做 inference 取 infer_len 張圖片，用來做比較
            frame_idxs = np.arange(self.infer_len)

        #* resize 並center crop 到256x256
        cnt = 0
        for idx in frame_idxs:
            frame_name = frame_namelist[idx]
            img_np = npz_file[frame_name]
            img = Image.fromarray(img_np)
            img = img.resize((self.W,self.H), resample=Image.LANCZOS)
            img = self.crop_image(img)      #* (256,256)

            image_seq.append(np.array(img))
            cnt += 1
        
        image_seq = torch.from_numpy(np.stack(image_seq))
        image_seq = self.transform(image_seq)

        return image_seq,frame_idx,interval_len,frame_namelist, True


    def get_information(self,index,frame_idx,interval_len,frame_namelist):

        #* 讀取選定video 的 information txt
        video_idx = index
        video_dir = self.video_dirs[video_idx]
        inform_path = '{}/{}.txt'.format(self.inform_root,video_dir)

        frame_num = -1
        frame_list = []

        with open(inform_path, 'r') as file:
            for line in file:
                frame_num+=1
                if frame_num==0:
                    continue
                frame_informlist = line.split()
                frame_list.append(frame_informlist)

        #* 讀取圖片的 intrinsic 
        fx,fy,cx,cy = np.array(frame_list[0][1:5], dtype=float)

        intrinsics = np.array([ [fx,0,cx],
                                [0,fy,cy],
                                [0,0,1]])
        
        intrinsics_ori = np.array([ [fx,0,cx],
                                [0,fy,cy],
                                [0,0,1]])
        
        #* unnormalize
        intrinsics[0] = intrinsics[0]*self.W
        intrinsics[1] = intrinsics[1]*self.H

        #* 調整 crop 後的 cx cy
        if self.square_crop:
            intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
            intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

        w2c_seq = []
        if self.mode == "train":
            frame_idxs = [frame_idx, frame_idx+1, frame_idx+2]
        if self.mode == "finetune":
            frame_idxs = [frame_idx, frame_idx+1,frame_idx+2,frame_idx+3,frame_idx+4]
        
        if self.mode == "test":      #* 做 inference 取 infer_len 張圖片，用來做比較
            frame_idxs = np.arange(self.infer_len)

        #* 讀取每張圖片的camera extrinsic
        f_idx = 0
        for idx in range(len(frame_list)):
            #* 根據timestamp 與圖片檔名配對，找到對應的extrinsic
            if int(frame_list[idx][0])//1000 != int(frame_namelist[frame_idxs[f_idx]].split('.')[0])//1000: 
                continue

            w2c = np.array(frame_list[idx][7:], dtype=float).reshape(3,4)
            w2c_4x4 = np.eye(4)
            w2c_4x4[:3,:] = w2c
            w2c_seq.append(w2c_4x4)

            f_idx+=1
            if f_idx == len(frame_idxs):
                break

        return intrinsics, w2c_seq , intrinsics_ori
    
    def crop_image(self,img):
        original_width, original_height = img.size

        # center crop 的大小
        new_width = min(original_height,original_width)
        new_height = new_width

        # 保留中心的crop 的部分
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        # 使用PIL的crop方法来截取图像
        cropped_img = img.crop((left, top, right, bottom))

        return cropped_img

    def __getitem__(self,index):

        img, frame_idx,interval_len, frame_namelist, good_video = self.get_image(index)

        #* video frame 數量 < max interval 
        #* 或 < inference 的長度
        if good_video == False:
            return self.__getitem__(index+1)

        intrinsics,w2c,intrinsics_ori = self.get_information(index,frame_idx,interval_len,frame_namelist)

        K = intrinsics
        K_ori = intrinsics_ori
        K_inv = np.linalg.inv(K)
        R_0 = w2c[0][:3,:3]
        t_0 = w2c[0][:3,3]
        R_1 = w2c[1][:3,:3]
        t_1 = w2c[1][:3,3]
        R_2 = w2c[2][:3,:3]
        t_2 = w2c[2][:3,3]

        # compute 
        R_0_inv = R_0.transpose(-1,-2)
        R_01 = R_1@R_0_inv
        t_01 = t_1-R_01@t_0

        R_02 = R_2@R_0_inv
        t_02 = t_2-R_02@t_0
        
        R_1_inv = R_1.transpose(-1,-2)
        R_12 = R_2@R_1_inv
        t_12 = t_2-R_12@t_1
        
        w2c_tensor = [] 
        for i in range(len(w2c)):
            w2c_tensor.append(torch.tensor(w2c[i]))
        w2c_tensor = torch.stack(w2c_tensor)

        #* training 回傳
        example = {
            "rgbs": img,
            "src_points": np.zeros((1,3), dtype=np.float32),
            "K": K.astype(np.float32),
            "K_ori": K_ori.astype(np.float32),
            "K_inv": K_inv.astype(np.float32),
            "R_01": R_01.astype(np.float32),
            "t_01": t_01.astype(np.float32),
            "R_02": R_02.astype(np.float32),
            "t_02": t_02.astype(np.float32),
            "R_12": R_12.astype(np.float32),
            "t_12": t_12.astype(np.float32),
            "w2c_seq": w2c_tensor,
        }

        Rs = []
        ts = []
        for i in range(len(w2c)):
            Rs.append(w2c[i][:3,:3])
            ts.append(w2c[i][:3,3])
        
        #* finetune 回傳
        example_finetune = {
            "rgbs": img,
            "src_points": np.zeros((1,3), dtype=np.float32),
            "K": K.astype(np.float32),
            "K_ori": K_ori.astype(np.float32),
            "K_inv": K_inv.astype(np.float32),
            "R_s": np.stack(Rs).astype(np.float32),
            "t_s": np.stack(ts).astype(np.float32),
            "w2c_seq": w2c_tensor,
        }

        #* testing 回傳
        example_test = {
            "rgbs": img,
            "src_points": np.zeros((1,3), dtype=np.float32),
            "K": K.astype(np.float32),
            "K_ori": K_ori.astype(np.float32),
            "K_inv": K_inv.astype(np.float32),
            "R_s": np.stack(Rs).astype(np.float32),
            "t_s": np.stack(ts).astype(np.float32),
            "w2c_seq": w2c_tensor,
        }

        if self.mode == "train":
            return example
        elif self.mode == "test":
            return example_test,0,0
        elif self.mode == "finetune":
            return example_finetune


if __name__ == '__main__':
    test = Re10k_dataset("../dataset","train")
    # print(test.video_dirs[:10])
    data = test[0]
    print(data["rgbs"].shape)
    # print(data['img'].shape)
    # print(data['intrinsics'])
    # print(data['w2c'][0])
    # print(test.__len__())
    # print(test.interval_len)

    # for i in range(data['img'].shape[0]):
    #     image = data['img'][i].numpy()
    #     image = (image+1)/2
    #     image *= 255
    #     image = image.astype(np.uint8)
    #     image = rearrange(image,"C H W -> H W C")
    #     image = Image.fromarray(image)
    #     image.save(f"../test_folder/test_{i}.png")