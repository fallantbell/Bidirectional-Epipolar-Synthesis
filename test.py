import shutil
import torch
import os
import numpy as np
import shutil

# exp = "exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error"
# src = f"Siamese_folder"

# cnt = 0 
# for file in os.listdir(src):
#     file_path = os.path.join(src, file)
#     if file[6] == '1':
#         cnt +=1
#         os.remove(file_path)
# print(cnt)

noise = torch.rand(1, 5, device='cpu')
ids_shuffle = torch.argsort(noise, dim=1)
ids_restore = torch.argsort(ids_shuffle, dim=1)

print(noise)
print(ids_shuffle)
print(ids_restore)