import os
import shutil

exp = "exp_bidirection_epipolar_error"
num = "04"

src = f"experiments/realestate/{exp}/evaluate_frame_21_video_200_gap_10"
target = f"fid_cal/{exp}"

#* 移動單個frame

# target_gt = f"{target}/gt/{num}"
# target_pred = f"{target}/pred/{num}"

# os.makedirs(target,exist_ok=True)
# os.makedirs(target_gt,exist_ok=True)
# os.makedirs(target_pred,exist_ok=True)

# for folder in os.listdir(src):
#     if folder == "eval.txt":
#         continue
#     src_gt_file = f"{src}/{folder}/gt_{num}.png"
#     src_pred_file = f"{src}/{folder}/predict_{num}.png"

#     target_gt_file = f"{target_gt}/{folder}_gt.png"
#     target_pred_file = f"{target_pred}/{folder}_predict.png"

#     shutil.copy2(src_gt_file, target_gt_file)
#     shutil.copy2(src_pred_file, target_pred_file)

#* 將多個frame 合併

target_gt = f"{target}/gt/1_5"
target_pred = f"{target}/pred/1_5"
os.makedirs(target_gt,exist_ok=True)
os.makedirs(target_pred,exist_ok=True)

cnt = 0
for num in ["01","02","03","04","05"]:
    src_gt_folder = f"fid_cal/exp_bidirection_epipolar_error/gt/{num}"

    for file in os.listdir(src_gt_folder):
        target_gt_file = f"{target_gt}/{cnt}.png"
        cnt = cnt+1
        src_gt_file = f"{src_gt_folder}/{file}"
        shutil.copy2(src_gt_file, target_gt_file)
    
cnt = 0
for num in ["01","02","03","04","05"]:
    src_pred_folder = f"fid_cal/exp_bidirection_epipolar_error/pred/{num}"

    for file in os.listdir(src_pred_folder):
        target_pred_file = f"{target_pred}/{cnt}.png"
        cnt = cnt+1
        src_pred_file = f"{src_pred_folder}/{file}"
        shutil.copy2(src_pred_file, target_pred_file)

