import os
import shutil

exp = "exp_alternately_epipolar_error"
num = "01"

src = f"experiments/realestate/{exp}/evaluate_frame_21_video_200_gap_10"
target = f"fid_cal/{exp}"
target_gt = f"{target}/gt/{num}"
target_pred = f"{target}/pred/{num}"

os.makedirs(target,exist_ok=True)
os.makedirs(target_gt,exist_ok=True)
os.makedirs(target_pred,exist_ok=True)

for folder in os.listdir(src):
    if folder == "eval.txt":
        continue
    src_gt_file = f"{src}/{folder}/gt_{num}.png"
    src_pred_file = f"{src}/{folder}/predict_{num}.png"

    target_gt_file = f"{target_gt}/{folder}_gt.png"
    target_pred_file = f"{target_pred}/{folder}_predict.png"

    shutil.copy2(src_gt_file, target_gt_file)
    shutil.copy2(src_pred_file, target_pred_file)

