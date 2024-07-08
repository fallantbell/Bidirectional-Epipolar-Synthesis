#!/bin/bash

# python -m torch.distributed.launch \
# --nproc_per_node=4 --master_port=14901 \
# main.py \
# --dataset realestate --name exp_bidirectional_epipolar_full \
# --base ./configs/realestate/realestate_16x16_sine_cview_adaptive_epipolar.yaml \
# --max_iter 400001 --visual-iter 10000 \
# --gpu 6,7,8,9

python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=14902 \
error_accumulation.py \
--dataset realestate --name exp_bidirectional_epipolar_full_error \
--base ./configs/realestate/realestate_16x16_sine_cview_adaptive_epipolar_error.yaml \
--gpu 6,7,8,9

python ./evaluation/evaluate_realestate.py \
--len 21 --video_limit 200 \
--base realestate_16x16_sine_cview_adaptive_epipolar \
--exp exp_bidirection_epipolar_full_error --gpu 9