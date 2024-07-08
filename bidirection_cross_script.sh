#!/bin/bash

python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=14903 \
main.py \
--dataset realestate --name exp_bidirectional_cross \
--base ./configs/realestate/realestate_16x16_sine_cview_adaptive_epipolar_cross.yaml \
--len 2 --max_iter 400001 --visual-iter 10000 \
--gpu 0,1,2,3

python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=14904 \
error_accumulation.py \
--dataset realestate --name exp_bidirectional_cross_error \
--base ./configs/realestate/realestate_16x16_sine_cview_adaptive_epipolar_cross_error.yaml \
--gpu 0,1,2,3
