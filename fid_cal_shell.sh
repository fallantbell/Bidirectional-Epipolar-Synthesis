#!/bin/bash

python -m pytorch_fid --device \
cuda:4 \
fid_cal/exp_forward_epipolar_full_10kdata_error/gt/01 \
fid_cal/exp_forward_epipolar_full_10kdata_error/pred/01 > fid_cal/exp_forward_epipolar_full_10kdata_error/1.txt

python -m pytorch_fid --device \
cuda:4 \
fid_cal/exp_forward_epipolar_full_10kdata_error/gt/05 \
fid_cal/exp_forward_epipolar_full_10kdata_error/pred/05 > fid_cal/exp_forward_epipolar_full_10kdata_error/5.txt

python -m pytorch_fid --device \
cuda:4 \
fid_cal/exp_forward_epipolar_full_10kdata_error/gt/10 \
fid_cal/exp_forward_epipolar_full_10kdata_error/pred/10 > fid_cal/exp_forward_epipolar_full_10kdata_error/10.txt

python -m pytorch_fid --device \
cuda:4 \
fid_cal/exp_forward_epipolar_full_10kdata_error/gt/15 \
fid_cal/exp_forward_epipolar_full_10kdata_error/pred/15 > fid_cal/exp_forward_epipolar_full_10kdata_error/15.txt

python -m pytorch_fid --device \
cuda:4 \
fid_cal/exp_forward_epipolar_full_10kdata_error/gt/20 \
fid_cal/exp_forward_epipolar_full_10kdata_error/pred/20 > fid_cal/exp_forward_epipolar_full_10kdata_error/20.txt
