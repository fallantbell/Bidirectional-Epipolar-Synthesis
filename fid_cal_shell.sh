#!/bin/bash

python -m pytorch_fid --device \
cuda:5 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/gt/01 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/pred/01 > fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/1.txt

python -m pytorch_fid --device \
cuda:5 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/gt/05 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/pred/05 > fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/5.txt

python -m pytorch_fid --device \
cuda:5 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/gt/10 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/pred/10 > fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/10.txt

python -m pytorch_fid --device \
cuda:5 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/gt/15 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/pred/15 > fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/15.txt

python -m pytorch_fid --device \
cuda:5 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/gt/20 \
fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/pred/20 > fid_cal/exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error/20.txt
