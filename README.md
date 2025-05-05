## Introduction

![image](https://github.com/fallantbell/Bidirectional-Epipolar-Synthesis/blob/master/assets/teaser.pdf)  

Novel view synthesis from a single image poses a significant challenge, which aims to gen-
erate novel scene views given a reference image and a sequence of camera poses. The primary
difficulty lies in effectively leveraging a generative model to achieve high-quality image gener-
ation while simultaneously ensuring consistency and faithfulness across synthesized views. In
this paper, we propose a novel approach to address the consistency and faithfulness issues in
view synthesis. Specifically, we develop a new attention layer, termed bidirectional epipolar at-
tention, which utilizes a pair of complementary epipolar lines to guide the associations between
features from different viewpoints. Each bidirectional epipolar layer calculates forward and
backward epipolar lines, enabling geometrically constrained attention that improves cross-view
consistency. To ensure faithful synthesis, we introduce an epipolar-aware reconstruction mod-
ule that prevents creating novel content in regions where the newly generated image overlaps
with existing ones. Extensive experimental results demonstrate that our method outperforms
previous approaches to novel view synthesis, achieving superior performance in both image
quality and consistency

![image](https://github.com/fallantbell/Bidirectional-Epipolar-Synthesis/blob/master/assets/overview.pdf)  

## Installation
- Clone the repository:
```
git clone https://github.com/fallantbell/Bidirectional-Epipolar-Synthesis
cd Bidirectional-Epipolar-Synthesis
python scripts/download_vqmodels.py
```
- Dependencies:  
```
conda env create -f environment.yaml
conda activate bi_epipolar
pip install opencv-python ffmpeg-python matplotlib tqdm omegaconf pytorch-lightning einops importlib-resources imageio imageio-ffmpeg numpy-quaternion
```

## Data preparation

### RealEstate10K:
1. Download the dataset from [RealEstate10K](https://google.github.io/realestate10k/).
2. Download videos from RealEstate10K dataset, decode videos into frames. You might find the [RealEstate10K_Downloader](https://github.com/cashiwamochi/RealEstate10K_Downloader) written by cashiwamochi helpful. 

## Training:

1. Train the model:
```
python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=14804 \
main.py \
--batch-size 2 \
--ckpt-iter 50000 \
--max_iter 400001 --visual-iter 2501 \
--dataset realestate --data-path ../dataset \
--name exp_bidirectional_epipolar \
--base ./configs/realestate/realestate_16x16_sine_cview_adaptive_epipolar.yaml \
--gpu 0,1,2,3
```

2. Train Siamese mask autoencoder:
```
CUDA_VISIBLE_DEVICES="6,7,8,9" \
python main_siam.py
```

## Evaluation:
Generate and evaluate the synthesis results:
```
python ./evaluation/evaluate_realestate.py \
--len 6 --video_limit 250 \
--base realestate_16x16_sine_cview_adaptive_epipolar \
--data-path ../dataset \
--ckpt 400000 --siamese_ckpt Siamese_folder/mask09_fulldata.pt \
--exp exp_bidirectional_epipolar --gpu 0 \
--type bi --mask_ratio 0.9
```

## Acknowledgement
### Code
- The implementation is based on [LoR](https://github.com/xrenaa/Look-Outside-Room)
- The implementation of SiamMAE is based on [SiamMAE](https://github.com/Jeremylin0904/SiamMAE_DeepLearning_final)
