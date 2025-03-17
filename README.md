## Installation
- Clone the repository:
```
git clone https://github.com/xrenaa/Look-Outside-Room
cd Look-Outside-Room
python scripts/download_vqmodels.py
```
- Dependencies:  
```
conda env create -f environment.yaml
conda activate lookout
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
--dataset realestate --name exp_bidirectional_epipolar \
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
--ckpt 400000 \
--exp exp_bidirectional_epipolar --gpu 0 \
--type bi --mask_ratio 0.9
```
