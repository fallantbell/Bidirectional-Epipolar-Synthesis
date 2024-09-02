# 初始化變數來儲存總和和計數
total_percsim = 0
total_ssim = 0
total_psnr = 0
count = 0

data_path = "experiments/realestate/exp_forward_epipolar_full_10kdata_error/evaluate_frame_21_video_250_ckpt_100000/"

for eval in ['0','1','2','3']:
    eval_path = f"{data_path}/eval_{eval}.txt"
    count = 0
    # 讀取文件
    with open(eval_path, 'r') as file:
        for line in file:
            # 分割每一行的內容來提取數值
            parts = line.split(',')
            percsim = float(parts[1].split(':')[1].strip())
            ssim = float(parts[2].split(':')[1].strip())
            psnr = float(parts[3].split(':')[1].strip())
            
            # 累加數值
            total_percsim += percsim
            total_ssim += ssim
            total_psnr += psnr
            
            # 增加計數
            count += 1
            if count == 250:
                break

# 計算平均值
avg_percsim = total_percsim / 1000
avg_ssim = total_ssim / 1000
avg_psnr = total_psnr / 1000

# 輸出平均值
print(f'Average percsim: {avg_percsim:.4f}')
print(f'Average ssim: {avg_ssim:.4f}')
print(f'Average psnr: {avg_psnr:.4f}')
