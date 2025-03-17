# 讀取檔案
with open('exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error.txt', 'r') as file:
    lines = file.readlines()

# 遍歷每一行，並過濾出 PSNR 大於 17 的編號
for line in lines:
    # 分析每一行的內容
    parts = line.split(',')
    psnr_value = float(parts[-1].split(':')[1].strip())  # 取得PSNR的值並轉為浮點數
    if psnr_value > 17:
        # 取得編號並顯示
        id_number = parts[0].split('-')[0]
        print(f"ID: {id_number}, PSNR: {psnr_value}")
