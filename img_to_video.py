import cv2
import os
import re

def custom_sort(file_name):
    # 提取文件名中的數字部分
    num_part = re.findall(r'\d+', file_name)
    if num_part:
        return int(num_part[0])  # 將數字部分轉換為整數進行排序
    else:
        return file_name

def write_video(type,video_num):

    true_type = ""
    if type == 'gt':
        true_type = "gt"
        type = "exp_forward_epipolar_error"

    # 資料夾路徑
    if type == 'exp_fixed_bi_epipolar_maskcam_sepsoft-4_4gpu':
        folder_path = f'experiments/realestate/{type}/evaluate_frame_6_video_250_ckpt_200000/{video_num}'
    else:
        folder_path = f'experiments/realestate/{type}/evaluate_frame_21_video_1000_ckpt_100000/{video_num}'

    # 影片儲存路徑及檔名
    os.makedirs(f'saved_video/{video_num}',exist_ok=True)
    output_video = f'saved_video/{video_num}/{type}.mp4'

    # 取得資料夾內所有圖片的檔案名稱
    image_files = sorted([f for f in os.listdir(folder_path) if f[:2]!='gt'], key=custom_sort)

    if true_type=="gt":
        output_video = f'saved_video/{video_num}/GT.mp4'
        image_files = sorted([f for f in os.listdir(folder_path) if f[:2]=='gt'], key=custom_sort)

    # print(image_files)
    # print(image_files)
    # 讀取第一張圖片以獲取寬度和高度
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, _ = first_image.shape

    # 設定影片編碼器、FPS、影片大小
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5 # 5
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 逐一讀取圖片並寫入影片
    cnt = 0
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
        cnt +=1
        if cnt >=20:
            break 

    # 釋放影片寫入器
    video_writer.release()

    print('影片已儲存至', output_video)

if __name__ == '__main__':

    model_type = [
        'exp_fixed_bi_epipolar_maskcam_sepsoft-4_2gpu_error',
                  ]
    video_num = "005"

    for type in model_type:
        write_video(type,video_num)