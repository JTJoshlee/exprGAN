import os
from pathlib import Path
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
#data_path = r"E:\style_exprGAN\CK+\Emotion"
image_path = r"E:\style_exprGAN\CK+\cohn-kanade-images"
# neutral_path = r"E:\style_exprGAN\data\neutral"
# smile_path = r"E:\style_exprGAN\data\smile"
# neutral_crop_path = r"E:\style_exprGAN\data\neutral_crop"
# smile_crop_path = r"E:\style_exprGAN\data\smile_crop"

class Args(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

args = {
    'image_size' : (128,128),
    'neutral_crop_path' : r".\data\neutral_crop_64",
    'smile_crop_path' : r".\data\smile_crop_64",
    'neutral_path' : r"E:\style_exprGAN\data\neutral_crop_128",
    'smile_path' : r"E:\style_exprGAN\data\smile_crop_128",
    'neutral_landmark_path' : r"E:\style_exprGAN\data\neutral_feature_points",
    'smile_landmark_path' : r"E:\style_exprGAN\data\smile_feature_points",
    'neutral_align_path' : r"E:\style_exprGAN\data\neutral_align_128",
    "smile_align_path" : r"E:\style_exprGAN\data\smile_align_128",
    "neutral_crop_align_path" : r"E:\style_exprGAN\data\neutral_crop_align_128",
    "smile_crop_align_path" : r"E:\style_exprGAN\data\smile_crop_align_128",
    "neutral_equalize_brightness" : r"E:\style_exprGAN\data\neutral_equalize_brightness",
    "smile_equalize_brightness" : r"E:\style_exprGAN\data\smile_equalize_brigthness"
}
args = Args(args)
os.makedirs(args.neutral_crop_path, exist_ok=True)
os.makedirs(args.smile_crop_path, exist_ok=True)
def read_txt_file(file_path):
    """讀取 txt 文件的內容，去除空白字符並回傳內容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    return content

def process_emotion_file(file_path):
    """判斷文件內容是否為指定值並進行輸出"""
    content = read_txt_file(file_path)
    if content == "5.0000000e+00":
        file_path = Path(file_path)
        target_path = Path(image_path) / file_path.parts[-3] / file_path.parts[-2]
        file_name = os.listdir(target_path)
        target_neutral_image = os.path.join(target_path, file_name[0])
        target_smile_iamge = os.path.join(target_path, file_name[-1])

        shutil.copy(target_neutral_image, neutral_path)
        shutil.copy(target_smile_iamge, smile_path)
        print("happy")
        print(f"target path {target_path}")
        print(f"Content of {file_path}:\n{content}\n")

def process_subject_emotion_folder(emotionalLabel_path):
    """遍歷情感標籤文件夾中的所有 txt 文件，並處理符合條件的文件"""
    for emotionLabel in os.listdir(emotionalLabel_path):
        if emotionLabel.endswith('.txt'):
            file_path = os.path.join(emotionalLabel_path, emotionLabel)
            process_emotion_file(file_path)

def traverse_data_folder(data_path):
    """遍歷資料夾中的每個受試者資料夾，處理其內的情感文件"""
    for subject_name in os.listdir(data_path):
        subject_facial = os.path.join(data_path, subject_name)
        if not os.path.isdir(subject_facial):
            continue  # 跳過非資料夾的項目
        for subject_emotion in os.listdir(subject_facial):
            emotionalLabel_path = os.path.join(subject_facial, subject_emotion)
            if os.path.isdir(emotionalLabel_path):
                process_subject_emotion_folder(emotionalLabel_path)




def crop_image(input_folder, output_folder):
    crop_box = (30, 30, 100, 100)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            cropped_img = img.crop(crop_box)
            cropped_img = cropped_img.convert('L')
            
            cropped_img = cropped_img.resize(args.image_size)
            print("cropped_img",cropped_img.size)
                        
            np_img = np.array(cropped_img)
        
            # Stack the grayscale image into 3 identical channels
            np_img_3ch = np.stack([np_img, np_img, np_img], axis=-1)  # Shape: (128, 128, 3)
            print("np", np_img_3ch.size)
            # Convert back to PIL Image
            cropped_img_3ch = Image.fromarray(np_img_3ch.astype('uint8'), mode='RGB')  # Specify 'RGB' mode explicitly
            print("cropped_img_3ch",cropped_img_3ch.size)
            # Save the 3-channel image
            output_path = os.path.join(output_folder, filename)
            cropped_img_3ch.save(output_path)

def align_image(input_folder, output_folder, landmark_path):    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        image_name, ext = os.path.splitext(filename)
        image_landmark_path = os.path.join(landmark_path, 'landmark_'+image_name+'.npy')
        img = cv2.imread(img_path) 
        landmarks = np.load(image_landmark_path)
        
        

        # 計算所有點的中心
        center = np.mean(landmarks, axis=0).astype(np.int32)
        
        output_size = (128, 128)
        img_center_x = output_size[0] // 2  # 320 (如果輸出大小是 640x480)
        img_center_y = output_size[1] // 2  # 240

        # 計算需要的平移量：圖片中心 - 當前位置
        dx = img_center_x - center[0]  # 正值表示向右移，負值表示向左移
        dy = img_center_y - center[1]  # 正值表示向下移，負值表示向上移

        # 建立平移矩陣
        M = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])
        aligned = cv2.warpAffine(img, M, output_size)
        aligned = Image.fromarray(aligned.astype('uint8'), mode='RGB')
            # Save the 3-channel image
        output_path = os.path.join(output_folder, filename)
        aligned.save(output_path)


def compute_average_brightness(image_folder):
    """計算資料集的平均亮度"""
    brightness_values = []
    
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 轉為灰階

        if img is None:
            continue  # 跳過無法讀取的圖片

        brightness = np.mean(img)  # 計算亮度
        brightness_values.append(brightness)

    return np.mean(brightness_values)


def equalized_birghtness(input_folder, output_folder, target_brightness=None):
    inv_gamma = 1.0 / 1.8
    
    if target_brightness is None:
        target_brightness = compute_average_brightness(input_folder)
    for filename in os.listdir(input_folder):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0]

        # 計算當前圖片的亮度
        current_brightness = np.mean(y_channel)

        # 計算亮度調整係數
        scale = target_brightness / (current_brightness + 1e-6)  # 避免除以 0

        # 限制調整範圍，避免過度增強或降低亮度
        scale = np.clip(scale, 0.5, 2.0)

        # 調整亮度
        y_channel = np.clip(y_channel * scale, 0, 255).astype(np.uint8)

        # 合併回 YUV 並轉回 BGR
        img_yuv[:, :, 0] = y_channel
        normalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # 保存圖片
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, normalized_img)
    
# 主程式執行
# data_path = r"E:\style_exprGAN\CK+\Emotion"  # 設定你的資料夾路徑
# traverse_data_folder(data_path)
if __name__ == '__main__':
    #align_image(args.neutral_path, args.neutral_align_path, args.neutral_landmark_path)
    #align_image(args.smile_path, args.smile_align_path, args.smile_landmark_path)
    #crop_image(args.neutral_align_path, args.neutral_crop_align_path)
    #crop_image(args.smile_align_path, args.smile_crop_align_path)
    equalized_birghtness(args.neutral_crop_align_path, args.neutral_equalize_brightness)
    equalized_birghtness(args.smile_crop_align_path, args.smile_equalize_brightness)