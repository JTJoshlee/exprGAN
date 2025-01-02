import os
from pathlib import Path
import shutil
from PIL import Image
import cv2
import numpy as np
#data_path = r"E:\style_exprGAN\CK+\Emotion"
image_path = r"E:\style_exprGAN\CK+\cohn-kanade-images"
neutral_path = r"E:\style_exprGAN\data\neutral"
smile_path = r"E:\style_exprGAN\data\smile"
# neutral_crop_path = r"E:\style_exprGAN\data\neutral_crop"
# smile_crop_path = r"E:\style_exprGAN\data\smile_crop"

class Args(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

args = {
    'image_size' : (128,128),
    'neutral_crop_path' : r".\data\neutral_crop_128",
    'smile_crop_path' : r".\data\smile_crop_128",
    'neutral_path' : r"E:\style_exprGAN\data\neutral",
    'smile_path' : r"E:\style_exprGAN\data\smile"
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
    crop_box = (125, 25, 550, 442)
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

# 主程式執行
# data_path = r"E:\style_exprGAN\CK+\Emotion"  # 設定你的資料夾路徑
# traverse_data_folder(data_path)
if __name__ == '__main__':
    crop_image(neutral_path, args.neutral_crop_path)
    crop_image(smile_path, args.smile_crop_path)