
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

class ExpressionDataset(Dataset):
    def __init__(self, neutral_path, smile_path):
        self.neutral_path = neutral_path
        self.smile_path = smile_path
        self.neutral_images = [f for f in os.listdir(neutral_path)]
        self.smile_images = [f for f in os.listdir(smile_path)]
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip()
            
        ])
        self.noise_std = 70
        self.label = [1,0]
    def __getitem__(self, index):
       

# 隨機選擇是從哪個資料夾中選擇圖片
        selected_folder = random.choice([self.neutral_images, self.smile_images])
        print(f"select folder {selected_folder}")
# 從選擇的資料夾中隨機選擇一張圖片
        selected_image_path = random.choice(selected_folder)
        image_name = os.path.basename(selected_image_path)
        img = Image.open(selected_image_path)
        img_np = np.array(img)
        img_tensor = self.transform(img_np).to("cuda")
        if selected_folder == self.neutral_images:
            self.label = [1,0]
        else:
            self.label = [0,1]

        return{
            "image":img_tensor,
            "label":self.label,
            "name":image_name

        }
    def __len__(self):
        return len(self.neutral_images) + len(self.smile_images)

    