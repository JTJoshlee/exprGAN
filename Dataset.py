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


class Dataset(Dataset):
    def __init__(self, neutral_path, smile_path, transform=None):
        self.neutral_path = neutral_path
        self.smile_path = smile_path
        self.neutral_images = [f for f in os.listdir(neutral_path)]
        self.smile_images = [f for f in os.listdir(smile_path)]
        self.transform = transforms.Compose([
            #transforms.Resize((128, 128),antialias=True),
            transforms.RandomHorizontalFlip(),
            
            
        ])
        self.noise_std = 70
        self.neutral_indices = list(range(len(self.neutral_images)))
        self.neutral_indices *= 4
        self.smile_indices = list(range(len(self.smile_images)))
        self.smile_indices *= 4
        quarter_length = len(self.neutral_indices) //4
        neutral_second_half = self.neutral_indices[quarter_length:]
        smile_second_half = self.smile_indices[quarter_length:]
        random.shuffle(neutral_second_half)
        random.shuffle(smile_second_half)
        self.neutral_indices = self.neutral_indices[:quarter_length] + neutral_second_half
        self.smile_indices = self.smile_indices[:quarter_length] + smile_second_half
        
        # self.neutral_used = set()  # 用來追蹤已選擇的 neutral 圖片索引
        # self.smile_used = set()    # 用來追蹤已選擇的 smile 圖片索引   

    # def __iter__(self):
    #     half_length = len(self.neutral_indices) //2
    #     neutral_second_half = self.neutral_indices[half_length:]
    #     smile_second_half = self.smile_indices[half_length:]
    #     random.shuffle(neutral_second_half)
    #     random.shuffle(smile_second_half)
    #     self.neutral_indices = self.neutral_indices[:half_length] + neutral_second_half
    #     self.smile_indices = self.smile_indices[:half_length] + smile_second_half

    def __getitem__(self, index):
        # if len(self.neutral_used) == len(self.neutral_images):
        #     # 如果所有 neutral 圖片都已經選過，重置 neutral_used 並重新洗牌
        #     self.neutral_used = set()
        #     random.shuffle(self.neutral_indices)
        
        # # 隨機選擇一個未使用的 neutral 索引
        # neutral_idx = self.neutral_indices[index % len(self.neutral_images)]
        # self.neutral_used.add(neutral_idx)
        # same_id_or_not = random.choice([0 ,1])
        # if same_id_or_not == 0:
        #     smile_idx = neutral_idx
        # elif same_id_or_not == 1:     
            
        #     # 檢查是否還有未選擇的 smile 索引
        #     if len(self.smile_used) == len(self.smile_images):
        #         # 如果所有 smile 圖片都已經選過，重置 smile_used 並重新洗牌
        #         self.smile_used = set()
        #         random.shuffle(self.smile_indices)
            
        #     # 隨機選擇一個未使用的 smile 索引
        #     smile_idx = random.choice([i for i in range(len(self.smile_images)) if i not in self.smile_used])
            
        #     self.smile_used.add(smile_idx)
        neutral_idx = self.neutral_indices[index]
        smile_idx = self.smile_indices[index]
        #print(self.neutral_indices)
        #print(self.smile_indices)
        neutral_img_path = os.path.join(self.neutral_path, self.neutral_images[neutral_idx])
        neutral_name = os.path.basename(neutral_img_path)
        neutral_img = Image.open(neutral_img_path)
        neutral_img_np = np.array(neutral_img)
        noise = np.random.randn(*neutral_img_np.shape) * self.noise_std
        noisy_image = np.clip(neutral_img_np + noise, 0, 255).astype(np.uint8)
        noisy_image_tensor = torch.from_numpy(noisy_image).float() / 255.0      
        # noisy_image = np.clip(neutral_img_np, 0, 255).astype(np.uint8)
        # noisy_image_tensor = torch.from_numpy(noisy_image).float() / 255.0 
        neutral_tensor = self.transform(noisy_image_tensor).to("cuda")
        neutral_tensor = neutral_tensor.permute(2, 0, 1)
        same_id = torch.tensor(0, device="cuda", dtype=torch.float)
        # print("image tensor", neutral_img.size)
        # print("noise tensor", noisy_image_tensor.shape)  
        # print("neutral tensor", neutral_tensor.shape)
        
        smile_img_path = os.path.join(self.smile_path, self.smile_images[smile_idx])
        smile_name = os.path.basename(smile_img_path)
        smile_img = Image.open(smile_img_path)
        smile_img_np = np.array(smile_img)
        noise = np.random.randn(*smile_img_np.shape) * self.noise_std
        smile_image = np.clip(smile_img_np + noise, 0, 255).astype(np.uint8)
        smile_image_tensor = torch.from_numpy(smile_image).float() / 255.0
        # smile_image = np.clip(smile_img_np, 0, 255).astype(np.uint8)
        # smile_image_tensor = torch.from_numpy(smile_image).float() / 255.0   
        
        smile_tensor = self.transform(smile_image_tensor).to("cuda")
        smile_tensor = smile_tensor.permute(2, 0, 1)         
        #smile_tensor = self.transform(smile_img).to("cuda")
        
        if neutral_idx == smile_idx:
            same_id = torch.tensor(1, device="cuda", dtype=torch.float).unsqueeze(0)   
        else:
            same_id = torch.tensor(0, device="cuda", dtype=torch.float).unsqueeze(0)

       

           

        return {
            'neutral': neutral_tensor,
            'smile': smile_tensor,
            'neutral_name': neutral_name,
            'smile_name': smile_name,
            'same_id': same_id
            
        }
    
    def __len__(self):
        
        return 4*len(self.neutral_images)



    