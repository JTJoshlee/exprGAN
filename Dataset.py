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
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip()
            
        ])
        self.noise_std = 70
            
        
    def __getitem__(self, index):
        neutral_idx = random.randint(0, len(self.neutral_images)-1)
        smile_idx = random.randint(0, len(self.smile_images)-1)
        
        neutral_img_path = os.path.join(self.neutral_path, self.neutral_images[neutral_idx])
        neutral_name = os.path.basename(neutral_img_path)
        neutral_img = Image.open(neutral_img_path).convert('L')
        neutral_img_np = np.array(neutral_img)
        noise = np.random.randn(*neutral_img_np.shape) * self.noise_std
        noisy_image = np.clip(neutral_img_np + noise, 0, 255).astype(np.uint8)
        noisy_image_tensor = torch.from_numpy(noisy_image).float() / 255.0      
        noisy_image_tensor = noisy_image_tensor.unsqueeze(0)
        neutral_tensor = self.transform(noisy_image_tensor).to("cuda")
        same_id = torch.tensor(0, device="cuda", dtype=torch.float)  
        
        
        smile_img_path = os.path.join(self.smile_path, self.smile_images[smile_idx])
        smile_name = os.path.basename(smile_img_path)
        smile_img = Image.open(smile_img_path).convert('L')
        smile_img_np = np.array(smile_img)
        noise = np.random.randn(*smile_img_np.shape) * self.noise_std
        smile_image = np.clip(smile_img_np + noise, 0, 255).astype(np.uint8)
        smile_image_tensor = torch.from_numpy(smile_image).float() / 255.0      
        smile_image_tensor = smile_image_tensor.unsqueeze(0)
        smile_tensor = self.transform(smile_image_tensor).to("cuda")        
        #smile_tensor = self.transform(smile_img).to("cuda")
        
        if neutral_idx == smile_idx:
            same_id = torch.tensor(1, device="cuda", dtype=torch.float)   
        else:
            same_id = torch.tensor(0, device="cuda", dtype=torch.float)

       

           

        return {
            'neutral': neutral_tensor,
            'smile': smile_tensor,
            'neutral_name': neutral_name,
            'smile_name': smile_name,
            'same_id': same_id
            
        }
    
    def __len__(self):
        
        return len(self.neutral_images) * len(self.smile_images)



    