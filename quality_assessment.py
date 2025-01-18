import pytorch_ssim
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

import cv2
transform = transforms.ToTensor()
def calculate_psnr(img1, img2):
    image1_tensor = transform(img1).unsqueeze(0) #添加batch維度
    image2_tensor = transform(img2).unsqueeze(0) #添加batch維度
      
    mse = torch.mean((image1_tensor - image2_tensor) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    print("psnr value", 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
)
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))



def calaulate_ssim(img1, img2):
    image1_tensor = transform(img1).unsqueeze(0) #添加batch維度
    image2_tensor = transform(img2).unsqueeze(0) #添加batch維度
   
   
    ssim_val = pytorch_ssim.ssim(image1_tensor, image2_tensor)
    print("ssim val", ssim_val)
    return ssim_val





if __name__ == "__main__":
    image1_path = r"E:\style_exprGAN\data\neutral_crop_align_128\S010_006_00000001.png"
    image2_path = r"E:\style_exprGAN\data\smile_crop_align_128\S010_006_00000015.png"
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    

    
    calculate_psnr(image1, image2)
    calaulate_ssim(image1, image2)