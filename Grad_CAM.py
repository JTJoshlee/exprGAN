from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
import torchvision
import torch
from Enc import EASNNetwork
import matplotlib.pyplot as plt
import os


def Grad_CAM(model, original_tensor):
    target_layer = [model.encoder.layers[-2]]
    model.eval()
    batch_size = int(4)
    
    cam = GradCAM(model=model, target_layers=target_layer)
    targets_neutral = [ClassifierOutputTarget(0)]
    targets_smile = [ClassifierOutputTarget(1)]
    neutral_cam_images = []
    complementary_neutral_img = []
    smile_cam_images = []
    complementary_smile_img = []
    neutral_output_dir = "E:/style_exprGAN/data/with_ID_attention_0.5layer-2_neutral"
    os.makedirs(neutral_output_dir, exist_ok=True)
    smile_output_dir = "E:/style_exprGAN/data/with_ID_attention_0.5_layer-2_smile"
    os.makedirs(smile_output_dir, exist_ok=True)
    for i in range(batch_size):
        
        input_tensor = original_tensor['neutral'][i].unsqueeze(0)   
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets_neutral)   
  
        grayscale_cam = grayscale_cam[0, :]  # 提取第 i 个图像的 Grad-CAM
        
        img_np = original_tensor['neutral'][i].cpu().detach().numpy()  # 提取第 i 个图像并转换为 numpy
        
        img_np = np.transpose(img_np, (1, 2, 0))  # 转换为 (H, W, C)
    
    # 将 Grad-CAM 映射到原始图像
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        neutral_cam_images.append(grayscale_cam)
        
    # 保存生成的 CAM 图像
        neutral_name = f"CAM_neutral_{original_tensor['neutral_name'][i]}.jpg"
        neutral_path = os.path.join(neutral_output_dir, neutral_name)
        cv2.imwrite(neutral_path, cam_image)  # 使用不同的文件名
        complementary_x = complementary_img(grayscale_cam, original_tensor['neutral'][i])
        complementary_neutral_img.append(complementary_x)
        

    #return cam_images
    neutral_cam_images = torch.tensor(neutral_cam_images).to('cuda')
    complementary_neutral_img = torch.tensor(complementary_neutral_img).to('cuda')

    for j in range(batch_size):
        input_tensor_smile = original_tensor['smile'][j].unsqueeze(0)
        
        grayscale_cam_smile = cam(input_tensor=input_tensor_smile, targets=targets_smile)
        grayscale_cam_smile = grayscale_cam_smile[0,:]
        img_np_smile = original_tensor['smile'][j].cpu().detach().numpy()
        img_np_smile = np.transpose(img_np_smile, (1,2,0))
        
        cam_image_smile = show_cam_on_image(img_np_smile, grayscale_cam_smile, use_rgb=True)
        smile_name = f"CAM_smile_{original_tensor['smile_name'][j]}.jpg"
        smile_path = os.path.join(smile_output_dir, smile_name)
        cv2.imwrite(smile_path, cam_image_smile)  # 使用不同的文件名
        #cv2.imwrite(f"E:/style_exprGAN/data/layer-2_smile/CAM_smile_{original_tensor['smile_name'][j]}.jpg", cam_image_smile)
        smile_cam_images.append(grayscale_cam_smile)

        complementary_y = complementary_img(grayscale_cam_smile, original_tensor['smile'][j])
        complementary_smile_img.append(complementary_y)

    smile_cam_images = torch.tensor(smile_cam_images).to('cuda')
    complementary_smile_img = torch.tensor(complementary_smile_img).to('cuda')

    complementary_neutral_output, _ = model(complementary_neutral_img)
    complementary_smile_output, _ = model(complementary_smile_img)


    return complementary_neutral_output, complementary_smile_output

def complementary_img(Mcam_img, input_img):  

    Mcam_img_tensor = torch.tensor(Mcam_img).to('cuda')
    input_img_tensor = input_img.clone().detach()

    alpha = 10.0
    beta = 0.5
    
    delta = torch.sigmoid(alpha * (Mcam_img_tensor - beta))
    if torch.isnan(delta).any():
        print("NaN detected in delta")

    complement_mask = 1 - delta
    if torch.isnan(complement_mask).any():
        print("NaN detected in complement_mask")

    mask_image = input_img_tensor * complement_mask
    if torch.isnan(mask_image).any():
        print("NaN detected in x_bar_j")
    
    mask_image = mask_image.detach().cpu().numpy()
    
    return  mask_image

# 顯示互補圖像
   


# def Grad_CAM(model, x_tensor):
#     target_layer = [model.encoder.layers[-1]]
#     model.eval()
#     x_tensor = x_tensor[0].unsqueeze(0)
#     cam = GradCAM(model=model, target_layers=target_layer)
#     targets = [ClassifierOutputTarget(0)]
#     grayscale_cam = cam(input_tensor=x_tensor, targets=targets)

    

#     cam_images = []
#     for i in range(grayscale_cam.shape[0]):  # 遍历 batch 中的每个图像
#         grayscale_cam = grayscale_cam[i, :]  # 提取第 i 个图像的 Grad-CAM
#         print(f"grayscale_cam {grayscale_cam.shape}")
        
#         img_np = x_tensor[i].cpu().detach().numpy()  # 提取第 i 个图像并转换为 numpy
#         img_np = np.transpose(img_np, (1, 2, 0))  # 转换为 (H, W, C)
#         print(f"img_mp {img_np.shape}")
#         # 将 Grad-CAM 映射到原始图像
#         cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
#         # 保存生成的 CAM 图像
#         cv2.imwrite(f'E:/style_exprGAN/ORL_data/choosed/CAM_test_{i}.jpg', cam_image)  # 使用不同的文件名

#         cam_images.append(cam_image)  # 将生成的 CAM 图像添加到列表

#     return cam_images

