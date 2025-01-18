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
from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import KFold
from Dataset import Dataset as Ds
class Args(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

args = {
        'device' : 'cuda',
        'batch_size' : int(8),        
        "neutral_output_dir" : r"./data/neutral_test_layer-3",
        "smile_output_dir" : r"./data/smile_test_layer-3",
        "neutral_graycam_dir" : r"./data/neutral/graycam",
        "smile_graycam_dir" : r"./data/smile/graycam"
        
    }
args = Args(args)



def Grad_CAM(model, original_tensor):
    
    target_layer = [model.encoder.layers[-3]]
    batch_size = args.batch_size
    
    cam = GradCAM(model=model, target_layers=target_layer)
    targets_neutral = [ClassifierOutputTarget(0)]
    targets_smile = [ClassifierOutputTarget(1)]
    neutral_cam_images = []
    complementary_neutral_img = []
    smile_cam_images = []
    complementary_smile_img = []
    neutral_output_dir = args.neutral_output_dir
    os.makedirs(neutral_output_dir, exist_ok=True)
    smile_output_dir = args.smile_output_dir
    os.makedirs(smile_output_dir, exist_ok=True)
    neutral_graycam_dir = args.neutral_graycam_dir
    os.makedirs(neutral_graycam_dir, exist_ok=True)
    smile_graycam_dir = args.smile_graycam_dir
    os.makedirs(smile_graycam_dir, exist_ok=True)





    for i in range(batch_size):
        
        
        input_tensor = original_tensor['neutral'][i].unsqueeze(0)
          
        #print("input_tensor", input_tensor.shape)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets_neutral)
           
        #print("gray cam shape", grayscale_cam.shape)
        grayscale_cam = grayscale_cam[0, :]  # 提取第 i 个图像的 Grad-CAM
         
        #print("grayscale_cam", grayscale_cam.shape)
        img_np = original_tensor['neutral'][i].permute(1, 2, 0)
        img_np = img_np.cpu().detach().numpy()  # 提取第 i 个图像并转换为 numpy
        #print("img_np", img_np.shape)

        #img_np = img_np.squeeze(0)                
        #img_np = np.transpose(img_np, (1, 2, 0))  # 转换为 (H, W, C)
        
        
    # 将 Grad-CAM 映射到原始图像
        
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        neutral_cam_images.append(grayscale_cam)
        
    # 保存生成的 CAM 图像
        neutral_name = f"CAM_neutral_{original_tensor['neutral_name'][i]}.jpg"
        neutral_path = os.path.join(neutral_output_dir, neutral_name)
        cv2.imwrite(neutral_path, cam_image)  # 使用不同的文件名
        graycam_normalize_image = graycam_normalization(grayscale_cam)
        neutral_graycam_name = f"graycam_neutral_{original_tensor['neutral_name'][i]}.jpg"
        neutral_graycam_path = os.path.join(neutral_graycam_dir, neutral_graycam_name)
        cv2.imwrite(neutral_graycam_path,  graycam_normalize_image)
        
        
        complementary_x = complementary_img(grayscale_cam, original_tensor['neutral'][i])
        complementary_neutral_img.append(complementary_x)
        

    #return cam_images
    neutral_cam_images = np.array(neutral_cam_images)
    neutral_cam_images = torch.tensor(neutral_cam_images).to('cuda')
    complementary_neutral_img = np.array(complementary_neutral_img)
    complementary_neutral_img = torch.tensor(complementary_neutral_img).to('cuda')

    for j in range(batch_size):
        input_tensor_smile = original_tensor['smile'][j].unsqueeze(0)
        
        grayscale_cam_smile = cam(input_tensor=input_tensor_smile, targets=targets_smile)
        grayscale_cam_smile = grayscale_cam_smile[0,:]

        
        img_np_smile = original_tensor['smile'][j].permute(1, 2, 0)
        img_np_smile = img_np_smile.cpu().detach().numpy()
        #img_np_smile = np.transpose(img_np_smile, (1,2,0))
        
        cam_image_smile = show_cam_on_image(img_np_smile, grayscale_cam_smile, use_rgb=True)
        smile_name = f"CAM_smile_{original_tensor['smile_name'][j]}.jpg"
        smile_path = os.path.join(smile_output_dir, smile_name)
        cv2.imwrite(smile_path, cam_image_smile)  # 使用不同的文件名
        #cv2.imwrite(f"E:/style_exprGAN/data/layer-2_smile/CAM_smile_{original_tensor['smile_name'][j]}.jpg", cam_image_smile)
        graycam_normalize_smile_image = graycam_normalization(grayscale_cam_smile)
        smile_graycam_name = f"graycam_smile_{original_tensor['smile_name'][i]}.jpg"
        smile_graycam_path = os.path.join(smile_graycam_dir, smile_graycam_name)
        cv2.imwrite(smile_graycam_path, graycam_normalize_smile_image)
        
        
        
        
        
        smile_cam_images.append(grayscale_cam_smile)

        complementary_y = complementary_img(grayscale_cam_smile, original_tensor['smile'][j])
        complementary_smile_img.append(complementary_y)

    smile_cam_images = np.array(smile_cam_images)
    smile_cam_images = torch.tensor(smile_cam_images).to('cuda')
    complementary_smile_img = np.array(complementary_smile_img)
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


def graycam_normalization(grayscale_cam):
    grayscale_cam_normalized = (grayscale_cam - np.min(grayscale_cam)) / (np.max(grayscale_cam) - np.min(grayscale_cam) + 1e-8)
    grayscale_cam_uint8 = (grayscale_cam_normalized * 255).astype(np.uint8)
    grayscale_cam_inverted = 255 - grayscale_cam_uint8
    return grayscale_cam_inverted

if __name__ == "__main__":
    batch_size = int(8)

    neutral_path = r".\data\neutral_crop_align_128"
    smile_path = r".\data\smile_crop_align_128"
    
    
    
    state_dict = torch.load(r'E:\style_exprGAN\model\expression_classifier_epoch2500.pth')
    model_Enc = EASNNetwork().to("cuda")
    model_Enc.load_state_dict(state_dict)
    model_Enc.eval()
    train_dataset = Ds(neutral_path, smile_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in train_loader:
        com_neutral, com_smile = Grad_CAM(model_Enc, batch)
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

