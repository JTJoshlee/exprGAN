from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import cv2
import torch
from scipy.spatial import Delaunay
import os
neutral_path = r"E:\style_exprGAN\data\neutral_crop\S092_004_00000001.png"
smile_path = r"E:\style_exprGAN\data\smile_crop\S092_004_00000024.png"
neutral_landmark_path = r"E:\style_exprGAN\data\neutral_feature_points\landmark_S092_004_00000001.npy"
smile_landmark_path = r"E:\style_exprGAN\data\smile_feature_points\landmark_S092_004_00000024.npy"
appearance_path = r"E:\style_exprGAN\data\appearance"
os.makedirs(appearance_path, exist_ok=True)
class Warp():
    def __init__(self):
        self.x_warp = np.zeros((128,128))
    def apply_affine_transform(self, src, src_tri, dst_tri, size):
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    
    def morph_triangle(self, img, t1, t2, t, alpha):
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))

        t1_rect, t2_rect, t_rect = [], [], []
        for i in range(3):
            t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
            t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

        img1_rect = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r[2], r[3])
        warp_image = self.apply_affine_transform(img1_rect, t1_rect, t_rect, size)

        interpolated = cv2.addWeighted(img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]], 1-alpha, warp_image, alpha, 0)
        img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + interpolated * mask



    def create_smooth_mask(self, image, image_shape, landmarks, radius=30):
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        d = 1 # 像素鄰域直徑
        sigma_color = 10  # 顏色空間標準差
        sigma_space = 3
        radius = 5        # 濾波窗口的半徑
        epsilon = 0.1     # 正則化參數 (應該是平方後的值)
        epsilon = epsilon**2
        for point in landmarks:
            cv2.circle(mask, tuple(map(int, point)), radius, 1, -1)
            #blurred = cv2.bilateralFilter(mask, d, sigma_color, sigma_space)
            blurred = cv2.ximgproc.guidedFilter(guide=mask, src=mask, radius=radius, eps=epsilon)
            #blurred = cv2.GaussianBlur(mask, (radius*2+1, radius*2+1), 0)
        
        print(blurred.shape)
        return blurred
    
    def B_theta(self, x, theta):
        return np.where(x >= theta, 255, 0)

    def align_source_landmarks(self, source_image, source_landmarks, target_image, target_landmarks, thereshold=25, alpha=0.7, smooth_radius=10):
        img = source_image.copy()
        
        tri = Delaunay(source_landmarks)
        
        for i in range(len(tri.simplices)):
            t1 = source_landmarks[tri.simplices[i]]
            t2 = target_landmarks[tri.simplices[i]]
            t = (1-alpha) * t1 + alpha * t2
            self.morph_triangle(img, t1, t2, t, alpha)
        
        # 创建平滑mask
        smooth_mask = self.create_smooth_mask(img, source_image.shape, source_landmarks, smooth_radius)
        
        # 应用平滑mask
        result_mask = source_image * (1 - smooth_mask[:,:,np.newaxis]) + img * smooth_mask[:,:,np.newaxis]
        result_mask = np.uint8(result_mask)
        result = img
        result = np.uint8(result)
        
        target_image = np.uint8(target_image)
        abs_diff = np.abs(np.int16(result) - np.int16(target_image))
        abs_diff_mask = np.abs(np.int16(result_mask) - np.int16(target_image))
        np.set_printoptions(threshold=np.inf)
        #print(img)
        #print(result)
        #print(img_2)
        print(abs_diff)
        M_appeaance = self.B_theta(abs_diff_mask, thereshold)
        # plt.figure('target')
        # plt.imshow(target_image)
        plt.figure('abs')
        plt.imshow(abs_diff)
        plt.figure('abs_mask')
        plt.imshow(abs_diff_mask)
        plt.figure('result_mask')
        plt.imshow(result_mask)

        plt.figure('result')
        plt.imshow(result)

        
        
        return result, M_appeaance
    

    
        

if __name__ == '__main__':
    new_size = (128, 128)
    
    neutral_landmark = np.load(neutral_landmark_path)
    neutral_image = cv2.imread(neutral_path)
    neutral_image = cv2.resize(neutral_image, new_size, interpolation=cv2.INTER_LINEAR)
    
    #neutral_image = cv2.resize(neutral_image, new_size, interpolation=cv2.INTER_LINEAR)
    
    
    smile_landmark = np.load(smile_landmark_path)
    smile_image = cv2.imread(smile_path)
    smile_image = cv2.resize(smile_image, new_size, interpolation=cv2.INTER_LINEAR)
    
    print(f"smileimage:{smile_image.shape}")
    #smile_image = cv2.resize(smile_image, new_size, interpolation=cv2.INTER_LINEAR)
    
    source_image = neutral_image
    target_image = smile_image
    
    
    source_landmarks = neutral_landmark
    target_landmarks = smile_landmark
    warpping = Warp()
    result, minus = warpping.align_source_landmarks(source_image, source_landmarks, target_image, target_landmarks, thereshold=30)
    plt.figure('neutral_image')
    plt.imshow(neutral_image)
    plt.figure('final')
    plt.imshow(minus, cmap='gray')
    plt.show()
    
    