import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import cv2
import torch
from scipy.spatial import Delaunay
import glob
class SEM():
    def __init__(self):
        self.threshold = 150
        self.resize = (128,128)
    def dataProcess(self, image_path):
        data = cv2.imread(image_path)
        data = cv2.resize(data, self.resize, interpolation=cv2.INTER_LINEAR)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        if data.ndim < 3:
            data = cv2.merge([data, data, data])
        else:
            print("data dimension error")
        data = np.array(data, dtype=np.float32)
        return data
    
    def B_theta(self, x):
        return np.where(x >= self.threshold, 0, 255)
    

    def semantic_or(self, x_gradCAM_path, y_gradCAM_path):
        x_gradCAM = self.dataProcess(x_gradCAM_path)
        y_gradCAM = self.dataProcess(y_gradCAM_path)
        x_grad = self.B_theta(x_gradCAM)
        y_grad = self.B_theta(y_gradCAM)
        semantic_binary = np.bitwise_or(x_grad, y_grad)
        plt.imshow(semantic_binary)
        plt.show()
        
        return semantic_binary

class Appearance():
    def __init__(self):
        self.resize = (128,128)
        self.x_warp = np.zeros(self.resize)
    
    def dataProcess(self, image_path):
        data = cv2.imread(image_path)
        data = cv2.resize(data, self.resize, interpolation=cv2.INTER_LINEAR)
        return data
        
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
        
        
        return blurred
    
    def B_theta(self, x, theta):
        return np.where(x >= theta, 255, 0)



    def align_source_landmarks(self, neutral_path, source_landmarks, smile_path, target_landmarks, thereshold=25, alpha=0.7, smooth_radius=10):
        source_image = self.dataProcess(neutral_path)
        target_image = self.dataProcess(smile_path)

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
        M_appeaance = self.B_theta(abs_diff_mask, thereshold)
        
        
        plt.figure('abs_mask')
        plt.imshow(abs_diff_mask)
        plt.figure('result_mask')
        plt.imshow(result_mask)

        

        
        
        return result, M_appeaance
    
class style_map():
    def __init__(self):
        self.result_path = r"E:\style_exprGAN\ORL_data\choosed\style_map"
        self.resize = (128,128)
    


    def and_two_maps(self, semantic_map, appearance_map):
        return np.bitwise_and(semantic_map, appearance_map)
    
    def save_smileMaps(self, semantic_map, appearance_map):
        files = os.listdir(self.result_path)
        maps = self.and_two_maps(semantic_map, appearance_map)
        plt.figure('and map')
        plt.imshow(maps)
        
        #np.save(os.path.join(self.result_path,'stylemap', {len(files)+1}), maps)
        

if __name__ == "__main__":
    new_size = (128,128)
    neutral_gradCAM = r"E:\style_exprGAN\ORL_data\choosed\neutral_cam\CAM_neutral_5.png.jpg"
    smile_gradCAM = r"E:\style_exprGAN\ORL_data\choosed\smile_cam\CAM_smile_5.png.jpg"
    Semantic = SEM()
    Map = style_map()
    appearance = Appearance()
    neutral_path = r"E:\style_exprGAN\ORL_data\choosed\neutral\2.png"
    smile_path = r"E:\style_exprGAN\ORL_data\choosed\smile\2.png"
    neutral_landmark_path = r"E:\style_exprGAN\ORL_data\choosed\neutral_feature_points\landmark_2.npy"
    smile_landmark_path = r"E:\style_exprGAN\ORL_data\choosed\smile_feature_points\landmark_2.npy"
    
    
    semantic_map = Semantic.semantic_or(neutral_gradCAM, smile_gradCAM)

    neutral_landmark = np.load(neutral_landmark_path)
    smile_landmark = np.load(smile_landmark_path)
    
    result, appearance_map = appearance.align_source_landmarks(neutral_path, neutral_landmark, smile_path, smile_landmark, thereshold=30)
    
    Map.save_smileMaps(appearance_map, semantic_map)
    
    
    plt.figure('appearance map')
    plt.imshow(appearance_map, cmap='gray')
    plt.show()
    