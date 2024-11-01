from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import cv2
import torch
from scipy.spatial import Delaunay
from torch_tps import ThinPlateSpline
neutral_path = r"E:\style_exprGAN\ORL_data\choosed\neutral\1.png"
smile_path = r"E:\style_exprGAN\ORL_data\choosed\smile\1.png"
neutral_landmark_path = r"E:\style_exprGAN\ORL_data\choosed\neutral_feature_points\landmark_1.npy"
smile_landmark_path = r"E:\style_exprGAN\ORL_data\choosed\smile_feature_points\landmark_1.npy"
transform = transforms.Compose([
            transforms.Resize((128, 128))           
        ])

neutral_landmark = np.load(neutral_landmark_path)
neutral_image = cv2.imread(neutral_path)
new_size = (128, 128)
neutral_image = cv2.resize(neutral_image, new_size, interpolation=cv2.INTER_LINEAR)
smile_landmark = np.load(smile_landmark_path)
smile_image_ori = cv2.imread(smile_path)
#smile_image = cv2.resize(smile_image_ori, new_size, interpolation=cv2.INTER_LINEAR)

smile_image_ori_pil = Image.fromarray(smile_image_ori)
smile_image = transform(smile_image_ori_pil)

plt.imshow(smile_image)
plt.show()
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

def morph_triangle(img, t1, t2, t, alpha):
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
    warp_image = apply_affine_transform(img1_rect, t1_rect, t_rect, size)

    interpolated = cv2.addWeighted(img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]], 1-alpha, warp_image, alpha, 0)
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + interpolated * mask

def create_smooth_mask(image, image_shape, landmarks, radius=30):
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

def align_source_landmarks(source_image, source_landmarks, target_landmarks, alpha=0.7, smooth_radius=10):
    img = source_image.copy()
    
    tri = Delaunay(source_landmarks)
    
    for i in range(len(tri.simplices)):
        t1 = source_landmarks[tri.simplices[i]]
        t2 = target_landmarks[tri.simplices[i]]
        t = (1-alpha) * t1 + alpha * t2
        morph_triangle(img, t1, t2, t, alpha)
    
    # 创建平滑mask
    smooth_mask = create_smooth_mask(img, source_image.shape, source_landmarks, smooth_radius)
    
    # 应用平滑mask
    result_mask = source_image * (1 - smooth_mask[:,:,np.newaxis]) + img * smooth_mask[:,:,np.newaxis]
    result = img
    result_mask = np.uint8(result_mask)
    plt.figure('result_mask')
    plt.imshow(result_mask)
    return np.uint8(result)

source_image = neutral_image
target_image = smile_image

source_landmarks = neutral_landmark
landmarks = smile_landmark

result = align_source_landmarks(source_image, source_landmarks, landmarks)



neutral_image = Image.open(neutral_path).convert('L')
neutral_image = transform(neutral_image)

jaw = landmarks[0:17]

#左眉 (Left eyebrow: 5 points)  18 ~ 22
left_eyebrow = landmarks[17:22]

#右眉 (Right eyebrow: 5 points)  23 ~ 27
right_eyebrow = landmarks[22:27]

#鼻子 (Nose: 9 points) 28 ~ 31 , 32 ~ 36
vertical_nose = landmarks[27:31]
horizontal_nose = landmarks[31:36]

#左眼 (Left eye: 6 points)  37 ~ 42
left_eye = landmarks[36:42]

#右眼 (Right eye: 6 points)  43 ~ 48
right_eye = landmarks[42:48]

#口 (Mouth: 20 points) 49 ~ 68
mouth = landmarks[48:68]

# 複製原始圖像
image_copy2 = neutral_image.copy()
# 在image_copy圖像上繪圖
img_draw = ImageDraw.Draw(image_copy2)
# 畫出 - 顎 (Jaw: 17 points) 1 ~ 17
img_draw.line(jaw.flatten().tolist(), fill='orange', width=2)
# 畫出 - 左眉 (Left eyebrow: 5 points)  18 ~ 22
img_draw.line(left_eyebrow.flatten().tolist(), fill='brown', width=2)
# 畫出 - 右眉 (Right eyebrow: 5 points)  23 ~ 27
img_draw.line(right_eyebrow.flatten().tolist(), fill='brown', width=2)
# 畫出 - 鼻子 (Nose: 9 points) 28 ~ 31 , 32 ~ 36
img_draw.line(vertical_nose.flatten().tolist(), fill='#00FF00', width=2)
img_draw.line(horizontal_nose.flatten().tolist(), fill='#00FF00', width=2)
img_draw.line(np.take(landmarks,[30,31],0).flatten().tolist(), fill='#00FF00', width=2)
img_draw.line(np.take(landmarks,[30,35],0).flatten().tolist(), fill='#00FF00', width=2)
# 畫出 - 左眼 (Left eye: 6 points)  37 ~ 42
img_draw.line(np.take(landmarks,[36,37,38,39,40,41,36],0).flatten().tolist(), fill='#00FF00', width=2)
# 畫出 - 右眼 (Right eye: 6 points)  43 ~ 48
img_draw.line(np.take(landmarks,[42,43,44,45,46,47,42],0).flatten().tolist(), fill='#00FF00', width=2)
# 畫出 - 口 (Mouth: 20 points) 49 ~ 68
img_draw.line(mouth.flatten().tolist(), fill='pink', width=3)
img_draw.line(np.take(landmarks,[60,67],0).flatten().tolist(), fill='pink', width=2)
# 畫出 - 68個點的facial landmarks

# 在PIL要畫一個可以控制大小的圖要透過以下的手法
r = 1 # 設定半徑

# 迭代出每一個點(x,y)
for i in range(landmarks.shape[0]):
    (x,y) = landmarks[i,:]
    # 以圖的中心點(x,y)來計算框住圓的邊界框座標[(x1,y1),(x2,y2)]
    img_draw.ellipse((x-r,y-r, x+r, y+r), fill='white') 
plt.figure("neutral")
plt.imshow(neutral_image)
plt.figure("result")

plt.imshow(result)
plt.figure("smile_image")
plt.imshow(smile_image)
plt.figure("111")
plt.imshow(image_copy2); plt.show()



