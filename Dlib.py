import dlib
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

transform = transforms.Compose([
            transforms.Resize((128, 128))           
        ])

model_path = r"E:\style_exprGAN\model\shape_predictor_68_face_landmarks_GTX.dat"
neutral_path = r"E:\style_exprGAN\data\neutral_crop"
smile_path = r"E:\style_exprGAN\data\smile_crop"
class Dlib():
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68,2), dtype=dtype)

        for i in range(0,68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords
    def get_landmarks(self, im, face_detector, shape_predictor):
        rects = face_detector(im, 1)
        shape = shape_predictor(im, rects[0])
        coords = self.shape_to_np(shape, dtype="int")
            
        return coords

    def input_image(self, file, input_file_path, emotion):
        basename = os.path.basename(file)
        name, ext = os.path.splitext(basename)
        file_path = os.path.join(input_file_path, file)
        image = Image.open(file_path).convert('L')

        image = transform(image)

        img = np.array(image)

        dets = self.face_detector(img, 0)
        
        image_copy = image.copy()

        img_draw = ImageDraw.Draw(image_copy)

        for i, d in enumerate(dets):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
                .format( i, d.left(), d.top(), d.right(), d.bottom()))
            
            # 透過畫線來畫一個四方框的框線並控制粗細
            img_draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill='#00FF00', width=2)
            
      
        predictor_path = model_path

        shape_predictor = dlib.shape_predictor(predictor_path)
        image_copy2 = image.copy()

        # 在image_copy圖像上繪圖
        img_draw = ImageDraw.Draw(image_copy2)
        
        # 取得68個人臉關鍵點的座標
        landmarks = self.get_landmarks(img, self.face_detector, shape_predictor)
        print(landmarks)
        save_path = f'E:/style_exprGAN/data/{emotion}_feature_points'
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f'landmark_{name}.npy')
        np.save(file_path, landmarks)



        # #顎 (Jaw: 17 points) 1 ~ 17
        # jaw = landmarks[0:17]

        # #左眉 (Left eyebrow: 5 points)  18 ~ 22
        # left_eyebrow = landmarks[17:22]

        # #右眉 (Right eyebrow: 5 points)  23 ~ 27
        # right_eyebrow = landmarks[22:27]

        # #鼻子 (Nose: 9 points) 28 ~ 31 , 32 ~ 36
        # vertical_nose = landmarks[27:31]
        # horizontal_nose = landmarks[31:36]

        # #左眼 (Left eye: 6 points)  37 ~ 42
        # left_eye = landmarks[36:42]

        # #右眼 (Right eye: 6 points)  43 ~ 48
        # right_eye = landmarks[42:48]

        # #口 (Mouth: 20 points) 49 ~ 68
        # mouth = landmarks[48:68]

        # # 複製原始圖像
        # image_copy2 = image.copy()
        # # 在image_copy圖像上繪圖
        # img_draw = ImageDraw.Draw(image_copy2)
        # # 畫出 - 顎 (Jaw: 17 points) 1 ~ 17
        # img_draw.line(jaw.flatten().tolist(), fill='orange', width=2)
        # # 畫出 - 左眉 (Left eyebrow: 5 points)  18 ~ 22
        # img_draw.line(left_eyebrow.flatten().tolist(), fill='brown', width=2)
        # # 畫出 - 右眉 (Right eyebrow: 5 points)  23 ~ 27
        # img_draw.line(right_eyebrow.flatten().tolist(), fill='brown', width=2)
        # # 畫出 - 鼻子 (Nose: 9 points) 28 ~ 31 , 32 ~ 36
        # img_draw.line(vertical_nose.flatten().tolist(), fill='#00FF00', width=2)
        # img_draw.line(horizontal_nose.flatten().tolist(), fill='#00FF00', width=2)
        # img_draw.line(np.take(landmarks,[30,31],0).flatten().tolist(), fill='#00FF00', width=2)
        # img_draw.line(np.take(landmarks,[30,35],0).flatten().tolist(), fill='#00FF00', width=2)
        # # 畫出 - 左眼 (Left eye: 6 points)  37 ~ 42
        # img_draw.line(np.take(landmarks,[36,37,38,39,40,41,36],0).flatten().tolist(), fill='#00FF00', width=2)
        # # 畫出 - 右眼 (Right eye: 6 points)  43 ~ 48
        # img_draw.line(np.take(landmarks,[42,43,44,45,46,47,42],0).flatten().tolist(), fill='#00FF00', width=2)
        # # 畫出 - 口 (Mouth: 20 points) 49 ~ 68
        # img_draw.line(mouth.flatten().tolist(), fill='pink', width=3)
        # img_draw.line(np.take(landmarks,[60,67],0).flatten().tolist(), fill='pink', width=2)
        # # 畫出 - 68個點的facial landmarks

        # # 在PIL要畫一個可以控制大小的圖要透過以下的手法
        # r = 1 # 設定半徑

        # # 迭代出每一個點(x,y)
        # for i in range(landmarks.shape[0]):
        #     (x,y) = landmarks[i,:]
        #     # 以圖的中心點(x,y)來計算框住圓的邊界框座標[(x1,y1),(x2,y2)]
        #     img_draw.ellipse((x-r,y-r, x+r, y+r), fill='white') 


        # plt.imshow(image_copy2); plt.show()


if __name__ == "__main__":
    GetLandmark = Dlib()
    for file in os.listdir(neutral_path):   
        print(f"file: {file}")     
        GetLandmark.input_image(file, neutral_path, "neutral")

    for file in os.listdir(smile_path):
        GetLandmark.input_image(file, smile_path, "smile")