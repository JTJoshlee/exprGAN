#from fast_pytorch_kmeans import KMeans
from sklearn.cluster import KMeans
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import shutil
def Kmeans_attentionMap(image_tensor):
    print(image_tensor.shape)
    kmeans = KMeans(n_clusters=5, init='k-means++', verbose=1, n_init=20)
    labels = kmeans.fit_predict(image_tensor)
    centroids = kmeans.cluster_centers_
    

    return labels, centroids

def find_closest_image(centroid, images):
        distances = torch.norm(images - centroid, dim=1)  # 計算所有圖片到中心點的距離
        closest_idx = torch.argmin(distances)  # 找出最接近的圖片索引
        return closest_idx.item()


def classify_kMeans(labels, image_tensor, image_name):
        for i in range(len(image_tensor)):
            save_dir = os.path.join(kmeans_data_file,f"kmeans_{labels[i]}")
            save_path = os.path.join(save_dir, f"{image_name[i]}")
            os.makedirs(save_dir,exist_ok=True)
            shutil.copy(image_tensor[i],save_path)


if __name__ == "__main__":
    attentionMap_file = r"E:\style_exprGAN\data\kmeans_appearance_map"
    dict_path = r"E:\style_exprGAN"
    smile_images_file = r"E:\style_exprGAN\data\smile_crop_align_128"
    kmeans_data_file = r"E:\style_exprGAN\data\attentionMap_Kmeans"
    attentionMap_image_list = []
    image_paths = []
    image_name = []
    attentionMap_image_path_list = []
    for image in os.listdir(attentionMap_file):
        image_name.append(image)
        attentionMap_image_path = os.path.join(attentionMap_file, image)        
        image_paths.append(attentionMap_image_path)
        attentionMap_image  = Image.open(attentionMap_image_path)
        attentionMap_image_list.append(attentionMap_image)

        #smile_image_path = os.path.join(smile_images_file, image)
        #smile_image_path_list.append(smile_image_path)
    

    image_array = np.array(attentionMap_image_list)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = image_tensor.view(image_tensor.shape[0], -1)    
    labels, centroids = Kmeans_attentionMap(image_tensor)
   
    #classify_kMeans(labels, image_paths, image_name)
    k = 5
    closest_images = []
    for i in range(k):
        closest_idx = find_closest_image(centroids[i], image_tensor)
        closest_images.append(image_paths[closest_idx])  # 記錄最接近的圖片路徑
    
    # 顯示代表圖片
    plt.figure(figsize=(12, 6))
    for i, img_path in enumerate(closest_images):
        image_name = os.path.basename(img_path)
        save_dir = os.path.join(kmeans_data_file,f"kmeans_{i}","middle")
        os.makedirs(save_dir, exist_ok=True)        
        save_path = os.path.join(save_dir,f'{image_name}')
        
        img = Image.open(img_path)
        img = img.copy()
        #img.save(save_path, format="PNG")
        
        plt.imshow(img)
        plt.show()
        plt.title(f"Cluster {i}")
        plt.axis("off")