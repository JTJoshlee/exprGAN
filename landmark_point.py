import os
import numpy as np

neutral_path = r"E:\style_exprGAN\ORL_data\choosed\neutral_feature_points\points_1.png.npy"
smile_path = r"E:\style_exprGAN\ORL_data\choosed\smile_feature_points\points_1.png.npy"



neutral = np.load(neutral_path)
smile = np.load(smile_path)

# 打印或檢查加載的數據
print(f"neutral: {neutral}")
print(f"smile: {smile}")