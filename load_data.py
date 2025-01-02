import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from torchvision import transforms as T
from PIL import Image
appearance_map = r"E:\style_exprGAN\data\appearance_map"
smile_path = r"E:\style_exprGAN\ORL_data\choosed\smile"


smile_image = Image.open(appearance_map).convert('L')
smile_image = smile_image.resize((128, 128))
plt.imshow(smile_image, cmap='gray')
plt.show()