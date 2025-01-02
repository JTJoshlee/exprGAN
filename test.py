import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageFile
smile_path = r"E:\style_exprGAN\ORL_data\choosed\data_smile\smile (4).jpg"
image_size = 128
img = Image.open(smile_path)

transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])
img_trans = transform(img)
img_trans = img_trans.permute(1, 2, 0)
plt.imshow(img_trans)
plt.show()