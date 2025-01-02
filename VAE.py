import torch
import numpy as np

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
#from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer
from PIL import Image

image_path = r".\data\smile_crop_128\S010_006_00000015.png"
vae = VQGanVAE(
    dim = 64,
    vq_codebook_size = 4096,
    vq_codebook_dim = 64,
    channels = 3
).cuda()


# 加载保存的 state_dict
state_dict = torch.load('./image128_results/vae.200000.pt')

# 将权重加载到模型中
vae.load_state_dict(state_dict)
vae = vae.cuda()
smile_image = Image.open(image_path)
smile_image = smile_image.resize((128, 128))
smile_image_np = np.array(smile_image)
#smile_image_np = np.repeat(smile_image_np[:, :, np.newaxis], 3, axis=2)

# 转换为 PyTorch Tensor，形状为 (batch_size, channels, height, width)
smile_image_tensor = torch.tensor(smile_image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
output = vae(smile_image_tensor)
print(output)

