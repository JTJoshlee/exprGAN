import torch
import numpy as np
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer
from PIL import Image

image_path = r".\data\smile_crop\S010_006_00000015.png"
vae = VQGanVAE(
    dim = 128,
    vq_codebook_size = 256,
    vq_codebook_dim = 256,
    channels = 3
).cuda()


vae.load_state_dict(torch.load('results/vae.5.pt'))
smile_image = Image.open(image_path).convert('L')
smile_image_np = np.array(smile_image)
output = vae(smile_image_np)