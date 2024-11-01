import torch
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

vae = VQGanVAE(
    dim = 256,
    codebook_size = 65536
).cuda()

vae.load(r'E:\style_exprGAN\results\vae.4000.pt')