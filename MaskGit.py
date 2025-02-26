import torch
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

vae = VQGanVAE(
    dim = 256,
    codebook_size = 65536
).cuda()

vae.load(r'E:\style_exprGAN\results\vae.4000.pt')

transformer = MaskGitTransformer(
    num_tokens = 65536,
    seq_len = 256,
    dim = 512,
    depth = 8,
    dim_head = 64,
    heads = 8,
    ff_mult = 4
)