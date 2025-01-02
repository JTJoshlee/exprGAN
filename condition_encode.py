import torch
import torch.nn.functional as F
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

vae = VQGanVAE(
    dim = 256,
    codebook_size = 65536
).cuda()


transformer = MaskGitTransformer(
    num_tokens = 65536,
    seq_len = 256,
    dim = 512,
    depth = 8,
    dim_head = 64,
    heads = 8,
    ff_mult = 4,    
)

base_maskgit = MaskGit(
    vae = vae,
    transformer = transformer,
    image_size = 256,
    cond_drop_prob = 0.25
).cuda()
