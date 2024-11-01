import torch
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer

#smile_data_path = 

vae = VQGanVAE(
    dim = 256,
    codebook_size = 65536
)

trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 128,
    folder = r'E:\style_exprGAN\ORL_data\choosed\data_smile',
    batch_size = 4,
    grad_accum_every = 8,
    num_train_steps = 50000
).cuda()

trainer.train()