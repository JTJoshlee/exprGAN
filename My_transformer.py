import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import argparse
import math
from functools import partial
from os import PathLike
from pathlib import Path
from random import random
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator
from beartype import beartype
from einops import rearrange, repeat
from torch import einsum, isnan, nn
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from muse_maskgit_pytorch.attn import ein_attn, sdp_attn
#from muse_maskgit_pytorch.t5 import DEFAULT_T5_NAME, get_encoded_dim, get_model_and_tokenizer, t5_encode_text
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.vqgan_vae_taming import VQGanVAETaming

try:
    from muse_maskgit_pytorch.attn import xformers_attn

    xformer_attn = True
except ImportError:
    xformer_attn = False

def exists(val):
    return val is not None

def default(val, d):
    return val if val is not None else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    

    return inner
def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
    logits = logits.masked_fill(-mask, -1)

    randperm = logits.argsort(dim=-1).float()
    
    num_padding = (~mask).sum(dim=-1, keepdim=True)
    
    randperm -= num_padding
    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)

        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )

class TransformerBlocks(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, flash=True, xformers=False):
        super().__init__()
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        xformers_attn.Attention(dim=dim, dim_head=dim_head, heads=heads),
                        xformers_attn.Attention(
                            dim=dim, dim_head=dim_head, heads=heads, cross_attend=True
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        
        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, context_mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context=context,) + x

            x = ff(x) + x
        
        return self.norm(x)
        
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        dim: int,
        seq_len: int,
        dim_out: Optional[int] = None,
        add_mask_id: bool = False,
        self_cond: bool = False,
        **kwargs,

    ):
        super().__init__()
        self.dim = dim,
        self.mask_id = num_tokens if add_mask_id else None
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len
        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

        self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias=False)

    def forward_with_cond_scale(self, *args, cond_scale=3.0, return_embed=False, **kwargs):
        if cond_scale == 1:
            return self.forward(*args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs)

        logits, embed = self.forward(*args, return_embed=True, cond_drop_prob=0.0, **kwargs)
        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return (scaled_logits, embed) if return_embed else scaled_logits
    
    def forward(
        self,
        x,
        return_embed=False,
        return_logits=False,
        attentionMap_embeds: Optional[torch.Tensor] = None,
        self_cond_embed=None,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        labels=None,
        ignore_index=0,
        cond_drop_prob=0.0,

    ):
        device, b, n = x.device, *x.shape
        #context_mask = (attentionMap_embeds != 0).any(dim=-1)
        # if self.training and cond_drop_prob > 0.0:
        #     mask = prob_mask_like((b, 1), 1.0 - cond_drop_prob, device)
        #     context_mask = mask
        #     print(mask)
        conditioning_token_ids = rearrange(conditioning_token_ids, "b ... -> b (...)")
        cond_token_emb = self.token_emb(conditioning_token_ids)
        context = cond_token_emb
        #context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value=True)

        # embed tokens
        #讓x 有位置訊息的認知
        x = x.long()
        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context=context, )
        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed
        
        if not exists(labels):
            return logits
        
        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, "... 1 -> ..."), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index)

        if not return_logits:
            return loss
        
        return loss, logits
    


# self critic wrapper

class SelfCritic(nn.Module):
    def __init__(self,net):
        super.__init__()
        self.net = net
        self.to_pred  = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)
    
    def forward(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits
        
        logits = rearrange(logits, "... 1 -> ...")
        return F.binary_cross_entropy_with_logits(logits, labels)


class MaskGitTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        if kwargs.pop("add_mask_id", True) is not True:
            raise ValueError("MaskGitTransformer does not accept add_mask_id argument")
        super().__init__(*args, add_mask_id=True, **kwargs)


class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        if kwargs.pop("dim_out", 1) != 1:
            raise ValueError("TokenCritic does not accept dim_out argument")
        super().__init__(*args, dim_out=1, **kwargs)

    

def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob
    
def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)



def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs

#noise schedules
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)
    
#main maskgit classes
@beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        image_size,
        transformer: MaskGitTransformer,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic: bool = False,
        vae:Optional[Union[VQGanVAE, VQGanVAETaming]] = None,
        cond_image_size: Optional[int] = None,
        noise_schedule: Callable = cosine_schedule,
        cond_drop_prob: float = 0.2,
        self_cond_prob: float = 0.9,
        no_mask_token_prob: float = 0.0,
        critic_loss_weight: float=1.0,
        cond_vae: Optional[Union[VQGanVAE, VQGanVAETaming]] = None,
        
    ):
        super().__init__()



        self.vae = vae.copy_for_eval() if vae is not None else None


        if cond_vae is not None:
            if cond_image_size is None:
                raise ValueError("cond_image_size must be specified if conditioning")
            self.cond_vae = cond_vae.eval()
        else:
            self.cond_vae = self.vae


        if token_critic and self_token_critic:
            raise ValueError("cannot have both self_token_critic and token_critic")
        self.token_critic = SelfCritic(transformer) if self_token_critic else token_critic
        self.critic_loss_weight = critic_loss_weight
        self.image_size = image_size
        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule        
        self.transformer = transformer
        self.self_cond = transformer.self_cond
        self.cond_image_size = cond_image_size
        self.resize_image_for_cond_image = exists(cond_image_size)
        self.cond_drop_prob = cond_drop_prob
        self.self_cond_prob = self_cond_prob
        self.no_mask_token_prob = no_mask_token_prob

    @property
    def device(self):
        return self.accelerator.device if self.accelerator else next(self.parameters()).device

    def save(self, path):
        if self.accelerator:
            self.accelerator.save(self.state_dict(), path)
        else:
            torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        if not path.exists() and path.is_file():
            raise ValueError(f"cannot find file {path} (does not exist or is not a file)")
        state_dict = torch.load(str(path), map_location="cpu")
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            self.load_state_dict(state_dict, strict=False)

    def print(self, *args, **kwargs):
            return self.accelerator.print(*args, **kwargs) if self.accelerator else print(*args, **kwargs)


    @torch.no_grad()
    @eval_decorator

    def generate(
        self,
        neutral_image: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        fmap_size=None,
        timesteps=18,
        cond_scale=3,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        temperature=1.0,
        force_not_use_token_critic=False,
        critic_noise_scale=1,
    ):
        
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))
        
        #begin with neutral image

        device = next(self.parameters()).device
        
        batch_size = len(neutral_image)
        seq_len = fmap_size**2 
        shape = (batch_size, seq_len)

        
        ##直接把neutral image當作底
        _, ids, _ = self.vae.encode(neutral_image)
        ids = rearrange(ids, "b ... -> b (...)")
        
        scores = torch.zeros(shape, dtype=torch.float32, device=device)
        starting_temperature = temperature
        

        demask_fn = self.transformer.forward_with_cond_scale

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        if self.resize_image_for_cond_image:
                if cond_image is None:
                    raise ValueError("conditioning image must be passed in to generate for super res maskgit")
                with torch.no_grad():
                    _, cond_ids, _ = self.cond_vae.encode(cond_image)
        
        self_cond_embed = None

        for timestep, steps_until_x0 in tqdm(
                zip(
                    torch.linspace(0, 1, timesteps, device=device),
                    reversed(range(timesteps)),
                ),
                total=timesteps,
                dynamic_ncols=True,
            ):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices
            
           
            ids = ids.scatter(1, masked_indices, self.mask_id)
            
            logits, embed = demask_fn(
                ids,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_ids,
                cond_scale=cond_scale,
                return_embed=True,
            )

            self_cond_embed = embed if self.self_cond else None

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_id

            ids = torch.where(is_mask, pred_ids, ids)

            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale,
                )
                scores = rearrange(scores, "... 1 -> ...")

                scores = scores + (uniform(scores.shape, device=device) - 0.5) * critic_noise_scale * (
                    steps_until_x0 / timesteps
                )
            else:
                    probs_without_temperature = logits.softmax(dim=-1)

                    scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                    scores = rearrange(scores, "... 1 -> ...")

                    if not can_remask_prev_masked:
                        scores = scores.masked_fill(~is_mask, -1e5)
                    else:
                        assert (
                            self.no_mask_token_prob > 0.0
                        ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

            
        ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)

        if not exists(self.vae):
            return ids

        images = self.vae.decode_from_ids(ids)
        
        return images

    def forward(
        self,
        neutral_image: torch.Tensor,
        cond_image: torch.Tensor,
        cond_token_ids: Optional[torch.Tensor] = None,
        ignore_index=-1,
        train_only_generator=False,
        sample_temperature=None,
        cond_drop_prob=None,

    ):
        if neutral_image.dtype == torch.float:
            if self.vae is None:
                raise ValueError("you must pass in a vae if you want to train from raw images")

            if not all([height_or_width == self.image_size for height_or_width in neutral_image.shape[-2:]]):
                
                raise ValueError("the image you passed in is not of the correct dimensions")

            with torch.no_grad():
                _, ids, _ = self.vae.encode(neutral_image)
        elif self.resize_image_for_cond_image is True:
                raise ValueError(
                    "you cannot pass in raw image token ids if you want autoresizing of images for conditioning"
                )
        else:
            ids = neutral_image

        #get some basic variables
        ids = rearrange(ids, "b ... -> b (...)")
        batch, seq_len, device, cond_drop_prob = (
            *ids.shape,
            ids.device,
            default(cond_drop_prob, self.cond_drop_prob),
        )
        #調整image size
        if self.resize_image_for_cond_image:
            cond_images = F.interpolate(cond_image, self.cond_image_size, mode="nearest")

        # tokenize conditional images if needed
        if cond_images is not None:
            if cond_token_ids is not None:
                raise ValueError(
                    "if conditioning on low resolution, cannot pass in both images and token ids"
                )
            if self.cond_vae is None:
                raise ValueError(
                    "you must pass in a cond vae if you want to condition on low resolution images"
                )

            assert all(
                [height_or_width == self.cond_image_size for height_or_width in cond_images.shape[-2:]]
            )

            with torch.no_grad():
                _, cond_token_ids, _ = self.cond_vae.encode(cond_images)



        #prepare mask
        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask_id = self.transformer.mask_id
        labels = torch.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x: torch.Tensor = torch.where(mask, mask_id, ids)

        self_cond_embed = None
        
        if self.transformer.self_cond and random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    conditioning_token_ids=cond_token_ids,
                    cond_drop_prob=0.0,
                    return_embed=True,

                )

                self_cond_embed.detach_()

        #get loss

        ce_loss, logits = self.transformer(
            x,
            self_cond_embed=self_cond_embed,
            conditioning_token_ids=cond_token_ids,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
        )

        if isnan(ce_loss):
            self.print(f"ERROR: found NaN loss: {ce_loss}")
            raise ValueError("NaN loss")
        
        if not exists(self.token_critic) or train_only_generator:
            print(ce_loss)
            return ce_loss

        #token critic loss

        sampled_ids = gumbel_sample(logits, temperature=default(sample_temperature, random()))

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (ids != critic_input).float()

        bce_loss = self.token_critic(
                critic_input,
                conditioning_token_ids=cond_token_ids,
                labels=critic_labels,
                cond_drop_prob=cond_drop_prob,
            )
        
        return ce_loss + self.critic_loss_weight * bce_loss


class Muse(nn.Module):
    def __init__(self, base: MaskGit, superres: MaskGit):
        super().__init__()
        self.base_maskgit = base.eval()

        assert superres.resize_image_for_cond_image
        self.superres_maskgit = superres.eval()

    @torch.no_grad()
    def forward(
        self,
        neutral_image_tensor:torch.tensor,
        cond_image_tensor:torch.tensor,
        cond_scale=3.0,
        temperature=1.0,
        timesteps=18,
        superres_timesteps=None,
        return_lowres=False,
        return_pil_images=True,
    ):
        lowres_image = self.base_maskgit.generate(
            neutral_image=neutral_image_tensor,
            cond_image=cond_image_tensor,
            temperature=temperature,
            timesteps=timesteps
        )

        superres_image = self.superres_maskgit.generate(
            neutral_image=neutral_image_tensor,
            cond_scale=cond_scale,
            cond_images=lowres_image,
            temperature=temperature,
            timesteps=default(superres_timesteps, timesteps),
        )

        if return_pil_images:
            lowres_image = list(map(T.ToPILImage(), lowres_image))
            superres_image = list(map(T.ToPILImage(), superres_image))

        if not return_lowres:
            return superres_image

        return superres_image, lowres_image



if __name__ == "__main__":
    @dataclass
    class Arguments:
        total_params: Optional[int] = None
        image_size: int = 128
        num_tokens: int = 4096
        num_train_steps: int = -1
        num_epochs: int = 5
        dim: int = 64
        channels: int = 3
        batch_size: int = 4
        lr: float = 1e-4
        gradient_accumulation_steps: int = 1
        save_results_every: int = 1000
        save_model_every: int = 5000
        vq_codebook_size: int = 4096
        vq_codebook_dim: int = 64
        lr_scheduler: str = "constant"
        lr_warmup_steps: int = 0
        seq_len: int = 1024
        depth: int = 2
        dim_head: int = 64
        heads: int = 8
        ff_mult: int = 4
        t5_name: str = "t5-small"
        mixed_precision: str = "no"
        cond_image_size: Optional[int] = 128
        validation_prompt: str = "A photo of a dog"
        timesteps: int = 18
        optimizer: str = "Lion"
        only_save_last_checkpoint: bool = False
        validation_image_scale: float = 1.0
        no_center_crop: bool = False
        no_flip: bool = False
        dataset_save_path: Optional[str] = None
        clear_previous_experiments: bool = False
        max_grad_norm: Optional[float] = None
        seed: int = 42
        valid_frac: float = 0.05
        use_ema: bool = False
        ema_beta: float = 0.995
        ema_update_after_step: int = 1
        ema_update_every: int = 1
        apply_grad_penalty_every: int = 4
        image_column: str = "image"
        caption_column: str = "caption"
        attentionMap_column: str = "attentionMap"
        log_with: str = "wandb"
        use_8bit_adam: bool = False
        results_dir: str = "base_transform_results"
        logging_dir: Optional[str] = None
        vae_path: Optional[str] = r'E:\style_exprGAN\image128_results\vae.200000.pt'
        dataset_name: Optional[str] = None
        hf_split_name: Optional[str] = None
        streaming: bool = False
        neutral_image_file: Optional[str] = r"E:\style_exprGAN\data\neutral_crop_128"
        train_data_dir: Optional[str] = r"E:\style_exprGAN\data\smile_crop_128"
        attentionMap_dir: Optional[str] = r"E:\style_exprGAN\data\appearance_map"
        checkpoint_limit: Union[int, str] = None
        cond_drop_prob: float = 0.5
        scheduler_power: float = 1.0
        num_cycles: int = 1
        resume_path: Optional[str] = None
        taming_model_path: Optional[str] = None
        taming_config_path: Optional[str] = None
        weight_decay: float = 0.0
        cache_path: Optional[str] = None
        no_cache: bool = False
        link: bool = False
        latest_checkpoint: bool = False
        do_not_save_config: bool = False
        use_l2_recon_loss: bool = False
        debug: bool = False
        config_path: Optional[str] = None
        attention_type: str = "flash"
        precompute: bool = False
        precompute_path: str = ""
        layers = 4
        discr_layers = 4
        t5_offloading = False
    parser = argparse.ArgumentParser()
    args = parser.parse_args(namespace=Arguments())
    vae = VQGanVAE(
        dim=args.dim,
        vq_codebook_dim=args.vq_codebook_dim,
        vq_codebook_size=args.vq_codebook_size,
        l2_recon_loss=args.use_l2_recon_loss,
        channels=args.channels,
        layers=args.layers,
        discr_layers=args.discr_layers,
        ).to("cuda")

    vae.load(args.vae_path, map="cpu")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer: MaskGitTransformer = MaskGitTransformer(
        num_tokens=args.num_tokens if args.num_tokens else args.vq_codebook_size,
        # seq_len must be equivalent to fmap_size ** 2 in vae
        seq_len=args.seq_len,
        dim=args.dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        ff_mult=args.ff_mult,
        xformers=True,
    ).to(device)
    
    maskgit = MaskGit(
        vae=vae,  # vqgan vae
        transformer=transformer,  # transformer        
        image_size=args.image_size,  # image size
        cond_drop_prob=args.cond_drop_prob,  # conditional dropout, for classifier free guidance
        cond_image_size=args.cond_image_size,
    ).to(device)
    neutral = [f for f in os.listdir(args.neutral_image_file)]
    neutral_image_name = neutral[1]
    neutral_img_path = os.path.join(args.neutral_image_file, neutral_image_name)
    neutral_image = Image.open(neutral_img_path)
    neutral_img_np = np.array(neutral_image)
    noisy_image = np.clip(neutral_img_np, 0, 255).astype(np.uint8)
    noisy_image_tensor = torch.from_numpy(noisy_image).float() / 255.0      
    neutral_image_tensor = noisy_image_tensor.unsqueeze(0)
    neutral_image_tensor = neutral_image_tensor.permute(0, 3, 1, 2)
    neutral_image_tensor = neutral_image_tensor.to(device)

    cond = [f for f in os.listdir(args.attentionMap_dir)]
    cond_image_name = cond[0]
    cond_img_path = os.path.join(args.attentionMap_dir, cond_image_name)
    cond_image = Image.open(cond_img_path)
    cond_image_np = np.array(cond_image)
    cond_image_tensor = torch.from_numpy(cond_image_np).float() / 255.0      
    cond_image_tensor = cond_image_tensor.unsqueeze(0)
    cond_image_tensor = cond_image_tensor.permute(0, 3, 1, 2)
    cond_image_tensor = cond_image_tensor.to(device)
    
    images = maskgit.generate(
            neutral_image=neutral_image_tensor,
            cond_image=cond_image_tensor,
            
        ).to("cuda")
    

