import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
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
import cv2
import wandb
from muse_maskgit_pytorch.attn import ein_attn, sdp_attn
#from muse_maskgit_pytorch.t5 import DEFAULT_T5_NAME, get_encoded_dim, get_model_and_tokenizer, t5_encode_text
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.vqgan_vae import ResnetEncDec
from muse_maskgit_pytorch.vqgan_vae_taming import VQGanVAETaming
from loss import IdClassifyLoss
try:
    from muse_maskgit_pytorch.attn import xformers_attn

    xformer_attn = True
except ImportError:
    xformer_attn = False

from torch.utils.data import DataLoader, Dataset, random_split

from My_Maskgit_dataset import (
    ImageDataset,
    ImagecondDataset,
    get_dataset_from_dataroot,
    split_dataset_into_dataloaders
)

from torchvision.utils import make_grid
from torchvision.utils import save_image
from Enc import EASNNetwork, IdClassifier
from Dlib import Dlib
from warp import Warp
from Enc import IdClassifier



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
        y_points: Optional[torch.Tensor] = None,
        y_points_label: Optional[torch.Tensor] = None,
        cond_drop_prob=0.0,

    ):
        device, b, n = x.device, *x.shape
        context_mask = (x != 0).any(dim=-1)
        if self.training and cond_drop_prob > 0.0:
            mask = prob_mask_like((b, 1), 1.0 - cond_drop_prob, device)
            context_mask = context_mask & mask
            
        conditioning_token_ids = rearrange(conditioning_token_ids, "b ... -> b (...)")
        conditioning_token_ids = conditioning_token_ids.long()
        cond_token_emb = self.token_emb(conditioning_token_ids)
        context = cond_token_emb
        context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value=True)

        # embed tokens
        #讓x 有位置訊息的認知
        x = x.long()
        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context=context, context_mask=context_mask)
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
        
        return loss, logits, embed
    


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

def polynomial_schedule(t, power=2):
    return t**power

# class Landmark_points_decode(nn.Module):

#main maskgit classes
@beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        image_size,   
        transformer: MaskGitTransformer,
        Id_classifier: IdClassifier,
        #Encoder : EASNNetwork,
        #ID_Classifier : IdClassifier,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic: bool = False,
        vae:Optional[Union[VQGanVAE, VQGanVAETaming]] = None,
        cond_image_size: Optional[int] = None,
        noise_schedule: Callable = polynomial_schedule,
        cond_drop_prob: float = 0.4,
        self_cond_prob: float = 0.9,
        no_mask_token_prob: float = 0.0,
        critic_loss_weight: float=0.3,
        ###待做加入loss
        landmark_loss_weight: float = 0.01,
        warp_loss_weight: float = 1,
        dlib_loss_weight: float = 0.01,
        ID_loss: float = 1,
        cond_vae: Optional[Union[VQGanVAE, VQGanVAETaming]] = None,
        dim: int = 48,
        channels: int=3,
        y_point_decoder_layer: int=2,
        y_point_vae: Optional[Union[VQGanVAE, VQGanVAETaming]] = None, 
        
    ):
        super().__init__()



        self.vae = vae.copy_for_eval() if vae is not None else None
        self.y_point_vae = y_point_vae

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
        self.landmark_loss_weight = landmark_loss_weight
        self.warp_loss_weight = warp_loss_weight
        self.dlib_loss_weight = dlib_loss_weight
        self.ID_loss = ID_loss
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
        self.GetLandmark = Dlib()
        self.warp = Warp()
        self.ID_classifier = Id_classifier
        self.ID_loss_fn = IdClassifyLoss()
        self.RecnetEncDec = ResnetEncDec(dim=dim, channels=channels, layers=y_point_decoder_layer)
        self.y_point_input_size = image_size**2
        ###通道轉換層###
        self.channel_adapter = nn.Conv2d(
            in_channels=256, 
            out_channels=96, 
            kernel_size=1
        )
        self.y_point_decode = nn.Sequential(
            
            nn.Flatten(),  
            nn.Linear(12288 , 68 * 2)
        )
        # self.y_point_decode = nn.Sequential(
        #     nn.Linear(256 * 48, 2048),
        #     nn.LeakyReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 68 * 2)
        # )
        

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
        # neutral_landmark: Optional[torch.Tensor] = None,
        # smile_landmark: Optional[torch.Tensor] = None,
        base_image: Optional[torch.Tensor] = None,
        fmap_size=None,
        timesteps=18,
        cond_scale=3,
        topk_filter_thres=0.67,
        can_remask_prev_masked=False,
        temperature=1.0,
        force_not_use_token_critic=False,
        critic_noise_scale=1,
    ):
        
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))
        self.cond_image_size = (fmap_size, fmap_size)
        if neutral_image.shape[3] == 64:
            self.codebook_size = 2048
        else: self.codebook_size = 4096
        #begin with neutral image

        device = next(self.parameters()).device
        print("device",device)
        batch_size = len(neutral_image)
        seq_len = fmap_size**2 
        shape = (batch_size, seq_len)
        if base_image is not None:
            _, base_ids, _ = self.vae.encode(base_image)
            base_ids = rearrange(base_ids, "b ... -> b (...)") 
        
        ##直接把neutral image當作底
        # mask = 1 - cond_image
        # neutral_image = neutral_image * mask
        
        _, ids, _ = self.vae.encode(neutral_image)
        #print("ids max", ids.max())
        ids = rearrange(ids, "b ... -> b (...)")

        cond_images = F.interpolate(cond_image, self.cond_image_size, mode="nearest")
        cond_images_transformed = cond_images[:, 0, :, :]
        cond_images_ids = rearrange(cond_images_transformed, "b ... -> b (...)")
        
        scores = torch.zeros(shape, dtype=torch.float32, device=device)
        starting_temperature = temperature
        

        demask_fn = self.transformer.forward_with_cond_scale

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        if self.resize_image_for_cond_image:
                if cond_image is None:
                    raise ValueError("conditioning image must be passed in to generate for super res maskgit")
                #with torch.no_grad():
                    #_, cond_ids, _ = self.cond_vae.encode(cond_image)
        
        #cond_ids = rearrange(cond_ids, "b ... -> b (...)")
        if base_image is not None:
            cond_ids = torch.cat((ids, cond_images_ids, base_ids), dim=-1)
        else:
            cond_ids = torch.cat((ids, cond_images_ids), dim=-1)
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

        #_, ids, _ = self.vae.encode(neutral_image)  
        ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)

        if not exists(self.vae):
            return ids
        
        #print(ids.max(), ids.min())  # 檢查 ids 中的最大值和最小值
        if ids.max() > self.codebook_size:
            return None
        images = self.vae.decode_from_ids(ids)
        
        return images

    def forward(
        self,
        smile_image: torch.Tensor,
        neutral_image: torch.Tensor,
        cond_image: torch.Tensor,
        neutral_landmark:torch.Tensor,
        smile_landmark:torch.Tensor,
        base_image: Optional[torch.Tensor] = None,
        cond_token_ids: Optional[torch.Tensor] = None,
        ignore_index=-1,
        train_only_generator=False,
        sample_temperature=None,
        cond_drop_prob=None,
        fmap_size=None,
        timesteps=18,
        topk_filter_thres=0.67,
        starting_temperature=1.0,
        can_remask_prev_masked=False,
        cond_scale=3,

    ):
        #print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #print(neutral_image.shape[3])
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))
        self.cond_image_size = (fmap_size, fmap_size)
        if neutral_image.dtype == torch.float:
            if self.vae is None:
                raise ValueError("you must pass in a vae if you want to train from raw images")

            if not all([height_or_width == self.image_size for height_or_width in neutral_image.shape[-2:]]):
                
                raise ValueError("the image you passed in is not of the correct dimensions")

            with torch.no_grad():
                ###把cond image 當 mask###
                #mask = 1 - cond_image
                #print("cond image shape", cond_image.shape)
                #neutral_image = neutral_image * mask
                ###顯示圖片專用###
                # plt.figure("neutral_image")
                #neutral_image = neutral_image.cpu().detach().squeeze(0).permute(1,2,0)
                # plt.imshow(neutral_image)
                # plt.show()
                _, smile_ids, _ = self.vae.encode(smile_image)
                _, neutral_ids, _ = self.vae.encode(neutral_image)
        elif self.resize_image_for_cond_image is True:
                raise ValueError(
                    "you cannot pass in raw image token ids if you want autoresizing of images for conditioning"
                )
        else:
            
            smile_ids = smile_image
        
        if base_image is not None:
            _, base_ids, _ = self.vae.encode(base_image)
            base_ids = rearrange(base_ids, "b ... -> b (...)")
        #get some basic variables
        smile_ids = rearrange(smile_ids, "b ... -> b (...)")
        neutral_ids = rearrange(neutral_ids, "b ... -> b (...)")
        #print("smile ids shape", smile_ids.shape)
        #print("neutral ids shape", neutral_ids.shape)
        batch, seq_len, device, cond_drop_prob = (
            *smile_ids.shape,
            smile_ids.device,
            default(cond_drop_prob, self.cond_drop_prob),
        )
        shape = (batch, seq_len)
        #調整image size
        if self.resize_image_for_cond_image:
            cond_images = F.interpolate(cond_image, self.cond_image_size, mode="nearest")
        
        #print("cond image size", cond_images.shape)
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

            # assert all(
            #     [height_or_width == self.cond_image_size for height_or_width in cond_images.shape[-2:]]
            # )

            #with torch.no_grad():
                #_, cond_token_ids, _ = self.cond_vae.encode(cond_images)

        #cond_token_ids = rearrange(cond_token_ids, "b ... -> b (...)")
        #print(neutral_ids.shape)
        cond_images_transformed = cond_images[:, 0, :, :]
        #print(cond_images_transformed.shape)
        cond_images_ids = rearrange(cond_images_transformed, "b ... -> b (...)")
        if base_image is not None:
            cond_token_ids = torch.cat((neutral_ids, cond_images_ids, base_ids), dim=-1)
        else:
            cond_token_ids = torch.cat((neutral_ids, cond_images_ids), dim=-1)
        #prepare mask
        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")

        
        mask_id = self.transformer.mask_id
        labels = torch.where(mask, smile_ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask
        
        x: torch.Tensor = torch.where(mask, mask_id, smile_ids)
        #cond_token_ids_mask: torch.Tensor = torch.where(mask, mask_id, cond_token_ids)
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
        if neutral_image.shape[3] == int(64):
            smile_landmark = smile_landmark/2


        ce_loss, logits, embed = self.transformer(
            x,
            self_cond_embed=self_cond_embed,
            conditioning_token_ids=cond_token_ids,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            
            return_logits=True,
        )


        

    ### y^points 預測 landmark loss
        if neutral_image.shape[3] == int(64):
            smile_landmark = smile_landmark/2
            neutral_landmark = neutral_landmark/2
        
        
        pred_ids = gumbel_sample(logits, dim=-1)    
        
        pred_ids = rearrange(pred_ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
        y_points_decode_logits = self.y_point_vae.decode_from_ids(pred_ids)
        y_points_output = self.y_point_decode(y_points_decode_logits)
        y_points_output = y_points_output.view(-1, 68, 2)       
        #y_points_output = y_points_output.view(x.size(0), 68, 2)
        
        
        landmark_loss = F.mse_loss(y_points_output, smile_landmark)
              
        

        #generate_image = self.vae.decode_from_ids(pred_ids)
        #print("generate ids", generate_ids.shape)
    ### dlib loss
        ##########################
        # #start_time = time.time()
        generate_image = self.generate(
            neutral_image=neutral_image,
            cond_image=cond_image,
            # neutral_landmark=neutral_landmark,
            # smile_landmark=smile_landmark,
            base_image=base_image,
            timesteps=8,
        )
        # #end_time = time.time()

        # #execution_time = end_time - start_time

        # #print(f"產生image時間: {execution_time} 秒")
        ########################
        # dlib_loss = 100
        # warp_loss = 0
        # ID_loss = 0
        ##測試用
       
        if generate_image is None:
            generate_image = neutral_image
            print("use generate image")
        # print(generate_image.shape)
        # print(smile_landmark.type)
        # print(smile_landmark.shape)
        ####
        #ce_loss = F.l1_loss(generate_image, smile_image)
        ########################    
        #start_time = time.time()
        y_global_landmark = self.GetLandmark.input_image(generate_image)
        if y_global_landmark is None:
            y_global_landmark = neutral_landmark
            print("use neutral_landmark")
        # print("y_global_landmark",y_global_landmark)
        # print("neutral landmark", neutral_landmark)
        #y_global_landmark = y_global_landmark.detach().requires_grad_(True)
        dlib_loss = F.mse_loss(y_global_landmark, smile_landmark)  
        #end_time = time.time()

        #execution_time = end_time - start_time

        #print(f"landmark時間: {execution_time} 秒")  
        #################################
### warp loss
        #print("neutral landmark", neutral_landmark)
        #start_time = time.time()
        y_global_warp = self.warp.get_warp_image(smile_image, y_global_landmark, neutral_landmark)
        #print(y_global_warp.shape)
        #y_global_warp = y_global_warp.detach().requires_grad_(True)
        warp_loss = F.l1_loss(neutral_image, y_global_warp)
        #end_time = time.time()

        #execution_time = end_time - start_time

        #print(f"warp時間: {execution_time} 秒")
###處理ID這塊
        #start_time = time.time()    
        y_global_warp_float = y_global_warp.float()
        #y_global_warp = y_global_warp.clone().detach().requires_grad_(True)
        transform = T.Resize((128,128))
        if neutral_image.shape[3] == int(64):            
            y_global_warp_float = transform(y_global_warp_float)
            
            neutral_image = transform(neutral_image)
        ID_classified = self.ID_classifier(neutral_image, y_global_warp_float)
        ID_target = torch.ones(batch, dtype=torch.float32).unsqueeze(1).to("cuda")
        ID_loss = self.ID_loss_fn(ID_classified, ID_target)
        #end_time = time.time()

        #execution_time = end_time - start_time

        #print(f"ID時間: {execution_time} 秒")

            ###待做 處理ID的部分               

        #wandb.log({"landmark loss": landmark_loss.item()})
        print("landmark loss", landmark_loss)
        print("warp loss", warp_loss)
        print("dlib loss", dlib_loss)
        print("ID loss", ID_loss)
        print("ce loss", ce_loss)          
        
        if isnan(ce_loss):
            self.print(f"ERROR: found NaN loss: {ce_loss}")
            raise ValueError("NaN loss")
        ###做 Uncertainty Weigthing
        #log_vars = nn.Parameter(torch.zeros(5))
        total_loss = (ce_loss 
        + self.landmark_loss_weight*landmark_loss
        + self.warp_loss_weight*warp_loss
        + self.dlib_loss_weight*dlib_loss
        + self.ID_loss*ID_loss)
        if not exists(self.token_critic) or train_only_generator:
            ###做 Uncertainty Weigthing
            
            return total_loss

        #token critic loss

        sampled_ids = gumbel_sample(logits, temperature=default(sample_temperature, random()))

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (smile_ids != critic_input).float()

        bce_loss = self.token_critic(
                critic_input,
                conditioning_token_ids=cond_token_ids,
                labels=critic_labels,
                cond_drop_prob=cond_drop_prob,
            )

        
        return total_loss + self.critic_loss_weight * bce_loss #+ y_points_loss

    # def total_loss(self, ce_loss, y_points_loss, warp_loss, dlib_loss, ID_loss, log_vars):
    #     losses = [ce_loss, y_points_loss, warp_loss, dlib_loss, ID_loss]
    #     loss_sum = 0
    #     for i, loss in enumerate(losses):
    #         precision = torch.exp(-log_vars[i])  # 計算 1/σ^2
    #         loss_sum += precision * loss + log_vars[i]  # 1/σ^2 * L + log(σ)
    #     return loss_sum
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
        superres_image_tensor:torch.tensor,
        superres_cond_image_tensor:torch.tensor,
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
            neutral_image=superres_image_tensor,
            cond_scale=cond_scale,
            cond_image=superres_cond_image_tensor,
            base_image=lowres_image,
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
        project_name: str = "muse_maskgit"
        wandb_user: str = None
        run_name: str = None
        total_params: Optional[int] = None
        image_size: int = 64
        num_tokens: int = 2048
        num_train_steps: int = -1
        num_epochs: int = 100000
        dim: int = 48
        channels: int = 3
        batch_size: int = 4
        lr: float = 1e-4
        gradient_accumulation_steps: int = 1
        save_results_every: int = 10000
        save_model_every: int = 10000
        vq_codebook_size: int = 2048
        vq_codebook_dim: int = 48
        lr_scheduler: str = "constant"
        lr_warmup_steps: int = 0
        seq_len: int = 256
        depth: int = 6
        dim_head: int = 128
        heads: int = 8
        ff_mult: int = 4    
        mixed_precision: str = "no"
        cond_image_size: Optional[int] = 64
        timesteps: int = 18
        optimizer: str = "Lion"
        only_save_last_checkpoint: bool = True
        validation_image_scale: float = 1.0
        no_center_crop: bool = False
        no_flip: bool = False
        dataset_save_path: Optional[str] = "dataset"
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
        cond_image_column: str = "cond_image"
        log_with: str = "wandb"
        use_8bit_adam: bool = False
        results_dir: str = "try_results"
        logging_dir: Optional[str] = None
        vae_path: Optional[str] = r'C:\Style-exprGAN\exprGAN-main\image64_result64_vae\vae.400000.pt'
        base_maskgit_path: Optional[str] = r"C:\Style-exprGAN\exprGAN-main\image64_maskgit_condimage_multi\maskgit_bases.460000.pt" 
        superres_maskgit_path: Optional[str] = r"C:\Style-exprGAN\exprGAN-main\image64_results\maskgit_bases.1200000.pt"
        dataset_name: Optional[str] = None
        hf_split_name: Optional[str] = None
        streaming: bool = False
        neutral_image_file = r"C:\Style-exprGAN\exprGAN-main\data\neutral_crop_align_128"
        attentionMap_dir = r"C:\Style-exprGAN\exprGAN-main\data\appearance_map"
        train_data_dir: Optional[str] = r"C:\Style-exprGAN\exprGAN-main\data\smile_data"
        cond_image_dir: Optional[str] = r"C:\Style-exprGAN\exprGAN-main\data\appearance_map"
        ID_Classifier_path: Optional[str] = r'C:\Style-exprGAN\exprGAN-main\model\ID_classifier_model\ID_epoch7500.pth'    
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
        latest_checkpoint: bool = True
        do_not_save_config: bool = False
        use_l2_recon_loss: bool = False
        debug: bool = False
        config_path: Optional[str] = None
        attention_type: str = "xformers"
        precompute: bool = False
        precompute_path: str = ""
        layers = 2
        discr_layers = 4

    parser = argparse.ArgumentParser()
    args = parser.parse_args(namespace=Arguments())
    print(torch.cuda.is_available())
    kmeans_map_file = r"C:\Style-exprGAN\exprGAN-main\data\kmeans_map"
    
    vae = VQGanVAE(
        dim=args.dim,
        vq_codebook_dim=args.vq_codebook_dim,
        vq_codebook_size=args.vq_codebook_size,
        l2_recon_loss=args.use_l2_recon_loss,
        channels=args.channels,
        layers=args.layers,
        discr_layers=args.discr_layers,
        ).to("cuda")
    vae_checkpoint = torch.load(args.vae_path)
    
    vae.load_state_dict(vae_checkpoint)
    vae.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer = MaskGitTransformer(
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

    IdClassifier_checkpoint = torch.load(args.ID_Classifier_path)    
    IdClassifier_model=IdClassifier()
    IdClassifier_model.load_state_dict(IdClassifier_checkpoint)
    IdClassifier_model.requires_grad_(False)
    # transformer_checkpoint = torch.load(args.maskgit_path)
    # transformer.load_state_dict(transformer_checkpoint)
    base_maskgit = MaskGit(
        vae=vae,  # vqgan vae
        transformer=transformer,  # transformer
        Id_classifier=IdClassifier_model,        
        image_size=args.image_size,  # image size
        cond_drop_prob=args.cond_drop_prob,  # conditional dropout, for classifier free guidance
        cond_image_size=args.cond_image_size,
    ).to(device)
    maskgit = torch.load(args.base_maskgit_path)
    
    base_maskgit.load_state_dict(maskgit)
    base_maskgit.eval()
    # for module in base_maskgit.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.track_running_stats = False
    #         module.momentum = None

    # muse = Muse(
    #     base=base_maskgit
    # )
    transform_list = [
        T.Resize(args.image_size),
        T.RandomHorizontalFlip(),
        T.CenterCrop(args.image_size),
        T.RandomCrop(args.image_size, pad_if_needed=True),
        T.ToTensor()
    ]
    transform = T.Compose(transform_list)
    neutral = [f for f in os.listdir(args.neutral_image_file)]
    attentionmap_path = r"C:\Style-exprGAN\exprGAN-main\data\kmeans_map"
    attentionmap = [f for f in os.listdir(attentionmap_path)]
    ###VAE 產生結果
    # vae_image_path = r"C:\Style-exprGAN\exprGAN-main\vae_output"
    # for image_name in neutral:

    #     neutral_img_path = os.path.join(args.neutral_image_file, image_name)
    #     neutral_image = Image.open(neutral_img_path)
    #     neutral_image = transform(neutral_image)
    #     neutral_image = neutral_image.unsqueeze(0)
    #     fmap_size = vae.get_encoded_fmap_size(args.image_size)
    #     _, ids, _ = vae.encode(neutral_image)
    #     print(ids.shape)
    #     ids = rearrange(ids, "b ... -> b (...)" )
    #     ids =rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)
        
    #     image = vae.decode_from_ids(ids)
    #     save_dir = Path(vae_image_path)
    #     save_dir.mkdir(exist_ok=True, parents=True)
    #     save_file = save_dir.joinpath(f"{image_name}")
    #     save_image(image, save_file, "png")

    #neutral_image_np = np.array(neutral_image)
    #neutral_image_tensor = torch.tensor(neutral_image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    #plt.imshow(neutral_image_tensor.cpu().detach().squeeze(0).permute(1,2,0))
    cond = [f for f in os.listdir(args.attentionMap_dir)]
    cond_image_name = cond[6]
    cond_img_path = os.path.join(args.attentionMap_dir, cond_image_name)
    cond_image = Image.open(cond_img_path)
    cond_image_np = np.array(cond_image)
    cond_image_tensor = torch.tensor(cond_image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    cond_image_tensor = F.interpolate(cond_image_tensor, size=(64, 64), mode='bilinear', align_corners=False)
    
    kmeans_image_list = []
    kmeans_transform_list = []
    kmeans_map = [f for f in os.listdir(kmeans_map_file)]
    for filename in kmeans_map:      
        kmeans_img_path = os.path.join(kmeans_map_file, filename)
        kmeans_image = Image.open(kmeans_img_path)        
        kmeans_image_list.append(kmeans_image)
    for image in kmeans_image_list:        
        kmeans_image = transform(image)
        print(kmeans_image.shape)
        kmeans_transform_list.append(kmeans_image)

    def find_image(data_root, target_name, extensions):
        for ext in extensions:
            for path in Path(data_root).rglob(f"*.{ext}"):  # 遞迴搜尋所有副檔名符合的檔案
                if target_name.lower() in path.stem.lower():  # 檢查名稱
                    print(path)
                    return path  # 找到就回傳
        return None
    extensions = ["jpg", "jpeg", "png", "webp"]
    save_dir_path =  r"C:\Style-exprGAN\exprGAN-main\data\lower_image_file"   
    kmeans_path = r"C:\Style-exprGAN\exprGAN-main\data\neutral_crop_equalize_brightnes_128"
    ###產生lower image
    # for i in range(5):
    #     kmeans_path = fr"C:\Style-exprGAN\exprGAN-main\data\smile_data\kmeans_{i}"
    #     kmeans0 = [f for f in os.listdir(kmeans_path)]
    #     for image_name in kmeans0:
    #         neutral_img_path = os.path.join(kmeans_path, image_name)
    #         neutral_image = Image.open(neutral_img_path)
    #         cond_image_path = find_image(kmeans_path, "attention", extensions)
    #         cond_image = Image.open(cond_image_path)
    #         neutral_image = transform(neutral_image).unsqueeze(0).to("cuda")
    #         cond_image = transform(cond_image).unsqueeze(0).to("cuda")
            
    #         images = base_maskgit.generate(
    #             neutral_image= neutral_image,
    #             cond_image=cond_image,
    #             cond_scale=3,
    #             temperature=1,
    #             timesteps=30
    #         )
    #         print(images.shape)
    #         save_dir = Path(save_dir_path)
    #         save_dir.mkdir(exist_ok=True, parents=True)
    #         save_file = save_dir.joinpath(f"{image_name}")
    #         save_image(images, save_file, "png")
    image_name = "S045_003_00000001.png"
    neutral_img_path = os.path.join(kmeans_path, image_name)
    neutral_image = Image.open(neutral_img_path)
    neutral_image = transform(neutral_image).unsqueeze(0).to("cuda")
    neutral_image_batch = neutral_image.repeat(5, 1, 1, 1).to("cuda")
    print(neutral_image_batch.shape)
    
    kmeans_transform_list = torch.stack(kmeans_transform_list, dim=0).to("cuda")
    print(kmeans_transform_list.shape)
    images = base_maskgit.generate(
                neutral_image= neutral_image_batch,
                cond_image=kmeans_transform_list,
                cond_scale=3,
                temperature=1,
                timesteps=18
            ).to("cuda")
    
    print(image)


    save_dir = Path(args.results_dir).joinpath("MaskGit")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_file = save_dir.joinpath(f"maskgit_{13}.png")
    
    validation_image = neutral_image_batch.squeeze(0)
    cond_image = kmeans_transform_list.squeeze(0)
    images = images.squeeze(0)

    validation_subset = validation_image[:5]
    cond_subset = cond_image[:5]
    images_subset = images[:5]

    # 展開成 grid（每行 3 張，共 8 行）
    combined = torch.stack([validation_subset, cond_subset, images_subset], dim=1)  # (8, 3, 3, 64, 64)

# 重新排列成 (8×3, 3, 64, 64)，展平成 1 維批次
    combined = combined.view(-1, 3, 64, 64)  # (24, 3, 64, 64)

# 建立圖片網格，每行 3 張，共 8 行
    grid = make_grid(combined, nrow=3)
    # print("valid shape", validation_image.shape)
    # print("cond image", cond_image.shape)
    # print("images", images.shape)
    # grid = make_grid([validation_image, cond_image, images], nrow=8)
    
    save_image(grid, save_file, "png")
    
    
    dataset = get_dataset_from_dataroot(
                args.train_data_dir,
                args.neutral_image_file,
                cond_image_root = args.cond_image_dir,
                image_column=args.image_column,
                cond_image_column=args.cond_image_column,
                save_path=args.dataset_save_path,
            )
    embeds=[]
    dataset = ImagecondDataset(
            dataset,
            args.image_size,            
            center_crop=False if args.no_center_crop else True,
            flip=False if args.no_flip else True,
            using_taming=False if not args.taming_model_path else True,
            #random_crop=args.random_crop if args.random_crop else False,
            #alpha_channel=False if args.channels == 3 else True,
            embeds=embeds,
        )
    
    dataloader, validation_dataloader = split_dataset_into_dataloaders(
        dataset,
        args.valid_frac if not args.streaming else 0,
        args.seed,
        args.batch_size,
    )
    
    image_paths = [os.path.join(args.neutral_image_file, fname) for fname in os.listdir(args.neutral_image_file) if fname.endswith('.png')]
    neutral_dataset = ImageDataset(dataset=image_paths, image_size=(64, 64), flip=True, center_crop=True)
    print(len(neutral_dataset))
    neutral_dataloader = DataLoader(neutral_dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        #neutral_image = next(iter(neutral_dataloader))
        #print(neutral_image[0].shape) 
        imgs, neutral_imgs, cond_image = next(iter(dataloader))
        next_imgs, next_neutral_imgs, next_cond_image = next(iter(dataloader))
        
        print("imgs size", imgs.size())
        print("cond image size", cond_image.size())
        random_tensor = torch.randn((1,3,64,64)).to("cuda")
        imgs = imgs.to("cuda")
        cond_image = cond_image.to("cuda")
        next_neutral_imgs = next_neutral_imgs.to("cuda")
        images = maskgit.generate(
                neutral_image=neutral_imgs,
                cond_image=cond_image,
                cond_scale=3,
                temperature=1,
                timesteps=18
            ).to("cuda")
        
        
        save_dir = Path(args.results_dir).joinpath("MaskGit")
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir.joinpath(f"maskgit_{0}.png")
       
        validation_image = next_neutral_imgs.squeeze(0)
        cond_image = cond_image.squeeze(0)
        images = images.squeeze(0)

        validation_subset = validation_image[:args.batch_size]
        cond_subset = cond_image[:args.batch_size]
        images_subset = images[:args.batch_size]

        # 展開成 grid（每行 3 張，共 8 行）
        combined = torch.stack([validation_subset, cond_subset, images_subset], dim=1)  # (8, 3, 3, 64, 64)

# 重新排列成 (8×3, 3, 64, 64)，展平成 1 維批次
        combined = combined.view(-1, 3, 64, 64)  # (24, 3, 64, 64)

# 建立圖片網格，每行 3 張，共 8 行
        grid = make_grid(combined, nrow=3)
        # print("valid shape", validation_image.shape)
        # print("cond image", cond_image.shape)
        # print("images", images.shape)
        # grid = make_grid([validation_image, cond_image, images], nrow=8)
        
        save_image(grid, save_file, "png")
           
            
        # print("validation image[0]", validation_image.type())
        # plt.figure("validation image")
        # validation_immmm= validation_image[0].cpu().detach().permute(1,2,0)
        # plt.imshow(validation_immmm)
        # plt.show()
        neutral_image = neutral_image.cpu().detach().squeeze(0).permute(1,2,0)
        imgs = imgs.cpu().detach()
        cond_image = cond_image.cpu().detach()
        images = images.cpu().detach()
        next_imgs = next_neutral_imgs.cpu().detach()
        random_tensor = random_tensor.cpu().detach()
        # 如果 images 是 batch (形狀: [B, C, H, W])，選擇第一張圖片
        if images.ndim == 4:
            images = images[0]  # 取 batch 中的第一張圖片
            imgs = imgs[0]
            cond_image = cond_image[0]
            next_imgs = next_imgs[0]
            random_tensor = random_tensor[0]
        # 如果是單通道 (灰階) 或 RGB (3通道)
        if images.shape[0] == 1:  # 單通道
            images = images[0]  # 轉成 2D array
        elif images.shape[0] == 3:  # RGB
            images = images.permute(1, 2, 0)  # 調整維度從 [C, H, W] -> [H, W, C]
            imgs = imgs.permute(1,2,0)
            cond_image = cond_image.permute(1,2,0)
            next_imgs = next_imgs.permute(1,2,0)
            random_tensor = random_tensor.permute(1,2,0)
        # 顯示圖片
        #plt.figure("neutral image")
        #plt.imshow(neutral_image)
        #plt.show()
        # plt.figure("random tensor")
        # plt.imshow(random_tensor)
        # plt.figure("imgs")
        # plt.imshow(imgs)
        # plt.figure("cond image")
        # plt.imshow(cond_image)
        # plt.figure("images")
        # plt.imshow(images)
        # plt.axis("off")  # 隱藏座標軸
        # plt.show()

