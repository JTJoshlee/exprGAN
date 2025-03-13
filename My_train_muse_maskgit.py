import logging
import os
import transformers
import argparse
from typing import Optional, Union
import My_Maskgit_dataset
from dataclasses import dataclass
import datasets
from datasets import concatenate_datasets, load_dataset
from diffusers.optimization import SchedulerType, get_scheduler
import pickle
import bz2file as bz2
import torch
from torch.optim import Optimizer
import accelerate
from accelerate.utils import ProjectConfiguration
from rich import inspect
import wandb
from muse_maskgit_pytorch.utils import (
    get_latest_checkpoints
)
from muse_maskgit_pytorch import (
    
    VQGanVAE,
    VQGanVAETaming,
    get_accelerator,
)

from My_transformer import (    
    MaskGit,   
    MaskGitTransformer,  

)
from My_Maskgit_dataset import (
    ImagecondDataset,
    get_dataset_from_dataroot,
    split_dataset_into_dataloaders
)
from My_Maskgit_trainer import MaskGitTrainer

from muse_maskgit_pytorch.trainers.base_accelerated_trainer import get_optimizer
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    print("TPU support has been disabled, please install torch_xla and train on an XLA device")
    torch_xla = None
    xm = None

# remove some unnecessary errors from transformer shown on the console.
transformers.logging.set_verbosity_error()

def compressed_pickle(title, data):
    with bz2.BZ2File(title, "w") as f:
        pickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data


@dataclass
class Arguments:
    project_name: str = "muse_maskgit"
    wandb_user: str = None
    run_name: str = None
    total_params: Optional[int] = None
    image_size: int = 128
    num_tokens: int = 4096
    num_train_steps: int = -1
    num_epochs: int = 100000
    dim: int = 64
    channels: int = 3
    batch_size: int = 1
    lr: float = 1e-4
    gradient_accumulation_steps: int = 1
    save_results_every: int = 10000
    save_model_every: int = 10000
    vq_codebook_size: int = 4096
    vq_codebook_dim: int = 64
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    seq_len: int = 1024
    depth: int = 2
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4    
    mixed_precision: str = "no"
    cond_image_size: Optional[int] = 128
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
    results_dir: str = "image128_results"
    logging_dir: Optional[str] = None
    vae_path: Optional[str] = r'E:\style_exprGAN\image128_results\vae.120000.ema.pt'
    dataset_name: Optional[str] = None
    hf_split_name: Optional[str] = None
    streaming: bool = False
    train_data_dir: Optional[str] = r"E:\style_exprGAN\data\smile_data"
    cond_image_dir: Optional[str] = r"E:\style_exprGAN\data\appearance_map"
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
    layers = 4
    discr_layers = 4

parser = argparse.ArgumentParser()

def main():
    args = parser.parse_args(namespace=Arguments())

    if accelerate.utils.is_rich_available():
        from rich import print
        from rich.traceback import install as traceback_install

        traceback_install(show_locals=args.debug, width=120, word_wrap=True)

    
    if args.debug is True:
        logging.basicConfig(level=logging.DEBUG)
        transformers.logging.set_verbosity_debug()
        datasets.logging.set_verbosity_debug()
       
    else:
        logging.basicConfig(level=logging.INFO)


    project_config = ProjectConfiguration(
        project_dir=args.logging_dir if args.logging_dir else os.path.join(args.results_dir, "logs"),
        total_limit=args.checkpoint_limit,
        automatic_checkpoint_naming=True,
    )

    accelerator: accelerate.Accelerator = get_accelerator(
        log_with=args.log_with,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        
    )


    if args.vae_path and args.taming_model_path:
        raise ValueError("Can't pass both vae_path and taming_model_path at the same time!")
    if args.train_data_dir and args.dataset_name:
        raise ValueError("Can't pass both train_data_dir and dataset_name at the same time!")

    if accelerator.is_main_process:
        accelerator.print(f"Preparing MaskGit for training on {accelerator.device.type}")
        if args.debug:
            inspect(args, docs=False)

        accelerate.utils.set_seed(args.seed)

    #Load the dataset
    with accelerator.main_process_first():
        if args.no_cache:
                pass
        else:
            dataset = get_dataset_from_dataroot(
                args.train_data_dir,
                cond_image_root = args.cond_image_dir,
                image_column=args.image_column,
                cond_image_column=args.cond_image_column,
                save_path=args.dataset_save_path,
            )
            
            


    #load the VAE
    with accelerator.main_process_first():
        if args.vae_path:
            accelerator.print("Loading Muse VQGanVAE")

            if args.latest_checkpoint:
                args.vae_path, ema_model_path = get_latest_checkpoints(args.vae_path, use_ema=args.use_ema)
                if ema_model_path:
                    ema_vae = VQGanVAE(
                        dim=args.dim,
                        vq_codebook_dim=args.vq_codebook_dim,
                        vq_codebook_size=args.vq_codebook_size,
                        l2_recon_loss=args.use_l2_recon_loss,
                        channels=args.channels,
                        layers=args.layers,
                        discr_layers=args.discr_layers,
                        accelerator=accelerator,
                    )
                    accelerator.print(f"Resuming EMA VAE from latest checkpoint: {ema_model_path}")

                    ema_vae.load(ema_model_path, map="cpu")
                else:
                    ema_vae = None

                accelerator.print(f"Resuming VAE from latest checkpoint: {args.vae_path}")
            else:
                accelerator.print("Resuming VAE from: ", args.vae_path)
                ema_vae = None

            # use config next to checkpoint if there is one and merge the cli arguments to it
            # the cli arguments will take priority so we can use it to override any value we want.
            # if os.path.exists(f"{args.vae_path}.yaml"):
            # print("Config file found, reusing config from it. Use cli arguments to override any desired value.")
            # conf = OmegaConf.load(f"{args.vae_path}.yaml")
            # cli_conf = OmegaConf.from_cli()
            ## merge the config file and the cli arguments.
            # conf = OmegaConf.merge(conf, cli_conf)

            vae = VQGanVAE(
                dim=args.dim,
                vq_codebook_dim=args.vq_codebook_dim,
                vq_codebook_size=args.vq_codebook_size,
                l2_recon_loss=args.use_l2_recon_loss,
                channels=args.channels,
                layers=args.layers,
                discr_layers=args.discr_layers,
            ).to(accelerator.device)

            vae.load(args.vae_path, map="cpu")

        elif args.taming_model_path is not None and args.taming_config_path is not None:
            accelerator.print(f"Using Taming VQGanVAE, loading from {args.taming_model_path}")
            vae = VQGanVAETaming(
                vqgan_model_path=args.taming_model_path,
                vqgan_config_path=args.taming_config_path,
                accelerator=accelerator,
            )
            args.num_tokens = vae.codebook_size
            args.seq_len = vae.get_encoded_fmap_size(args.image_size) ** 2
        else:
            raise ValueError(
                "You must pass either vae_path or taming_model_path + taming_config_path (but not both)"
            )

    # freeze VAE before parsing to transformer
    vae.requires_grad_(False)



    #create transformer/ attention network

    if args.attention_type == "flash":
        xformers = False
        flash = True
    elif args.attention_type == "xformers":
        xformers = True
        flash = True
    elif args.attention_type == "ein":
        xformers = False
        flash = False
    else:
        raise NotImplementedError(f'Attention of type "{args.attention_type}" does not exist')
    

    transformer: MaskGitTransformer = MaskGitTransformer(
        # num_tokens must be same as codebook size above
        num_tokens=args.num_tokens if args.num_tokens else args.vq_codebook_size,
        # seq_len must be equivalent to fmap_size ** 2 in vae
        seq_len=args.seq_len,
        dim=args.dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        # feedforward expansion factor
        ff_mult=args.ff_mult,                
        flash=flash,
        xformers=xformers,
    )

    maskgit = MaskGit(
        vae=vae, 
        transformer=transformer,        
        image_size=args.image_size,
        cond_drop_prob=args.cond_drop_prob,
        cond_image_size=args.cond_image_size
    )

    # load the maskgit transforemer from disk if we have previously trained one
    with accelerator.main_process_first():
        if args.resume_path is not None and len(args.resume_path) > 1:
            load = True

            accelerator.print("Loading Muse MaskGit...")

            if args.latest_checkpoint:
                try:
                    accelerator.print("resume path", args.resume_path)
                    args.resume_path, ema_model_path = get_latest_checkpoints(
                        args.resume_path,
                        use_ema=args.use_ema,
                        model_type="maskgit",
                        cond_image_size=args.cond_image_size,
                    )
                    accelerator.print(f"Resuming MaskGit from latest checkpoint: {args.resume_path}")
                    # if args.use_ema:
                    #    print(f"Resuming EMA MaskGit from latest checkpoint: {ema_model_path}")

                except ValueError:
                    load = False

            else:
                accelerator.print("Resuming MaskGit from: ", args.resume_path)

            if load:
                maskgit.load(args.resume_path)

                resume_from_parts = args.resume_path.split(".")
                for i in range(len(resume_from_parts) - 1, -1, -1):
                    if resume_from_parts[i].isdigit():
                        current_step = int(resume_from_parts[i])
                        accelerator.print(f"Found step {current_step} for the MaskGit model.")
                        break
                if current_step == 0:
                    accelerator.print("No step found for the MaskGit model.")
            else:
                current_step = 0
        else:
            accelerator.print("Initialized new empty MaskGit model.")
            current_step = 0

    # Use the parameters() method to get an iterator over all the learnable parameters of the model
    total_params = sum(p.numel() for p in maskgit.parameters())
    args.total_params = total_params

    accelerator.print(f"Total number of parameters: {format(total_params, ',d')}")

    if args.precompute_path and not args.precompute:
        embeds = decompress_pickle(args.precompute_path)
    else:
        embeds = []


    #create the dataset object
    with accelerator.main_process_first():
        
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
            

    #create the dataloaders
    dataloader, validation_dataloader = split_dataset_into_dataloaders(
        dataset,
        args.valid_frac if not args.streaming else 0,
        args.seed,
        args.batch_size,
    )

    #Create the optimizer and scheduler, wrap optimizer in scheduler
    optimizer: Optimizer = get_optimizer(
        args.use_8bit_adam, args.optimizer, set(transformer.parameters()), args.lr, args.weight_decay
    )

    if args.num_train_steps > 0:
        num_lr_steps = args.num_train_steps * args.gradient_accumulation_steps
    else:
        num_lr_steps = args.num_epochs * len(dataloader)



    scheduler: SchedulerType = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=num_lr_steps,
        num_cycles=args.num_cycles,
        power=args.scheduler_power,
    )

    # Prepare the model, optimizer, and dataloaders for distributed training
    maskgit, optimizer, dataloader, validation_dataloader, scheduler = accelerator.prepare(
        maskgit, optimizer, dataloader, validation_dataloader, scheduler
    )



    # Wait for everyone to catch up, then print some info and initialize the trackers
    accelerator.wait_for_everyone()
    
    accelerator.print(f"[{accelerator.process_index}] Ready to create trainer!")
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.project_name,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "entity": f"{args.wandb_user or wandb.api.default_entity}",
                    "name": args.run_name,
                },
            },
        )

    embeds = []

    accelerator.wait_for_everyone()
    trainer = MaskGitTrainer(
        maskgit=maskgit,
        dataloader=dataloader,
        validation_dataloader=validation_dataloader,
        accelerator=accelerator,
        optimizer=optimizer,
        scheduler=scheduler,
        current_step=current_step + 1 if current_step != 0 else current_step,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        save_results_every=args.save_results_every,
        save_model_every=args.save_model_every,
        results_dir=args.results_dir,
        logging_dir=args.logging_dir if args.logging_dir else os.path.join(args.results_dir, "logs"),
        use_ema=args.use_ema,
        ema_vae=ema_vae,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
        apply_grad_penalty_every=args.apply_grad_penalty_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,        
        timesteps=args.timesteps,
        clear_previous_experiments=args.clear_previous_experiments,
        validation_image_scale=args.validation_image_scale,
        only_save_last_checkpoint=args.only_save_last_checkpoint,
        num_epochs=args.num_epochs,
        args=args,
    )

    # Prepare the trainer for distributed training
    accelerator.print("MaskGit Trainer initialized, preparing for training...")
    trainer = accelerator.prepare(trainer)

    # Train the model!
    accelerator.print("Starting training!")
    trainer.train()

    # Clean up and wait for other processes to finish (loggers etc.)
    if accelerator.is_main_process:
        accelerator.print("Training complete!")
        accelerator.end_training()
if __name__ == "__main__": 
    main()