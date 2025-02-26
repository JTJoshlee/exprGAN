from typing import List
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import BaseAcceleratedTrainer
from My_transformer import MaskGit
import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
from diffusers.optimization import SchedulerType
from typing import Callable, List, Optional, Union
from ema_pytorch import EMA
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from omegaconf import OmegaConf
import torch.nn.functional as F
from torchvision.utils import make_grid
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
except ImportError:
    torch_xla = None
    xm = None
    met = None

from tqdm import tqdm



class MaskGitTrainer(BaseAcceleratedTrainer):
    def __init__(
        self,
        maskgit: MaskGit,
        dataloader: DataLoader,
        validation_dataloader: DataLoader,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: SchedulerType,
        *,
        current_step: int,
        num_train_steps: int,
        num_epochs: int = 5,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = None,
        save_results_every: int = 10,
        save_model_every: int = 1000,
        log_metrics_every: int = 10,
        results_dir="./results",
        logging_dir="./results/logs",
        apply_grad_penalty_every=4,
        use_ema=True,
        ema_vae=None,
        ema_update_after_step=0,
        ema_update_every=1,
        timesteps=18,
        clear_previous_experiments=False,
        validation_image_path: Optional[str] = "E:\style_exprGAN\data\test_data\test_neutral\S135_012_00000001",
        validation_image_scale: float = 1.0,
        only_save_last_checkpoint=False,
        args=None,
        
    ):
        super().__init__(
            dataloader=dataloader,
            valid_dataloader=validation_dataloader,
            accelerator=accelerator,
            current_step=current_step,
            num_train_steps=num_train_steps,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_results_every=save_results_every,
            save_model_every=save_model_every,
            results_dir=results_dir,
            logging_dir=logging_dir,
            apply_grad_penalty_every=apply_grad_penalty_every,
            clear_previous_experiments=clear_previous_experiments,
            validation_image_scale=validation_image_scale,
            only_save_last_checkpoint=only_save_last_checkpoint,

        )
        self.save_results_every = save_results_every
        self.log_metrics_every = log_metrics_every
        self.batch_size = batch_size
        self.current_step = current_step
        self.timesteps = timesteps

        # arguments used for the training script,
        # we are going to use them later to save them to a config file.
        self.args = args

        #maskgit

        maskgit.vae.requires_grad_(False)
        self.model: MaskGit = maskgit

        self.optim: Optimizer = optimizer
        self.lr_scheduler: SchedulerType = scheduler

        self.use_ema = use_ema
        self.validation_image_path = validation_image_path
        if use_ema:
            ema_model = EMA(
                self.model,
                ema_model=ema_vae,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
            self.ema_model = ema_model
        else:
            self.ema_model = None
        
        if self.num_train_steps <= 0:
            self.training_bar = tqdm(initial=int(self.steps.item()), total=len(self.dl) * self.num_epochs)
        else:
            self.training_bar = tqdm(initial=int(self.steps.item()), total=self.num_train_steps)
        
        self.info_bar = tqdm(total=0, bar_format="{desc}")

    def image_path_to_tensor(
        self,
        image_path
    ):
        image = Image.open(image_path)
        image_np = np.array(image)
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()

        return image_tensor
    
    def save_validation_images(
        self, 
        validation_image,
        steps: int,
        cond_image=None,
        cond_scale=3,
        temperature=1,
        timesteps=18
    ):
        # validation_image = self.image_path_to_tensor(validation_image_path)
        # cond_image_tensor = self.image_path_to_tensor(cond_image)
        images = self.model.generate(
            neutral_image=validation_image,
            cond_image=cond_image,
            cond_scale=cond_scale,
            temperature=temperature,
            timesteps=timesteps
        ).to(self.accelerator.device)

        save_dir = self.results_dir.joinpath("MaskGit")
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir.joinpath(f"maskgit_{steps}.png")
       
        validation_image = validation_image.squeeze(0)
        cond_image = cond_image.squeeze(0)
        images = images.squeeze(0)
        grid = make_grid([validation_image, cond_image, images], nrow=3)
        if self.accelerator.is_main_process:
            save_image(grid, save_file, "png")
            self.log_validation_images([Image.open(save_file)], steps, )
        return save_file

    
    def train(self):
        self.steps = self.steps + 1
        self.model.train()

        if self.accelerator.is_main_process:
            proc_label = f"[P{self.accelerator.process_index}][Master]"
        else:
            proc_label = f"[P{self.accelerator.process_index}][Worker]"

        
        
        #logs
        for epoch in range(self.current_step // len(self.dl), self.num_epochs):
            for imgs, cond_image in iter(self.dl):
                train_loss =  0.0
                steps = int(self.steps.item())
                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    loss = self.model(imgs, cond_image)
                    self.accelerator.backward(loss)
                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optim.step()
                    self.lr_scheduler.step()
                    self.optim.zero_grad()

                    if self.use_ema:
                        self.ema_model.update()

                    gathered_loss = self.accelerator.gather_for_metrics(loss)
                    train_loss = gathered_loss.mean() / self.gradient_accumulation_steps

                    logs = {"loss": train_loss, "lr": self.lr_scheduler.get_last_lr()[0]}

                    if self.on_tpu:
                        self.accelerator.print(
                            f"\n[E{epoch + 1}][{steps}]{proc_label}: "
                            f"maskgit loss: {logs['loss']} - lr: {logs['lr']}"
                        )
                    else:
                        self.training_bar.update()
                        self.info_bar.set_description_str(
                            f"[E{epoch + 1}]{proc_label}: " f"maskgit loss: {logs['loss']} - lr: {logs['lr']}"
                        )

                    self.accelerator.log(logs, step=steps)

                if not (steps % self.save_model_every):
                    self.accelerator.print(
                        f"\n[E{epoch + 1}][{steps}]{proc_label}: " f"saving model to {self.results_dir}"
                    )

                    state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                    maskgit_save_name = "maskgit_superres" if self.model.cond_image_size else "maskgit"
                    file_name = (
                        f"{maskgit_save_name}.{steps}.pt"
                        if not self.only_save_last_checkpoint
                        else f"{maskgit_save_name}.pt"
                    )

                    model_path = self.results_dir.joinpath(file_name)
                    self.accelerator.wait_for_everyone()
                    self.accelerator.save(state_dict, model_path)

                    if self.args and not self.args.do_not_save_config:
                        # save config file next to the model file.
                        conf = OmegaConf.create(vars(self.args))
                        OmegaConf.save(conf, f"{model_path}.yaml")

                    if self.use_ema:
                        self.accelerator.print(
                            f"\n[E{epoch + 1}][{steps}]{proc_label}: "
                            f"saving EMA model to {self.results_dir}"
                        )

                        ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
                        file_name = (
                            f"{maskgit_save_name}.{steps}.ema.pt"
                            if not self.only_save_last_checkpoint
                            else f"{maskgit_save_name}.ema.pt"
                        )
                        model_path = str(self.results_dir / file_name)
                        self.accelerator.wait_for_everyone()
                        self.accelerator.save(ema_state_dict, model_path)

                        if self.args and not self.args.do_not_save_config:
                            # save config file next to the model file.
                            conf = OmegaConf.create(vars(self.args))
                            OmegaConf.save(conf, f"{model_path}.yaml")

                if not (steps % self.save_results_every):
                    # cond_image = None
                    # if self.model.cond_image_size:
                    #     cond_image = F.interpolate(imgs, self.model.cond_image_size, mode="nearest")
                        

                    if self.on_tpu:
                        self.accelerator.print(f"\n[E{epoch + 1}]{proc_label}: " f"Logging validation images")
                    else:
                        self.info_bar.set_description_str(
                            f"[E{epoch + 1}]{proc_label}: " f"Logging validation images"
                        )

                    saved_image = self.save_validation_images(
                        imgs,
                        steps=steps,
                        cond_image=cond_image,
                        timesteps=self.timesteps,
                    )
                    if self.on_tpu:
                        self.accelerator.print(
                            f"\n[E{epoch + 1}][{steps}]{proc_label}: saved to {saved_image}"
                        )
                    else:
                        self.info_bar.set_description_str(
                            f"[E{epoch + 1}]{proc_label}: " f"saved to {saved_image}"
                        )

                if met is not None and not (steps % self.log_metrics_every):
                    if self.on_tpu:
                        self.accelerator.print(f"\n[E{epoch + 1}][{steps}]{proc_label}: metrics:")
                    else:
                        self.info_bar.set_description_str(f"[E{epoch + 1}]{proc_label}: metrics:")

                self.steps += 1

            # if self.num_train_steps > 0 and int(self.steps.item()) >= self.num_train_steps:
            # if self.on_tpu:
            # self.accelerator.print(
            # f"\n[E{epoch + 1}][{int(self.steps.item())}]{proc_label}"
            # f"[STOP EARLY]: Stopping training early..."
            # )
            # else:
            # self.info_bar.set_description_str(
            # f"[E{epoch + 1}]{proc_label}" f"[STOP EARLY]: Stopping training early..."
            # )
            # break

            # loop complete, save final model
            self.accelerator.print(
                f"\n[E{epoch + 1}][{steps}]{proc_label}[FINAL]: saving model to {self.results_dir}"
            )
            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            maskgit_save_name = "maskgit_superres" if self.model.cond_image_size else "maskgit"
            file_name = (
                f"{maskgit_save_name}.{steps}.pt"
                if not self.only_save_last_checkpoint
                else f"{maskgit_save_name}.pt"
            )

            model_path = self.results_dir.joinpath(file_name)
            self.accelerator.wait_for_everyone()
            self.accelerator.save(state_dict, model_path)

            if self.args and not self.args.do_not_save_config:
                # save config file next to the model file.
                conf = OmegaConf.create(vars(self.args))
                OmegaConf.save(conf, f"{model_path}.yaml")

            if self.use_ema:
                self.accelerator.print(f"\n[{steps}]{proc_label}[FINAL]: saving EMA model to {self.results_dir}")
                ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
                file_name = (
                    f"{maskgit_save_name}.{steps}.ema.pt"
                    if not self.only_save_last_checkpoint
                    else f"{maskgit_save_name}.ema.pt"
                )
                model_path = str(self.results_dir / file_name)
                self.accelerator.wait_for_everyone()
                self.accelerator.save(ema_state_dict, model_path)

                if self.args and not self.args.do_not_save_config:
                    # save config file next to the model file.
                    conf = OmegaConf.create(vars(self.args))
                    OmegaConf.save(conf, f"{model_path}.yaml")

            # cond_image = None
            # if self.model.cond_image_size:
            #     self.accelerator.print(
            #         "With conditional image training, we recommend keeping the validation prompts to empty strings"
            #     )
            #     cond_image = F.interpolate(imgs, self.model.cond_image_size, mode="nearest")

            steps = int(self.steps.item()) + 1  # get the final step count, plus one
            self.accelerator.print(f"\n[{steps}]{proc_label}: Logging validation images")
            saved_image = self.save_validation_images(imgs, steps, cond_image=cond_image)
            self.accelerator.print(f"\n[{steps}]{proc_label}: saved to {saved_image}")

            if met is not None and not (steps % self.log_metrics_every):
                self.accelerator.print(f"\n[{steps}]{proc_label}: metrics:")




    