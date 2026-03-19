import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch.utils.checkpoint
import transformers
from diffusers import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from einops import rearrange
from Triple_Loss import TripletLoss
from torch.utils.tensorboard import SummaryWriter
logger = get_logger(__name__, log_level="INFO")

import os
import torch
import argparse
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, \
    generate_text, auto_resume_helper
from utils.logger import *
from models import xclip
from datasets.CAP import DADA2KS
from tqdm import tqdm

triplet_loss = torch.nn.TripletMarginLoss(margin=0.3,p=2.0,eps=1e-5,swap=False,reduction="mean")
# triplet_loss1=torch.nn.TripletMarginLoss(margin=0.3)
# loss= TripletLoss(margin=0.1)
def main(
    pool: str,
    output_dir: str,
    root_path: str,
    validation_steps: int = 100,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    NUM_FRAMES: int=16,
    max_grad_norm: float = 10.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    MODEL_ARCH: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
):
    # *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    # log_dir=r"/media/ubuntu/My Passport/log/log_0.1"
    # writer=SummaryWriter("log_0.1")
    # Make one log on every process with the configuration for debugging.
    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    # Load Accident models.
    writer = SummaryWriter("log_0.1")
    model, _ = xclip.load(None, MODEL_ARCH,
                          device="cpu", jit=False,
                          T=NUM_FRAMES,
                          droppath=0.5,
                          use_checkpoint=False,
                          use_cache=False,
                          logger=logger,
                          )

    # Freeze vae and text_encoder
    model.requires_grad_(True)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )#在config文件里面定义

    train_dataset = DADA2KS(root_path=root_path, interval=1, phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        pin_memory=True, drop_last=True)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model,optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_depthprecision == "bf16":
        weight_dtype = torch.bfloat16
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers("accident prediction")
    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    global_step = 0
    first_epoch = 0
    first_epoch = global_step // num_update_steps_per_epoch
    resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(0 ,10):
        model.train()
        train_loss = 0.0
        eps=1e-5
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                device=torch.device("cuda",0)
                nv = batch["nv"].to(device,dtype=weight_dtype)
                rv = batch["rv"].to(device,dtype=weight_dtype)
                av = batch["av"].to(device,dtype=weight_dtype)
                N_t = batch["N_t"]
                R_t = batch["R_t"]
                P_t = batch["P_t"]
                C_t = batch["C_t"]
                N_t = generate_text(N_t).to(device)
                R_t = generate_text(R_t).to(device)
                P_t = generate_text(P_t).to(device)
                C_t = generate_text(C_t).to(device)
                clip1 = model(nv, N_t)
                clip2 = model(rv, R_t)
                clip3 = model(rv, P_t)
                clip4 = model(av, C_t)
                # losss=InfoNCE_loss(clip1,clip2,clip3,clip4)
                loss1 = triplet_loss(clip1, clip3, clip4) # 使clip1和clip3距离拉近，clip2和clip1距离拉远
                loss2 = triplet_loss(clip2, clip4, clip1)
                # loss1 = loss(clip1, clip3, clip4) # 使clip1和clip3距离拉近，clip2和clip1距离拉远
                # loss2 = loss(clip2, clip4, clip1)
                # print("loss1:",loss1)
                # print("loss2:",loss2)
                losss=loss1+loss2+eps
                optimizer.zero_grad()
                accelerator.backward(losss)
                writer.add_scalar('Loss/train',losss,global_step)
                # print("steps.{}".format(losss), losss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad()
                logs = {"step_loss": losss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
#
#             # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
#
                if global_step % checkpointing_steps== 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        # save_path = os.path.join(output_dir, f"checkpoint-{epoch}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
#
#                 if global_step % validation_steps == 0:
#                     if accelerator.is_main_process:
#                         for idx, batch in enumerate(val_dataloader):
        # logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        # progress_bar.set_postfix(**logs)
        # if epoch == 2:c
        if global_step >=max_train_steps:
            writer.close()
        # if global_step >= max_train_steps:
        #     writer.close()
            break

# # Create the pipeline using the trained modules and save it.
#     accelerator.wait_for_everyone()
#     if accelerator.is_main_process:
#         unet = accelerator.unwrap_model(unet)
#         accelerator.save(unet.state_dict(),save_path)
# #
# #     accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unwrap", type=str, default=None)
    parser.add_argument("--config", type=str, default="./configs/k400/car-turn.yaml")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))
