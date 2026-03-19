import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import time
import copy
import  json
import re
import numpy as np
from grapth_my_dataset import DADA2KS_Graph_Inference
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import imageio

import monkey_patch_cached_download
import diffusers
import transformers

from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from pipeline import TextToVideoSDPipeline

from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder

from einops import rearrange
import clip
from model.my_box_vae_mode import SceneVAEModel
from models.control_my_mynet import ControlUNet3DConditionModel

from utils.lora import (
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    save_lora_weight,
    train_patch_pipe,
    monkeypatch_or_replace_lora,
    monkeypatch_or_replace_lora_extended
)

from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor,AutoConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModel


already_printed_trainables = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)


def normalize_lava_response(text):
    text = " ".join(text.split())
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1].strip()
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\s*-\s*$", "", text)
    text = re.sub(r"\s*-\s*\.$", ".", text)
    text = re.sub(r"\bego[- ]?cart\b", "ego-car", text, flags=re.IGNORECASE)
    text = re.sub(r"\bego[- ]?car\b", "ego-car", text, flags=re.IGNORECASE)
    return text.strip()


def finalize_reason_response(text):
    text = normalize_lava_response(text)
    first_sentence = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if first_sentence:
        return first_sentence.group(1).strip()

    last_break = max(text.rfind(","), text.rfind(";"))
    if last_break > 24:
        text = text[:last_break].strip()
    text = text.rstrip(" ,;-")
    if text and text[-1] not in ".!?":
        text += "."
    return text


def finalize_type_response(text):
    text = normalize_lava_response(text)
    sentence = text.split(".", 1)[0].strip()
    for sep in [", and ", " and ", " because ", " with ", " while ", " where "]:
        idx = sentence.lower().find(sep)
        if idx > 0:
            sentence = sentence[:idx].strip(" ,;:-")
            break

    phrase = sentence
    prefix_pattern = (
        r"(?i)^the type of the predicted accident(?: in the video)? is\s+"
    )
    phrase = re.sub(prefix_pattern, "", phrase).strip(" ,;:-")
    phrase = re.sub(r"(?i)^(it is|it's|this is)\s+", "", phrase).strip(" ,;:-")

    accident_phrase = re.search(
        r"(?i)\b((?:[a-z-]+\s+){0,4}accident)\b",
        phrase,
    )
    if accident_phrase:
        phrase = accident_phrase.group(1).strip()
    else:
        phrase = phrase.rstrip(" ,;:-")
        if not phrase.lower().endswith("accident"):
            phrase = f"{phrase} accident".strip()

    phrase = re.sub(r"(?i)^(a|an|the)\s+", "", phrase).strip(" ,;:-")
    phrase = re.sub(r"\s+", " ", phrase)
    article = "an" if phrase[:1].lower() in "aeiou" else "a"
    return f"The type of the predicted accident in the video is {article} {phrase}."


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def extend_datasets(datasets, dataset_items, extend=False):
    biggest_data_len = max(x.__len__() for x in datasets)
    extended = []
    for dataset in datasets:
        if dataset.__len__() == 0:
            del dataset
            continue
        if dataset.__len__() < biggest_data_len:
            for item in dataset_items:
                if extend and item not in extended and hasattr(dataset, item):
                    print(f"Extending {item}")
                    value = getattr(dataset, item)
                    value *= biggest_data_len
                    value = value[:biggest_data_len]
                    setattr(dataset, item, value)
                    print(f"New {item} dataset length: {dataset.__len__()}")
                    extended.append(item)



def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_state_dict(model, pretrained_model_path):
    state_dict = torch.load(pretrained_model_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model


def load_primary_models(pretrained_model_path,argss):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    contronet_path=r"./contro"
    control_unet=ControlUNet3DConditionModel.from_pretrained_control(contronet_path)
    num_objs = 7
    num_rels = 49
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model.eval().requires_grad_(False)
    pp = torch.load(r"./graph_model.bin",
                map_location=torch.device('cpu'))
    sg_model = SceneVAEModel(argss, num_objs, num_rels, clip_model)
    sg_model.load_state_dict(pp, strict=False)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    config = AutoConfig.from_pretrained(r"./Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf")
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        r"./Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        config=config,
    )
    
    output_dir = r"./lora"
    
    model = PeftModel.from_pretrained(model, output_dir)
    return noise_scheduler, tokenizer, text_encoder, vae, unet, sg_model, control_unet,model


def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, text_enable)



def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False)


def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])


def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())


def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
  
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
            unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            print("xformers has injected to unet")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


def inject_lora(use_lora, model, replace_modules, is_extended=False, dropout=0.0, lora_path='', r=16):
    injector = (
        inject_trainable_lora if not is_extended
        else
        inject_trainable_lora_extended
    )

    params = None
    negation = None

    if os.path.exists(lora_path):
        try:
            for f in os.listdir(lora_path):
                if f.endswith('.pt'):
                    lora_file = os.path.join(lora_path, f)

                    if 'text_encoder' in f and isinstance(model, CLIPTextModel):
                        monkeypatch_or_replace_lora(
                            model,
                            torch.load(lora_file),
                            target_replace_module=replace_modules,
                            r=r
                        )
                        print("Successfully loaded Text Encoder LoRa.")
                    if 'unet' in f and isinstance(model, UNet3DConditionModel):
                        monkeypatch_or_replace_lora_extended(
                            model,
                            torch.load(lora_file),
                            target_replace_module=replace_modules,
                            r=r
                        )
                        print("Successfully loaded UNET LoRa (excluding ControlNet).")

        except Exception as e:
            print(e)
            print("Could not load LoRAs. Injecting new ones instead...")

    if use_lora:
        REPLACE_MODULES = replace_modules
        injector_args = {
            "model": model,
            "target_replace_module": REPLACE_MODULES,
            "r": r
        }
        if not is_extended: injector_args['dropout_p'] = dropout

        params, negation = injector(**injector_args)
        for _up, _down in extract_lora_ups_down(
                model,
                target_replace_module=REPLACE_MODULES):

            if all(x is not None for x in [_up, _down]):
                print(f"Lora successfully injected into {model.__class__.__name__}.")

            break

    return params, negation


def save_lora(model, name, condition, replace_modules, step, save_path):
    if condition and replace_modules is not None:
        save_path = f"{save_path}/{step}_{name}.pt"
        save_lora_weight(model, save_path, replace_modules)


def handle_lora_save(
        use_unet_lora,
        use_text_lora,
        model,
        save_path,
        checkpoint_step,
        unet_target_modules,
        text_encoder_target_modules
):
    save_path = f"{save_path}/lora"
    os.makedirs(save_path, exist_ok=True)

    save_lora(
        model.unet,
        'unet',
        use_unet_lora,
        unet_target_modules,
        checkpoint_step,
        save_path,
    )
    save_lora(
        model.text_encoder,
        'text_encoder',
        use_text_lora,
        text_encoder_target_modules,
        checkpoint_step,
        save_path
    )

    train_patch_pipe(model, use_unet_lora, use_text_lora)




def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }



def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }

    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params


def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False


def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition:
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params


def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW


def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    return weight_dtype


def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)


def handle_cache_latents(
        should_cache,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        cached_latent_dir=None
):
    # Cache latents by storing them in VRAM.
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    # vae.to('cuda', dtype=torch.float16)
    device=Accelerator.device
    vae = vae.to(device)
    vae.enable_slicing()


def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break

                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled: unfrozen_params += 1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True
        print(f"{unfrozen_params} params have been unfrozen for training.")



def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    return latents



def sample_noise(latents, noise_strength, use_offset_noise):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None
    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise
    return noise_latents


def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1) \
        and validation_data.sample_preview


def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        text_encoder,
        vae,
        output_dir,
        use_unet_lora,
        use_text_lora,
        unet_target_replace_module=None,
        text_target_replace_module=None,
        is_checkpoint=False,
        controlnet=None,
        fusion_net=None,
        graph_model=None,
        step=None,
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype

    # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder))
    controlnet_out = copy.deepcopy(accelerator.unwrap_model(controlnet))

    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
        contronet=None,
    ).to(torch_dtype=torch.float32)
  
    pipeline.save_pretrained(save_path)

    if controlnet is not None:
        ctrl_path = os.path.join(output_dir, f"controlnet_{step}.pth")
        torch.save(controlnet_out.state_dict(), ctrl_path)
      
    if fusion_net is not None:
        fusion_path = os.path.join(output_dir, f"fusion_net_{step}.bin")
        accelerator.save(fusion_net.state_dict(), fusion_path)

    if graph_model is not None:
        graph_path = os.path.join(output_dir, f"graph_model_{step}.bin")
        accelerator.save(graph_model.state_dict(), graph_path)

    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()


def shape_change(bbx):
    B, F, N, C = bbx.shape
    bbx = bbx.view(B * F, N, -1)
    return bbx


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt


def main(
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        dataset_types: Tuple[str] = ('json'),
        validation_steps: int = 100,
        trainable_modules: Tuple[str] = ("attn1", "attn1.5", "attn2"),
        trainable_text_modules: Tuple[str] = ("all"),
        extra_unet_params=None,
        extra_text_encoder_params=None,
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        learning_rate: float = 5e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        text_encoder_gradient_checkpointing: bool = False,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        enable_torch_2_attn: bool = False,
        seed: Optional[int] = None,
        train_text_encoder: bool = False,
        use_offset_noise: bool = False,
        offset_noise_strength: float = 0.1,
        extend_dataset: bool = False,
        cache_latents: bool = False,
        cached_latent_dir=None,
        use_unet_lora: bool = False,
        use_text_lora: bool = False,
        unet_lora_modules: Tuple[str] = ["ResnetBlock2D"],
        text_encoder_lora_modules: Tuple[str] = ["CLIPEncoderLayer"],
        lora_rank: int = 16,
        lora_path: str = '',
        **kwargs
):
    *_, config = inspect.getargvalues(inspect.currentframe())
    global_step = 0
    first_epoch = 0

    def parse_args():
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument("--pretrained_diffusion_model_path", type=str,
                            default='/model/anonymity/StableDiffusion/stable-diffusion-v1-5',
                            help="Path to pretrained model or model identifier from huggingface.co/models.", )
        parser.add_argument('--data_dir', type=str, default='/data/anonymity/VisualGenome',
                            help='path to training dataset')
        parser.add_argument('--output_dir', type=str, default="/home/lotvs/Code/Z/VAE_BOX",
                            help='path to save checkpoint')
        parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")

        parser.add_argument('--dataloader_num_workers', type=int, default=8, help='num_workers')
        parser.add_argument('--dataloader_shuffle', type=bool, default=True, help='shuffle')
        parser.add_argument("--tracker_project_name", type=str, default="scene_vae",
                            help="The `project_name` passed to Accelerator", )
        parser.add_argument('--resolution', type=int, default=512, help='resolution')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument("--num_train_epochs", type=int, default=200)
        parser.add_argument("--max_train_steps", type=int, default=None,
                            help="Total number of training steps to perform.  If provided, overrides num_train_epochs.", )
        parser.add_argument("--checkpointing_steps", type=int, default=5000,
                            help="Save a checkpoint of the training state every X updates.")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")

        parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                            help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)", )
        parser.add_argument("--allow_tf32", action="store_true",
                            help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information")

        parser.add_argument("--learning_rate", type=float, default=1e-5,
                            help="Initial learning rate (after the potential warmup period) to use.")
        parser.add_argument("--lr_scheduler", type=str, default="constant",
                            help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
        parser.add_argument("--lr_warmup_steps", type=int, default=500,
                            help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        parser.add_argument("--adam_beta2", type=float, default=0.999,
                            help="The beta2 parameter for the Adam optimizer.")
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument("--vae_loss_weight", type=float, default=0.1, help="")
        parser.add_argument("--box_loss_weight", type=float, default=1, help="")
        parser.add_argument('--embedding_dim', type=int, default=64, help='embedding dim of GCN')
        parser.add_argument('--pretrained_model_path', type=str,
                            default='/media/lotvs/My Passport/OOD/data/OOD_leilei/model_pth_mask5/train_2023-11-06T00-10-32/checkpoint-30000')
        parser.add_argument('--video_frames', type=int, default=16, help='the number of video frames')
        parser.add_argument('--train_text_encoder', type=bool, default=True)
        parser.add_argument('--extra_text_encoder_params', type=bool, default=False)
        parser.add_argument('--use_text_lora', type=bool, default=True)
        parser.add_argument('--text_encoder_gradient_checkpointing', type=bool, default=True)
        parser.add_argument('--tlogging_dir', type=str, default='./logs')
        parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)))
        args = parser.parse_args()
        timestamp = time.strftime("%Y%m%d-%Hh%Mm%Ss", time.localtime())
        args.output_dir = os.path.join(args.output_dir, 'train', f'{args.tracker_project_name}-{timestamp}')
        return args

    argss = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    # if accelerator.is_main_process:
    #     output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet, sg_model, control_unet,lava_model = load_primary_models(
        pretrained_model_path, argss)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet,control_unet,lava_model])

    # # Enable xformers if available
    # handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    

    def collate_fn_dada_video_batch(batch):
        out = {}
        B = len(batch)
        T = 16  # 固定帧长
        # 视频帧
        out['rgb_video'] = [item['rgb_video'] for item in batch]  # [B, T, 3, H, W]
        out['depth_video'] = [item['depth_video'] for item in batch]  # [B, T, 3, H, W]
        out['mask_video'] = [item['mask_video'] for item in batch]  # [B, T, 3, H, W]
        out['prompt'] = [item['prompt'] for item in batch]
        out['a_prompt'] = [item['a_prompt'] for item in batch]
        out['lava_c_text'] = [item['lava_c_text'] for item in batch]
        out['lava_a_text'] = [item['lava_a_text'] for item in batch]
        out["save_name"]= [item['save_name'] for item in batch]
        out["idx"] = [item['idx'] for item in batch]
        out["aa_prompt"]= [item['aa_text'] for item in batch]

        # 合并图相关信息
        all_objs = []
        all_boxes = []
        all_triples = []
        all_labels = []
        all_angles = []
        all_obj_to_video = []
        all_triple_to_video = []
        all_obj_to_frame = [] 
        all_triple_to_frame = []  
        object_text_all = []
        relation_text_all = []
        obj_offset = 0
        for i, item in enumerate(batch):
            obj_tensor = item['object_id']  # [T, N_obj]
            box_tensor = item['boxes']  # [T, N_obj, 4]
            triple_tensor = item['relation_number']  # [T, N_rel, 3]
            angle_tensor = item['angles']  # [T, N_rel]
            obj_text = item['object_text']  # List[List[str]]
            rel_text = item['relation_text']
            label_tensor = item['gt_labels']

            for t in range(T):
                objs = obj_tensor[t]  # [N_obj]
                boxes = box_tensor[t]  # [N_obj, 4]
                triples = triple_tensor[t]  # [N_rel, 3]
                angles = angle_tensor[t]  # [N_rel]
                labels = label_tensor[t]
                n_obj = objs.shape[0]
                n_rel = triples.shape[0]
                all_labels.append(torch.tensor(labels))
                all_objs.append(objs)
                all_boxes.append(boxes)
             
                all_obj_to_video.append(torch.full((n_obj,), i, dtype=torch.long))
                all_obj_to_frame.append(torch.full((n_obj,), t, dtype=torch.long)) 
             
                if n_rel > 0:
                    triples = triples.clone()
                    triples[:, 0] += obj_offset
                    triples[:, 2] += obj_offset
                    all_triples.append(triples)
                    all_angles.append(angles)
                   
                    all_triple_to_video.append(torch.full((n_rel,), i, dtype=torch.long))
                    all_triple_to_frame.append(torch.full((n_rel,), t, dtype=torch.long))  
                obj_offset += n_obj
            # 文本 flatten
            object_text_all.extend(sum(obj_text, []))  # List[List[str]] → List[str]
            relation_text_all.extend(sum(rel_text, []))  # List[List[str]] → List[str]

        out['object_id'] = torch.cat(all_objs, dim=0)  # [N_obj_total]
        out['boxes'] = torch.cat(all_boxes, dim=0)  # [N_obj_total, 4]
        out['relation_number'] = torch.cat(all_triples, dim=0)  # [N_rel_total, 3]
        out['angles'] = torch.cat(all_angles, dim=0)  # [N_rel_total]
        out['obj_to_video'] = torch.cat(all_obj_to_video, dim=0)  # [N_obj_total]
        out['obj_to_frame'] = torch.cat(all_obj_to_frame, dim=0)
        out['triple_to_video'] = torch.cat(all_triple_to_video, dim=0)  # [N_rel_total]
        out['triple_to_frame'] = torch.cat(all_triple_to_frame, dim=0)
        out['object_text'] = object_text_all
        out['relation_text'] = relation_text_all
        out['labels'] = torch.cat(all_labels, dim=0)

        return out




    train_dataset = DADA2KS_Graph_Inference(root_path=r"./Test_Data/Test_Relation", interval=1,
                                  phase="val")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, collate_fn=collate_fn_dada_video_batch,
        drop_last=True, shuffle=False, num_workers=1)

    unet,text_encoder, train_dataloader, sg_model, control_unet,lava_model = accelerator.prepare(
        unet,
        text_encoder, train_dataloader, sg_model, control_unet,lava_model
    )
  
    unet_and_text_g_c(
        unet,
        text_encoder,
        gradient_checkpointing,
        text_encoder_gradient_checkpointing
    )
    # Enable VAE slicing to save memory.
    vae.enable_slicing()
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)
    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder,vae,sg_model]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    # num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Only show the progress bar once on each machine.

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    processor = VideoLlavaProcessor.from_pretrained(
        r"./Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf")
    processor.tokenizer.padding_side = "right" 

    for step, batch in enumerate(train_dataloader):
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
        if global_step % validation_steps == 0:
            if accelerator.is_main_process:
                with accelerator.autocast():
                    unet.eval()
                    text_encoder.eval()
                    control_unet.eval()
                    sg_model.eval()
                    unet_and_text_g_c(unet, text_encoder, False, False)
                    pipeline = TextToVideoSDPipeline.from_pretrained(
                        pretrained_model_path,
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet,
                        contronet=control_unet,
                    )
                    pipeline.contronet=control_unet

                diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                pipeline.scheduler = diffusion_scheduler
                # Check if we are training the text encoder
                text_trainable = (train_text_encoder or use_text_lora)
                device = accelerator.device
              
                video = torch.stack([torch.tensor(v) for v in batch['rgb_video']], dim=0).to(device=device,
                                                                                                     dtype=weight_dtype)
                origin_video=video.clone()
                depth_video = torch.stack([torch.tensor(v) for v in batch['depth_video']], dim=0).to(
                    device=device, dtype=weight_dtype)
                mask_video = torch.stack([torch.tensor(v) for v in batch['mask_video']], dim=0).to(
                    device=device, dtype=weight_dtype)
                # if not cache_latents:
                if cache_latents:
                    latents = tensor_to_vae_latent(video, vae)
                

                # Get video length
                video_length = latents.shape[2]

                cast_to_gpu_and_type([text_encoder], accelerator, torch.float32)
              
                if gradient_checkpointing or text_encoder_gradient_checkpointing:
                    unet.eval()
                    text_encoder.eval()
                    control_unet.eval()
                    sg_model.eval()
                selected_text = batch["prompt"]


                def encoding_graph_text(tokenizer, text_encoder, textss, device):
                    with torch.no_grad():
                        texts = []
                        for text in textss:
                            text = tokenizer(
                                text, max_length=tokenizer.model_max_length, padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                            ).input_ids[0].to(device)
                            encoder_text = text_encoder(text.unsqueeze(0))[1]
                            texts.append(encoder_text.squeeze(0))
                        texts = torch.stack(texts)
                        return texts

                def encoding_text(tokenizer, text_encoder, textss, device):
                    with torch.no_grad():
                        texts = []
                        for text in textss:
                            text = tokenizer(
                                text, max_length=tokenizer.model_max_length, padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                            ).input_ids[0].to(device)
                            encoder_text = text_encoder(text.unsqueeze(0))[0]
                            texts.append(encoder_text.squeeze(0))
                        texts = torch.stack(texts)
                        return texts

                object_id = batch["object_id"].to(device)
                object_text = batch["object_text"]
                relation_id = batch["relation_number"].to(device)
                relation_text = batch["relation_text"]
                boxes = batch["boxes"].to(device=device, dtype=torch.float32)
                angles = batch["angles"].to(device=device, dtype=torch.long)
                # c_prompt = batch["c_prompt"]
                a_prompt = [batch["prompt"][0].split(",")[0]]
                # ac_prompt = batch["ac_prompt"]
                save_name= batch["save_name"][0]
                idx=batch["idx"][0]
                index_str = f"{idx[0]}-{idx[1]}"

                obj_to_video = batch['obj_to_video'].to(device)
                triple_to_video = batch['triple_to_video'].to(device)
                obj_to_frame = batch['obj_to_frame'].to(device)
                gt_labels = batch['labels'].to(device)
                s_label, p_label, o_label = gt_labels.chunk(3, dim=1)
                g_s_label, g_p_label, g_o_label = [x.squeeze(1) for x in [s_label, p_label, o_label]]

                object_clip_emb = encoding_graph_text(tokenizer, text_encoder, object_text, device)
                relation_clip_emb = encoding_graph_text(tokenizer, text_encoder, relation_text, device)
                # c_prompt_clip_emb = encoding_text(tokenizer, text_encoder, c_prompt, device)
                a_prompt_clip_emb = encoding_text(tokenizer, text_encoder, a_prompt, device)
                # ac_prompt_clip_emb = encoding_text(tokenizer, text_encoder, ac_prompt, device)
                bsz = a_prompt_clip_emb.shape[0]

                object_clip_emb = encoding_graph_text(tokenizer, text_encoder, object_text, device)
                relation_clip_emb = encoding_graph_text(tokenizer, text_encoder, relation_text, device)
                # c_prompt_clip_emb = encoding_text(tokenizer, text_encoder, c_prompt, device)
                a_prompt_clip_emb = encoding_text(tokenizer, text_encoder, a_prompt, device)
                # ac_prompt_clip_emb = encoding_text(tokenizer, text_encoder, ac_prompt, device)
                prompt_clip_emb = a_prompt_clip_emb

                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,),
                                          device=a_prompt_clip_emb.device)

                timesteps = timesteps.long()

                task_type = random.choice(["predict", "backward"])

                # task_type = "predict"
                enhance_text, enhace_graph, graph_noise, mask = sg_model(
                    object_id, object_clip_emb, boxes, relation_id, relation_clip_emb, angles, obj_to_video,
                    triple_to_video,
                    video, prompt_clip_emb, obj_to_frame, timesteps, task_type)
                    
                if task_type=="predict":
                    videos = pipeline(
                        prompt=enhance_text,
                        negative_prompt=None,
                        num_frames=16,
                        height=224,
                        width=224,
                        num_inference_steps=25,
                        guidance_scale=9.0,
                        output_type="pt",
                        control_depth_videos=depth_video,
                        control_mask_videos=mask_video,
                        latents=latents,
                        task_type=task_type,
                    ).frames
                
                elif task_type=="backward":
                    videos = pipeline(
                        prompt=enhance_text,
                        negative_prompt=None,
                        num_frames=16,
                        height=224,
                        width=224,
                        num_inference_steps=25,
                        guidance_scale=9.0,
                        output_type="pt",
                        control_depth_videos=depth_video,
                        control_mask_videos=mask_video,
                        latents=latents,
                        task_type=task_type
                    ).frames
                
                save_folder=r"./output"
                save_folder_r=os.path.join(save_folder,f"{save_name}")
                if not os.path.exists(save_folder_r):
                    os.makedirs(save_folder_r)
                for video in videos:
                    video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)
                    video = video.byte().cpu().numpy()
                    for f, frame in enumerate(video):
                        frame_path = os.path.join(save_folder_r,f"frame_{f + 1}.jpg")
                        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                save_folder = r"./output"
                save_folder_r = os.path.join(save_folder, f"{save_name}")
                if not os.path.exists(save_folder_r):
                    os.makedirs(save_folder_r)

                def preprocess_video_tensor(video_tensor):
                   
                    if video_tensor.shape[0] == 3: 
                        video_np = rearrange(video_tensor, "c f h w -> f h w c")
                    elif video_tensor.shape[1] == 3: 
                        video_np = rearrange(video_tensor, "f c h w -> f h w c")
                    else:
                        raise ValueError("Unrecognized video tensor shape")

                    video_np = video_np.clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
                    return video_np

                video_generated = preprocess_video_tensor(videos.squeeze(0)) 
                video_original = preprocess_video_tensor(origin_video.squeeze(0))  

              
                min_frames = min(len(video_generated), len(video_original))
                video_generated = video_generated[:min_frames]
                video_original = video_original[:min_frames]

                gif_frames = []
                for f in range(min_frames):
                    left = video_original[f]
                    right = video_generated[f]
                    concat =np.concatenate((left, right), axis=1) 
                    gif_frames.append(concat)

              
                gif_path = os.path.join(save_folder_r, "comparison.gif")
                imageio.mimsave(gif_path, gif_frames, fps=8,loop=0)  
                print(f"Saved comparison GIF to: {gif_path}")
                
               
                # accident_types = [
                #     "ego-car hits a crossing pedestrian",
                #     "ego-car hits a pedestrian",
                #     "ego-car hits a crossing cyclist",
                #     "ego-car hits a cyclist",
                #     "ego-car hits a crossing motorbike",
                #     "ego-car hits a motorbike",
                #     "ego-car hits a crossing truck",
                #     "ego-car hits a truck",
                #     "ego-car is overtaken by a truck",
                #     "ego-car hits a crossing car",
                #     "ego-car hits a car",
                #     "ego-car is overtaken by a car",
                #     "ego-car hits large roadblocks",
                #     "ego-car hits a curb",
                #     "ego-car hits small roadblocks road pothholes",
                #     "ego-car hits trees",
                #     "ego-car hits telegraphpoles",
                #     "ego-car hits other road facilties",
                #     "motorbike hits large roadblocks",
                #     "truck hits large roadblocks",
                #     "car hits large roadblocks",
                #     "motorbike hits a curb",
                #     "truck hits a curb",
                #     "car hits a curb",
                #     "truck hits small roadblocks road pothholes",
                #     "car hits small roadblocks road pothholes",
                #     "truck hits trees",
                #     "car hits trees",
                #     "truck hits telegraphpoles",
                #     "car hits telegraphpoles",
                #     "motorbike hits other road facilities",
                #     "truck hits other road facilities",
                #     "car hits other road facilities",
                #     "motorbike falls down",
                #     "motorbike hits motorbike",
                #     "truck is out of control",
                #     "truck hits truck",
                #     "truck has failure of components",
                #     "car is out of control",
                #     "car hits car",
                #     "car has failure of components",
                #     "truck hits motorbike",
                #     "motorbike scratches truck",
                #     "truck hits car",
                #     "car scratches truck",
                #     "car hits motorbike",
                #     "motorbike scratches car",
                #     "motorbike hits pedestrian",
                #     "motorbike hits cyclist",
                #     "truck hits pedestrian",
                #     "truck hits cyclist",
                #     "car hits pedestrian",
                #     "car hits cyclist",
                #     "pedestrian falls down",
                #     "cyclist falls down",
                #     "cyclist hits cyclist",
                #     "ego-car is out of control",
                #     "ego-car has failure of components"

                if task_type == "predict":
                     user_part ="USER:<video> What is the type of this predicted accident video? ASSISTANT:"
                else:
                    user_part = (
                        "USER: <video> What specific visible reason caused this retrospective accident "
                        "video? Answer with one short sentence only, based only on what is visible in the video. "
                        "ASSISTANT:"
                    )

                lava_video = rearrange(videos, "b c f h w -> b f c h w")
                lava_video=lava_video[:, 1:16:2, :, :, :]
                # inputs = processor(text=user_part, videos=lava_video, return_tensors="pt").to(device)
                inputs = processor(text=user_part, videos=lava_video, padding=True, truncation=True,
                                       return_tensors="pt").to(device)
                prompt_len = inputs["input_ids"].shape[-1]
                sample_seed = None
                if task_type == "predict":  
                    generation_kwargs=None
                    prompt_len = inputs["input_ids"].shape[-1]
                    # Generate
                    # Generate token IDs and decode back into text
                    generate_ids = lava_model.generate(**inputs, max_length=prompt_len+80)
                    # generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    generated_text=processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                elif task_type == "backward":
                    sample_seed = 17
                    for ch in f"{save_name}/{index_str}":
                        sample_seed = (sample_seed * 131 + ord(ch)) % 2147483647
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)
                    generation_kwargs = {
                        "max_new_tokens": 80,
                        "do_sample": True,
                        "temperature": 0.6,
                        "top_p": 0.85,
                        "top_k": 20,
                        "repetition_penalty": 1.1,
                        "no_repeat_ngram_size": 3,
                        "num_return_sequences": 1,
                        "pad_token_id": processor.tokenizer.pad_token_id,
                        "eos_token_id": processor.tokenizer.eos_token_id,
                    }
                    generate_ids = lava_model.generate(**inputs, **generation_kwargs)
                    answer_ids = generate_ids[:, prompt_len:]
                    raw_generated_text = processor.batch_decode(
                        answer_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                
                
                    generated_text = finalize_reason_response(raw_generated_text)
                
                json_save_path = "./output/LLM_results.json"

                
                if os.path.exists(json_save_path):
                    with open(json_save_path, "r") as f:
                        result_dict = json.load(f)
                else:
                    result_dict = {}

                result_dict[f"{save_name} / {index_str}"] = {
                    "question": user_part,
                    "sampling_seed": sample_seed,
                    "generation_kwargs": generation_kwargs,
                    "response": generated_text,
                }
                with open(json_save_path, "w") as f:
                    json.dump(result_dict, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/lora_training_config.yaml")
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))
