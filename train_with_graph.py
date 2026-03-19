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

from grapth_my_dataset import DADA2KS_Graph_Train
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import diffusers
import transformers
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder

import datetime
from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor,AutoConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModel
import torch
from torch.utils.data import Dataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.profilers import SimpleProfiler
from einops import rearrange, repeat
import clip
from models.control_my_mynet import ControlUNet3DConditionModel
from model.my_box_vae_mode import SceneVAEModel
from utils.lora import (
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    save_lora_weight,
    train_patch_pipe,
    monkeypatch_or_replace_lora,
    monkeypatch_or_replace_lora_extended
)
already_printed_trainables = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.

check_min_version("0.10.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_diffusion_model_path", type=str,
                        default='/model/anonymity/StableDiffusion/stable-diffusion-v1-5',
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    parser.add_argument('--data_dir', type=str, default='',
                        help='path to training dataset')
    parser.add_argument('--output_dir', type=str, default="/home/lotvs/Code/Z/VAE_BOX", help='path to save checkpoint')
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
                        default=r'/media/lotvs/WD_4T/checkpoint-30000')
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


def load_primary_models(pretrained_model_path,args):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    control_unet=ControlUNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    num_objs = 7
    num_rels = 49
    clip_model, preprocess = clip.load("./ViT-B/32")
    clip_model.eval().requires_grad_(False)
    pretrain = torch.load(r"",
                    map_location=torch.device('cpu'))
    sg_model = SceneVAEModel(args, num_objs, num_rels, clip_model)
    sg_model.load_state_dict(pretrain, strict=False)
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

    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ['multi_modal_projector', 'vision_model']
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    config = {"max_epochs": 5,
              "val_check_interval": 0.2,  # how often we want to validate during an epoch,
              "check_val_every_n_epoch": 1,
              "gradient_clip_val": 1.0,
              "accumulate_grad_batches": 8,
              "lr": 1e-3,
              "batch_size": 1,
              "num_nodes": 1,
              "warmup_steps": 50,
              }

    return noise_scheduler, tokenizer, text_encoder, vae, unet,sg_model,control_unet,model



def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    # text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)
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
    # try:
    #     is_torch_2 = hasattr(F, 'scaled_dot_product_attention')

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
    device = torch.device("cuda",2)
    # vae.to('cuda', dtype=torch.float16)
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
    # t = rearrange(t, "b f c h w -> b f c h w")
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
        contronet=controlnet_out,
    ).to(torch_dtype=torch.float32)


    # handle_lora_save(
    #     use_unet_lora,
    #     use_text_lora,
    #     pipeline,
    #     output_dir,
    #     global_step,
    #     unet_target_replace_module,
    #     text_target_replace_module
    # )
    pipeline.save_pretrained(save_path)

    if controlnet is not None:
        ctrl_path = os.path.join(output_dir, f"controlnet_{step}.pth")
        torch.save(controlnet_out.state_dict(), ctrl_path)
        # 保存其他自定义模块
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

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    processor = VideoLlavaProcessor.from_pretrained(
        r"./Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf")
    processor.tokenizer.padding_side = "right"  # during training, one always uses padding on the right

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    noise_scheduler, tokenizer, text_encoder, vae, unet,sg_model,control_unet,lava_model= load_primary_models(
        pretrained_model_path,argss)

    # Freeze any necessary models
    freeze_models([vae, text_encoder,unet,control_unet,sg_model])

    # for name, param in unet.named_parameters():
    #     # print(name)
    #     if '.temp' in name:
    #         param.requires_grad = True

    for name, param in control_unet.named_parameters():
        # print(name)
        if '.temp' in name:
            param.requires_grad = True

    for name, param in sg_model.named_parameters():
        if name.startswith('decoder') or name.startswith('gconv_decoder') \
                or name.startswith('obj_embeddings_decoder') or name.startswith('rel_embeddings_decoder') \
                or name.startswith('mlp_box'):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # params = list(text_encoder.parameters())
    params += list(control_unet.parameters())
    # params += list(unet.parameters())
    params += list(sg_model.parameters())
    params += list(lava_model.parameters())

    optimizer= torch.optim.AdamW(params, lr=learning_rate,betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    def collate_fn_dada_video_batch(batch):
        out = {}
        B = len(batch)
        T = 16  
        out['rgb_video'] = [item['rgb_video'] for item in batch]  # [B, T, 3, H, W]
        out['depth_video'] = [item['depth_video'] for item in batch]  # [B, T, 3, H, W]
        out['mask_video'] = [item['mask_video'] for item in batch]  # [B, T, 3, H, W]
        out['n_rgb_video'] = [item['n_rgb_video'] for item in batch]  # [B, T, 3, H, W]
        out['c_prompt'] = [item['c_prompt'] for item in batch]
        out['a_prompt'] = [item['a_prompt'] for item in batch]
        out['ac_prompt'] = [item['ac_prompt'] for item in batch]
        out['lava_a_text'] = [item['lava_a_text'] for item in batch]
        out['lava_ac_text'] = [item['lava_ac_text'] for item in batch]
        out['lava_p_text'] = [item['lava_p_text'] for item in batch]
        out['lava_r_text'] = [item['lava_r_text'] for item in batch]
        out['lava_re_text']=[item['lava_re_text'] for item in batch]
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

    train_dataset = DADA2KS_Graph_Train(root_path=r"./Train_relation_json", interval=1,
                                  phase="train")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, collate_fn=collate_fn_dada_video_batch,
        drop_last=True, shuffle=True, num_workers=1)


    unet, optimizer, lr_scheduler, text_encoder, train_dataloader,sg_model,control_unet,lava_model= accelerator.prepare(
        unet,
        optimizer,
        lr_scheduler,
        text_encoder, train_dataloader,sg_model,control_unet,lava_model
    )
    # Use Gradient Checkpointing if enabled.

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

   
    def compute_infonce_loss_from_video_text(
            lava_v_f,  # shape: (B*F, 257, D)
            lava_t_f,  # shape: (B, L, D)
            B, frames,  # batch size, frame count
            con_labels,  # shape: (B,) - 0 for fact, 1 for counterfactual
            pad_token_id=None,
            input_ids=None,
            temperature=0.07,
            use_mean_text_pooling=False,
    ):
        D = lava_v_f.size(-1)
        lava_v_f = lava_v_f.view(B, frames, 257, D)  # (B, F, 257, D)
        video_cls = lava_v_f[:, :, 0, :]  # (B, F, D)
        video_feat = video_cls.mean(dim=1)  # (B, D)

        # === Text pooling ===
        if use_mean_text_pooling:
            assert input_ids is not None and pad_token_id is not None
            mask = (input_ids != pad_token_id).float()  # (B, L)
            text_feat = (lava_t_f * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)  # (B, D)
        else:
            text_feat = lava_t_f[:, 0, :]  # (B, D)

        # === Normalize ===
        video_feat = F.normalize(video_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        sim_matrix = video_feat @ text_feat.T / temperature  # (B, B)

     
        labels = con_labels.view(-1, 1)  # (B, 1)
        label_eq = (labels == labels.T)  # (B, B) 
        label_eq.fill_diagonal_(False)  
        label_neq = ~label_eq  

        contrastive_losses = []
        for i in range(B):
            pos_idx = label_eq[i].nonzero(as_tuple=True)[0]
            neg_idx = label_neq[i].nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            pos_sims = sim_matrix[i, pos_idx]
            neg_sims = sim_matrix[i, neg_idx]

            logits = torch.cat([pos_sims, neg_sims], dim=0)
            labels_i = torch.zeros_like(logits)
            labels_i[:len(pos_sims)] = 1.0 

            loss_i = F.binary_cross_entropy_with_logits(logits, labels_i)
            contrastive_losses.append(loss_i)

        if len(contrastive_losses) == 0:
            return torch.tensor(0.0, device=video_feat.device)

        return torch.stack(contrastive_losses).mean()

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

    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    def finetune_unet(batch,task_type,train_encoder=False):
        # Check if we are training the text encoder
        text_trainable = (train_text_encoder or use_text_lora)
        if global_step == 0:
            already_printed_trainables = False
            unet.train()
            control_unet.train()
            sg_model.train()
            lava_model.train()
        device=accelerator.device
        object_id = batch["object_id"].to(device)
        object_text = batch["object_text"]
        relation_id = batch["relation_number"].to(device)
        relation_text = batch["relation_text"]
        boxes = batch["boxes"].to(device=device, dtype=torch.float32)
        angles = batch["angles"].to(device=device, dtype=torch.long)
        c_prompt = batch["c_prompt"]
        a_prompt = batch["a_prompt"]
        ac_prompt = batch["ac_prompt"]
        lava_a_text=batch["lava_a_text"]
        lava_ac_text=batch['lava_ac_text']
        lava_p_text = batch['lava_p_text']
        lava_r_text = batch['lava_r_text']
        lava_re_text=batch["lava_re_text"]
        obj_to_video = batch['obj_to_video'].to(device)
        triple_to_video = batch['triple_to_video'].to(device)
        obj_to_frame = batch['obj_to_frame'].to(device)
        gt_labels = batch['labels'].to(device)
        s_label, p_label, o_label = gt_labels.chunk(3, dim=1)
        g_s_label, g_p_label, g_o_label = [x.squeeze(1) for x in [s_label, p_label, o_label]]
        trainable_text_modules = ("all"),
      
        object_clip_emb = encoding_graph_text(tokenizer, text_encoder, object_text, device)
        relation_clip_emb = encoding_graph_text(tokenizer, text_encoder, relation_text, device)
        c_prompt_clip_emb = encoding_text(tokenizer, text_encoder, c_prompt, device)
        a_prompt_clip_emb = encoding_text(tokenizer, text_encoder, a_prompt, device)
        ac_prompt_clip_emb = encoding_text(tokenizer, text_encoder, ac_prompt, device)
        bsz = a_prompt_clip_emb.shape[0]
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=c_prompt_clip_emb.device)
        timesteps = timesteps.long()
        video= torch.stack([torch.tensor(v) for v in batch['rgb_video']], dim=0).to(device=device, dtype=weight_dtype)
        depth_video = torch.stack([torch.tensor(v) for v in batch['depth_video']], dim=0).to(device=device, dtype=weight_dtype)
        mask_video = torch.stack([torch.tensor(v) for v in batch['mask_video']], dim=0).to(device=device, dtype=weight_dtype)
        n_video = torch.stack([torch.tensor(v) for v in batch['n_rgb_video']], dim=0).to(device=device, dtype=weight_dtype)
        if cache_latents:
            latents = tensor_to_vae_latent(video, vae)
            # Get video length
        video_length = latents.shape[2]
        # Sample noise that we'll add to the latents
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        num_frames = 16
        cond_frames = 6  # = 6
        pivot_frame = cond_frames - 1 
        # task_type = random.choice(["predict", "backward"])
        if task_type == "predict":
            noisy_latents[:, :, 0:6, :, :] = latents[:, :, 0:6, :, :]
            for f in range(num_frames):
                if f < cond_frames:
                    base_ratio = 0.3
                    source_cond = latents[:, :, f:f + 1, :, :]
                else:
                    decay_ratio = 1 - (f - cond_frames) / (num_frames - cond_frames)
                    base_ratio = 0.01 * decay_ratio
                    source_cond = latents[:, :, pivot_frame:pivot_frame + 1, :, :]
                noisy_latents[:, :, f:f + 1, :, :] = (
                        0.1 * source_cond +
                        (1 - base_ratio) * noisy_latents[:, :, f:f + 1, :, :]
                )
            noisy_latents[:, :, 0:6, :, :] = latents[:, :, 0:6, :, :]
            prompt_clip_emb=a_prompt_clip_emb
            counterfactual_video = n_video[:, 1:16:2, :, :, :]
            lava_video = video[:, 1:16:2, :, :, :]
            B = lava_video.shape[0]
            video_inputs = torch.cat([lava_video, counterfactual_video], dim=0).to(torch.uint8).cpu().numpy()
            con_labels = torch.cat([torch.zeros(B), torch.ones(B)], dim=0).long()  # (2B,)
            lava_text = lava_a_text + lava_r_text
            lava_batch = processor(text=lava_text, videos=video_inputs, padding=True, truncation=True,
                                   return_tensors="pt")
            labels =lava_batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            lava_batch["labels"] = labels
            input_ids =lava_batch["input_ids"].to(device)
            attention_mask = lava_batch["attention_mask"].to(device)
            pixel_values_videos =lava_batch["pixel_values_videos"].to(device)
            labels = lava_batch["labels"]
        elif task_type == "backward":
            noisy_latents[:, :, -6:] = latents[:, :, -6:]
            pivot_frame = abs(-6)  
            cond_frames = abs(-6)
            for f in range(num_frames):
                if f >= num_frames - cond_frames:
                    base_ratio = 0.3
                    source_cond = latents[:, :, f - (num_frames - cond_frames):f - (num_frames - cond_frames) + 1,
                                  :,
                                  :]
                else:
                    decay_ratio = (num_frames - cond_frames - f) / (num_frames - cond_frames)
                    base_ratio = 0.01 * decay_ratio
                    source_cond = latents[:, :, pivot_frame - 1:pivot_frame, :, :]

                noisy_latents[:, :, f:f + 1, :, :] = (
                        0.1 * source_cond +
                        (1 - base_ratio) * noisy_latents [:, :, f:f + 1, :, :]
                )

            noisy_latents[:, :, 10:16, :, :] = latents[:, :, -6:]
            prompt_clip_emb = c_prompt_clip_emb
            lava_video = video[:, 1:16:2, :, :, :]
            B=lava_video.shape[0]
            counterfactual_video = lava_video.flip(dims=[1])
            video_inputs = torch.cat([lava_video, counterfactual_video], dim=0).to(torch.uint8).cpu().numpy()
            con_labels = torch.cat([torch.zeros(B), torch.ones(B)], dim=0).long()  # (2B,)
            lava_text=lava_re_text+lava_p_text
            lava_batch = processor(text=lava_text, videos=video_inputs, padding=True, truncation=True,
                                   return_tensors="pt")
            labels = lava_batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            lava_batch["labels"] = labels
            input_ids = lava_batch["input_ids"].to(device)
            attention_mask = lava_batch["attention_mask"].to(device)
            pixel_values_videos = lava_batch["pixel_values_videos"].to(device)
            labels = lava_batch["labels"].to(device)

        lava_outputs,lava_v_f,lava_t_f = lava_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels,
        )
        # lava_loss = lava_outputs.loss

        enhance_text, enhace_graph,graph_noise,mask = sg_model(
                object_id, object_clip_emb, boxes, relation_id, relation_clip_emb, angles, obj_to_video, triple_to_video,
                video, prompt_clip_emb, obj_to_frame, timesteps,task_type)

        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        losses = []
        should_truncate_video = (video_length > 1 and text_trainable)

        # We detach the encoder hidden states for the first pass (video frames > 1)
        # Then we make a clone of the initial state to ensure we can train it in the loop.
        detached_encoder_state = enhance_text.clone().detach()
        trainable_encoder_state = enhance_text.clone()

    
        should_detach = noisy_latents.shape[2] > 1
        encoder_hidden_states = (
            detached_encoder_state if should_detach else trainable_encoder_state
        )

        down_block_res_samples_dict, mid_block_res_sample_dict = control_unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_down_cond=depth_video,controlnet_up_cond=mask_video
        )
        model_pred= unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, down_block_additional_residuals=down_block_res_samples_dict,
        mid_block_additional_residual=mid_block_res_sample_dict)
        if task_type == "predict":
            con_loss = compute_infonce_loss_from_video_text(
                lava_v_f=lava_v_f,
                lava_t_f=lava_t_f,
                B=2, frames=8,
                con_labels=con_labels,
                temperature=0.07,
                pad_token_id=processor.tokenizer.pad_token_id,
                input_ids=input_ids,
                use_mean_text_pooling=True) 

            loss1 = F.mse_loss(model_pred[:, :, 6:, :, :].float(), target[:, :, 6:, :, :].float(),
                               reduction="mean") + 0.3*F.mse_loss(enhace_graph[mask].float(), graph_noise[mask].float(),
                                                              reduction="mean")+ con_loss

        elif task_type == "backward":
           
            con_loss = compute_infonce_loss_from_video_text(
                lava_v_f=lava_v_f,
                lava_t_f=lava_t_f,
                B=2, frames=8,
                con_labels=con_labels,
                temperature=0.07,
                pad_token_id=processor.tokenizer.pad_token_id,
                input_ids=input_ids,
                use_mean_text_pooling=True, 
            )

            loss1 = F.mse_loss(model_pred[:, :, :-6:, :].float(), target[:, :, :-6, :, :].float(),
                               reduction="mean") + 0.3*F.mse_loss(enhace_graph[mask].float(), graph_noise[mask].float(),
                                                              reduction="mean")+con_loss


        losses.append(loss1)
        loss = losses[0] if len(losses) == 1 else losses[0] + losses[1]
        return loss, latents

    task_types = ["predict", "backward"]
    task_step = 0

    for epoch in range(first_epoch, 200):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet) ,accelerator.accumulate(text_encoder),accelerator.accumulate(control_unet),accelerator.accumulate(sg_model,accelerator.accumulate(lava_model)):
                with accelerator.autocast():

                    task_type = task_types[task_step % 3]
                    task_step += 1

                    loss, latents = finetune_unet(batch,task_type,train_encoder=train_text_encoder)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
              
                try:
                    accelerator.backward(loss)
                    params_to_clip = (
                            list(control_unet.parameters()) +
                            # list(unet.parameters()) +
                            # list(text_encoder.parameters()) +
                            list(sg_model.parameters())+
                            list(lava_model.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}")
                    continue
            print(global_step,"loss:",loss)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    # save_pipe(
                    #     pretrained_model_path,
                    #     global_step,
                    #     accelerator,
                    #     unet,
                    #     text_encoder,
                    #     vae,
                    #     output_dir,
                    #     use_unet_lora,
                    #     use_text_lora,
                    #     unet_lora_modules,
                    #     text_encoder_lora_modules,
                    #     is_checkpoint=True
                    # )
                    output_dir = f" "
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    lava_model.save_pretrained(output_dir)
                    print(f"LoRA adapter weights saved to {output_dir}")
                    # model = PeftModel.from_pretrained(model, output_dir)
                    save_pipe(
                        pretrained_model_path,
                        global_step,
                        accelerator,
                        unet,
                        text_encoder,
                        vae,
                        output_dir,
                        use_unet_lora,
                        use_text_lora,
                        unet_lora_modules,
                        text_encoder_lora_modules,
                        is_checkpoint=True,
                        controlnet=control_unet,  
                        fusion_net=None, 
                        graph_model=sg_model, 
                        step=global_step,
                    )
                # del pipeline
                torch.cuda.empty_cache()
                unet_and_text_g_c(
                    unet,
                    text_encoder,
                    gradient_checkpointing,
                    text_encoder_gradient_checkpointing
                )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/lora_training_config.yaml")
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))