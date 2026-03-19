# Copyright 2023 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
# Copyright 2023 The ModelScope Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from models.attentions import TransformerTemporalModel
# from deconv import Decoder2D
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
    transformer_g_c
)

import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from models.unet_3d_condition import UNet3DConditionModel
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config

from causl_block import P_QA

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
            self,
            conditioning_embedding_channels: int,
            conditioning_channels: int = 1,
            block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

        # self.conv_out = (
        #     nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        # )
        #


    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


control_stage_config = {
    "target": "cldm.cldm.ControlNet",
    "params": {
        "image_size": 28,  # unused
        "in_channels": 4,
        "hint_channels": 3,
        "model_channels": 320,
        "attention_resolutions": [4, 2, 1],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4, 4],
        "num_heads": 8,
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "context_dim": 1024,
        "use_checkpoint": True,
        "legacy": False
    }
}

@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor


class ControlUNet3DConditionModel(ModelMixin, ConfigMixin):
    r"""
    UNet3DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            conditioning_channels: int = 1,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",
            ),
            up_block_types: Tuple[str] = (
            "UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1024,
            attention_head_dim: Union[int, Tuple[int]] = 64,
            conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.sample_size = sample_size
        self.gradient_checkpointing = True
        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=attention_head_dim,
            in_channels=block_out_channels[0],
            num_layers=1,
        )

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        # control net conditioning embedding
        self.controlnet_cond_embedding1 = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        # class embedding
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # self.controlnet_up_blocks = nn.ModuleList([])
        # if isinstance(attention_head_dim, int):
        #     attention_head_dim = (attention_head_dim,) * len(up_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                # controlnet_block =controlnet_block
                self.controlnet_down_blocks.append(controlnet_block)
            if i==0:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            # for _ in range(layers_per_block):
            #     controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
            #     controlnet_block = zero_module(controlnet_block)
            #     # controlnet_block =controlnet_block
            #     self.controlnet_up_blocks.append(controlnet_block)
            # if i==0:
            #     controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
            #     controlnet_block = zero_module(controlnet_block)
            #     self.controlnet_up_blocks.append(controlnet_block)
            # if not is_final_block:
            #     controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
            #     controlnet_block = zero_module(controlnet_block)
            #     self.controlnet_up_blocks.append(controlnet_block)

        mid_block_channel = block_out_channels[-1]
        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block


        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=False,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

      
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

    @classmethod
    def from_unet(
            cls,
            unet: UNet3DConditionModel,
            controlnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
            load_weights_from_unet: bool = True,
            conditioning_channels: int = 1,
            guess_mode: bool = False,
            return_dict: bool = True,
            skip_conv_in: bool = False,  ### this is newly added
            skip_time_emb: bool = False,  ### this is newly added
    ):
        r"""
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            if hasattr(controlnet, "add_embedding"):
                controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value
        self.mid_block.gradient_checkpointing = value
        for module in self.down_blocks + self.up_blocks:
            if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
                module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            controlnet_down_cond: torch.FloatTensor,
            controlnet_up_cond: torch.FloatTensor,
            conditioning_scale: float = 1.0,
            # bbx: torch.Tensor,
            # fusion: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            guess_mode: bool = False,
            skip_conv_in: bool = False,  ### this is newly added
            skip_time_emb: bool = False,  ### this is newly added
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, num_frames, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet3DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.

        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep

        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        num_frames = sample.shape[2]
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        # 2. pre-process
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])

        sample = self.conv_in(sample)

        # im_q_outs = list()
        # un_im_q_outs = list()
        # un_im_do_outs = list()
        # if self.gradient_checkpointing:
        # video=fusion[0]
        # option=fusion[1]
        # non_option=fusion[2]
        # r_option = torch.cat([option, non_option], dim=1)
        # B, N, L, C = r_option.shape
        # r_option = r_option.view(B, N * L, C)
        # B_n, N_n, L_n, C_n = non_option.shape
        # non_option = non_option.view(B_n,N_n * L_n, C_n)
        # question=fusion[3]
        # map=fusion[4]
        # answer=fusion[5]
        # video=None
        # map=None
        # if self.gradient_checkpointing:
        #     sample,im_q_out, un_im_q_out, un_im_do_out = transformer_g_c(self.transformer_in, sample, num_frames, bbx,video,map)
        # else:
        #     sample,im_q_out, un_im_q_out, un_im_do_out  = self.transformer_in(sample, num_frames=num_frames, bbx=bbx,fusion=video,map=map)

        # zero_module(
        #     conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))


        if self.gradient_checkpointing:
            sample = transformer_g_c(self.transformer_in, sample, num_frames,bbx=None)
        else:
            sample = self.transformer_in(sample, num_frames=num_frames,bbx=None)
        b, f, c, h, w = controlnet_down_cond.shape
        controlnet_down_cond=controlnet_down_cond.reshape(b*f,c, h, w)
        controlnet_up_cond=controlnet_up_cond.reshape(b*f,c, h, w)

        controlnet_down_cond = self.controlnet_cond_embedding(controlnet_down_cond)
        controlnet_down_cond1= self.controlnet_cond_embedding1(controlnet_up_cond)
        sample = sample + controlnet_down_cond + controlnet_down_cond1


        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs, )
               
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples
        up_down_block_res_samples=down_block_res_samples
        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,

            )
           
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
            # print("mid_block:", sample.shape)
        # # # # 5. up
        # up_block_res_samples = (sample,)
        # for i, upsample_block in enumerate(self.up_blocks):
        #     is_final_block = i == len(self.up_blocks) - 1
        #
        #     res_samples = up_down_block_res_samples[-len(upsample_block.resnets):]
        #     up_down_block_res_samples =up_down_block_res_samples[: -len(upsample_block.resnets)]
        #
        #     # if we have not reached the final block and need to forward the
        #     # upsample size, we do it here
        #     if not is_final_block and forward_upsample_size:
        #         upsample_size = up_down_block_res_samples[-1].shape[2:]
        #
        #
        #     if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
        #         sample = upsample_block(
        #             hidden_states=sample,
        #             temb=emb,
        #             res_hidden_states_tuple=res_samples,
        #             encoder_hidden_states=encoder_hidden_states,
        #             upsample_size=upsample_size,
        #             attention_mask=attention_mask,
        #             num_frames=num_frames,
        #             cross_attention_kwargs=cross_attention_kwargs,
        # #
        #         )
        #
        #     else:
        #         sample = upsample_block(
        #             hidden_states=sample,
        #             temb=emb,
        #             res_hidden_states_tuple=res_samples,
        #             upsample_size=upsample_size,
        #             num_frames=num_frames,
        #         )
        #     up_block_res_samples += (sample,)
        #
        # controlnet_up_block_res_samples = ()
        #
        # for up_block_res_sample, controlnet_block in zip(up_block_res_samples, self.controlnet_up_blocks):
        #     up_block_res_sample = controlnet_block(up_block_res_sample)
        #     controlnet_up_block_res_samples = controlnet_up_block_res_samples + (up_block_res_sample,)
        #
        # up_block_res_samples = controlnet_up_block_res_samples

        #     if down_block_additional_residuals is not None:
        #         new_up_block_res_samples = ()
        #
        #         for up_block_res_sample, down_block_additional_residual in zip(
        #                 down_block_res_samples, down_block_additional_residuals
        #         ):
        #             down_block_res_sample = down_block_res_sample + down_block_additional_residual
        #             new_down_block_res_samples += (down_block_res_sample,)
        #
        #         down_block_res_samples = new_down_block_res_samples
        #         # 5. Control net blocks
        #
        #     controlnet_up_block_res_samples = ()
        #     for up_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_up_blocks):
        #         up_block_res_sample = controlnet_block(up_block_res_sample)
        #         controlnet_up_block_res_samples = controlnet_up_block_res_samples + (up_block_res_sample,)
        #         # print("up_block:", sample.shape)
        #     # post-process_qa
        # # out= self.P_QA(im_q_out, un_im_q_out, un_im_do_out, question, r_option, non_option,answer)
        #
        # # post-pross_yd
        #
        # # maps=self.decoder_map(im_q_out)
        #
        # # 6. post-process
        # if self.conv_norm_out:
        #     sample = self.conv_norm_out(sample)
        #     sample = self.conv_act(sample)
        #
        # sample = self.conv_out(sample)
        #
        # # reshape to (batch, channel, framerate, width, height)
        # sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        # # # 6. scaling
        # if guess_mode:
        #     scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
        #     scales = scales * conditioning_scale
        #     down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
        #     mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        # else:
        #     down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        #     mid_block_res_sample = mid_block_res_sample * conditioning_scale
        #
        # if self.config.global_pool_conditions:
        #     down_block_res_samples = [
        #         torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
        #     ]
        #     mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        # if not return_dict:
        return (down_block_res_samples, mid_block_res_sample)

        # return ControlNetOutput(
        #     down_block_res_samples= down_block_res_samples , mid_block_res_sample=mid_block_res_sample
        # )

        # return sample

    def zero_module(module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module
    @classmethod
    def clone(self):
        """
        Create a clone of the current instance.

        Returns:
            UNet3DConditionModel: A new instance that is a clone of the current instance.
        """
        # Implement cloning logic here
        # You need to copy the attributes of the current instance to the new instance
        # This might involve creating new instances of any mutable objects and copying their content
        cloned_instance = UNet3DConditionModel()
        # Copy attributes from self to cloned_instance
        # Example:
        # cloned_instance.some_attribute = self.some_attribute
        return cloned_instance

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")
        for k, v in model.state_dict().items():
            if 'controlnet_cond_embedding.' in k:
                state_dict.update({k: v})

            if 'controlnet_cond_embedding1.' in k:
                state_dict.update({k: v})

            if 'controlnet_down_blocks.' in k:
                state_dict.update({k: v})

            if 'controlnet_mid_block.' in k:
                state_dict.update({k: v})
            
            # if '.g_att' in k:
            #     state_dict.update({k: v})
            # if '.mamba' in k:
            #     state_dict.update({k: v})
            # if '.c_block' in k:
            #     state_dict.update({k: v})
            # if '.P_QA' in k:
            #     state_dict.update({k: v})
            # if '.decoder_map' in k:
            #     state_dict.update({k: v})
        model.load_state_dict(state_dict, strict=False)
        return model

    @classmethod
    def from_pretrained_control(cls, pretrained_model_path):
        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]
        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config)
        model_file = os.path.join(pretrained_model_path, "controlnet.pth")
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model


















if __name__=="__main__":
    device=torch.device("cuda:0")
    weight_type=torch.float16
    pretrained_model_path=r"./checkpoint-33000"
    unet = ControlUNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    unet=unet.to(device,dtype=weight_type)
    controlnet_images = {}
    x=torch.randn(1,4,16,28,28).to(device,dtype=weight_type)
    condition=torch.randn(16,3,224,224).to(device,dtype=weight_type)
    text=torch.randn(1,77,1024).to(device,dtype=weight_type)
    timesteps=torch.tensor([200]).to(device,dtype=weight_type)
    down_block_res_samples_dict, mid_block_res_sample_dict=unet(

        x, timesteps, encoder_hidden_states=text,  controlnet_cond=condition,
)
    print(down_block_res_samples_dict)
    print(mid_block_res_sample_dict)


