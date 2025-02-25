# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
from typing import Dict, Optional, Union

import torch
from einops import repeat
from torch import nn

from utils.registry import register_model
from .parts.multimae_module import MultiMAE


__all__ = [
    'pretrain_multimae_very_small',
    'pretrain_multimae_base',
    'pretrain_multimae_large',
    'multivit_base',
    'multivit_large',
]




@register_model
def pretrain_multimae_very_small(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        args,
        **kwargs):
    model = MultiMAE(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=12,
        depth=1,
        num_heads=1,
        mlp_ratio=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def pretrain_multimae_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        args,
        **kwargs):
    model = MultiMAE(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def pretrain_multimae_large(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        args,
        **kwargs):
    model = MultiMAE(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


class MultiViT(MultiMAE):
    """MultiViT: Multi-modal Vision Transformer
    This is MultiMAE without masking and with a simplified / faster forward pass


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """

    def process_input(self, x):

        # If input x is a Tensor, assume it's RGB
        x = {'bscan': x} if isinstance(x, torch.Tensor) else x
        # Need image size for tokens->image reconstruction
        if 'bscan' in x:
            B, _, H, W = x['bscan'].shape
        elif 'semseg' in x:
            B, H, W = x['semseg'].shape
            H *= self.input_adapters['semseg'].stride_level
            W *= self.input_adapters['semseg'].stride_level
        else:
            B, _, H, W = list(x.values())[0].shape  # TODO: Deal with case where not all have same shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(input_task_tokens=input_task_tokens, image_size=(H, W))
        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        return input_tokens, input_info

    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor], return_all_layers=False, **kwargs):
        """
        Forward pass through input adapters, transformer encoder and output adapters.

        :param x: Input tensor or dictionary of tensors
        :param return_all_layers: Set to True to return all transformer layers
        """

        input_tokens, input_info = self.process_input(x)

        # Pass tokens through Transformer
        if not return_all_layers:
            encoder_tokens = self.encoder(input_tokens)
        else:
            # Optionally access every intermediate layer
            encoder_tokens = []
            tokens = input_tokens
            for block in self.encoder:
                tokens = block(tokens)
                encoder_tokens.append(tokens)

        if self.output_adapters is None:
            return encoder_tokens

        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
            )
            for domain in self.output_adapters
        }

        return preds


@register_model
def multivit_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        args,
        **kwargs):
    model = MultiViT(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def multivit_large(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        args,
        **kwargs):
    model = MultiViT(
        args,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
