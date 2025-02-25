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
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class Masked3DMSELoss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: tuple = (8, 16, 16), stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = (
            patch_size[0] // stride,
            patch_size[1] // stride,
            patch_size[2] // stride
        )
        self.norm_pix = norm_pix

    def patchify(self, imgs, nt, nh, nw):
        p1, p2, p3 = self.scale_factor
        x = rearrange(
            imgs,
            "b c (nt p1) (nh p2) (nw p3) -> b (nt nh nw) (p1 p2 p3 c)",
            nt=nt, nh=nh, nw=nw, p1=p1, p2=p2, p3=p3
        )
        return x

    def unpatchify(self, x, nt, nh, nw):
        p1, p2, p3 = self.scale_factor
        imgs = rearrange(
            x,
            "b (nt nh nw) (p1 p2 p3 c) -> b c (nt p1) (nh p2) (nw p3)",
            nt=nt, nh=nh, nw=nw, p1=p1, p2=p2, p3=p3
        )
        return imgs

    def forward(self, input, target, mask=None):
        T, H, W = input.shape[-3:]
        p1, p2, p3 = self.scale_factor
        nt, nh, nw = T // p1, H // p2, W // p3

        if self.norm_pix:
            target = self.patchify(target, nt, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nt, nh, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nt nh nw) -> b nt nh nw", nt=nt, nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(T, H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, T, H, W -> B, T, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean() # Account for zero masks
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss


class TestMasked3DMSELoss(unittest.TestCase):
    def test_masked_3d_mse_loss(self): # PASSING
        patch_size = (8, 16, 16)
        loss = Masked3DMSELoss(patch_size=patch_size)
        input = torch.rand(1, 1, 128, 512, 512)
        target = torch.rand(1, 1, 128, 512, 512)
        mask = torch.ones(1, int((128 / patch_size[0]) * (512 / patch_size[1]) * (512 / patch_size[2])))
        print(loss(input, target, mask))
        print('Finished')


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param label_smoothing: Amount of smoothing in the loss (default is 0.0)
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, label_smoothing : float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size[0] // stride
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask=None):

        loss = F.cross_entropy(input, target, reduction='none', label_smoothing=self.label_smoothing)

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            H, W = input.shape[-2:]
            nh, nw = H // self.scale_factor, W // self.scale_factor
            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss


class MaskedMSELoss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size[0] // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean() # Account for zero masks
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss


class MaskedL1Loss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.l1_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss
