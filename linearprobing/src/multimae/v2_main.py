#!/usr/bin/env python

import sys
import socket
from functools import partial
import copy
import argparse

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torchvision.utils import save_image
import numpy as np


if socket.gethostname() == 'hemingway':
    sys.path.append('path')
elif socket.gethostname().startswith('s0-'):
    sys.path.append('path')
else:
    raise ValueError('Unknown host')

# # Import modules directly from MultiOptiMAE
# spec = importlib.util.spec_from_file_location("multimae_utils", multimae_utils)
# assert spec is not None
# module = importlib.util.module_from_spec(spec)
# sys.modules["multimae_utils"] = module
# spec.loader.exec_module(module)  # type: ignore

from parts.multimae_utils import pair  # type: ignore
from parts.input_adapters import (  # type: ignore
    PatchedInputAdapter, SemSegInputAdapter
)
from parts.output_adapters import SpatialOutputAdapter  # type: ignore
from parts.multimae_module import MultiMAE  # type: ignore


########################################################################
# Dirichlet from older PyTorch version

# This helper is exposed for testing.
def _Dirichlet_backward(x, concentration, grad_output):
    total = concentration.sum(-1, True).expand_as(concentration)
    grad = torch._dirichlet_grad(x, concentration, total)
    return grad * (grad_output - (x * grad_output).sum(-1, True))


class _Dirichlet(Function):
    @staticmethod
    def forward(ctx, concentration):
        x = torch._sample_dirichlet(concentration)
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, concentration = ctx.saved_tensors
        return _Dirichlet_backward(x, concentration, grad_output)


class Dirichlet(ExponentialFamily):
    r"""
    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    """

    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    support = constraints.simplex
    has_rsample = True

    def __init__(self, concentration, validate_args=None):
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        return _Dirichlet.apply(concentration)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )

    @property
    def mean(self):
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def mode(self):
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        mode = concentrationm1 / concentrationm1.sum(-1, True)
        mask = (self.concentration < 1).all(axis=-1)
        mode[mask] = torch.nn.functional.one_hot(
            mode[mask].argmax(axis=-1), concentrationm1.shape[-1]
        ).to(mode)
        return mode

    @property
    def variance(self):
        con0 = self.concentration.sum(-1, True)
        return (
            self.concentration
            * (con0 - self.concentration)
            / (con0.pow(2) * (con0 + 1))
        )

    def entropy(self):
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        return (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(a0)
            - (k - a0) * torch.digamma(a0)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )

    @property
    def _natural_params(self):
        return (self.concentration,)

    def _log_normalizer(self, x):
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))


########################################################################




DEFAULT_CONF = {
    "channels": 1,
    "stride_level": 1,
    "input_adapter": partial(PatchedInputAdapter, num_channels=1),
    "output_adapter": partial(SpatialOutputAdapter, num_channels=1),
}


DOMAIN_CONF = {
    "slo": copy.deepcopy(DEFAULT_CONF),
    "bscan": copy.deepcopy(DEFAULT_CONF),
    "bscanlayermap": {
        "num_classes": 13,
        "stride_level": 1,
        "input_adapter": partial(
            SemSegInputAdapter,
            num_classes=13,
            dim_class_emb=64,
            interpolate_class_emb=False,
        ),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=13),
    },
    # Original domains
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
    },
    'depth': copy.deepcopy(DEFAULT_CONF),
    # See here the explanation of the parameters for semseg:
    #   https://github.com/EPFL-VILAB/MultiMAE/issues/8#issuecomment-1127633408
    'semseg': {
        'num_classes': 133,
        'stride_level': 4,
        'input_adapter': partial(SemSegInputAdapter, num_classes=133,
                                 dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=133),
    },
}


def get_model(args):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")
    if isinstance(args.patch_size, int):
        args.patch_size = {domain: (args.patch_size, args.patch_size) for domain in args.all_domains}

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            image_size=args.input_size[domain],
        )
        for domain in args.in_domains
    }
  
    if 'large' in args.model:
        model = MultiMAE(
            args=args,
            input_adapters=input_adapters,
            output_adapters=None,
            num_global_tokens=args.num_global_tokens,
            drop_path_rate=args.drop_path,
            dim_tokens=1024,
            depth=24,
            num_heads=16,
        )
    else:
        model = MultiMAE(
            args=args,
            input_adapters=input_adapters,
            output_adapters=None,
            num_global_tokens=args.num_global_tokens,
            drop_path_rate=args.drop_path,
        )

    return model


default_args = argparse.Namespace(
    model="pretrain_multimae_base",
    in_domains=["bscan", "slo"],
    out_domains=None,
    patch_size=32,
    input_size=512,
    decoder_dim=256,
    alphas=1.0,
    num_encoded_tokens=98,
    drop_path=0.0,
    extra_norm_pix_loss=False,
    standarize_depth=True,
    num_global_tokens=1,
    decoder_depth=2,
    decoder_use_task_queries=True,
    decoder_num_heads=8,
    decoder_use_xattn=True,
    custom_sampling=False,
)


class MultiMAEWrapper(nn.Module):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        num_classes=1,
        all_tokens=False,
        modalities="bscan-slo",
        input_modality="bscan",
        weights=None,
        representation_method=None
    ):
        super().__init__()

        assert weights is not None
        state_dict = torch.load(weights, map_location="cuda")
        model_state_dict = state_dict["model"]
        try:
            self.args = state_dict["args"]
        except KeyError:
            self.args = default_args

        # Overwrite some args
        print('>> Overwriting some args for inference')
        # self.args.out_domains = ["bscan"]
        print(f"Using input modality: {input_modality}")
        self.input_modality = input_modality
        modalities = modalities.split("-")
        self.args.in_domains = modalities
        self.args.input_size = {}
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        for domain in modalities:
            if domain != "bscanlayermap":
                self.args.input_size[domain] = input_size
            else:
                self.args.input_size[domain] = (128, 128)
        self.args.patch_size = {}
        for domain in modalities:
            if domain != "bscanlayermap":
                self.args.patch_size[domain] = patch_size
            else:
                self.args.patch_size[domain] = (8, 8)
        self.args.grid_sizes = {}
        for domain in modalities:
            self.args.grid_sizes[domain] = []
            for i in range(len(input_size)):
                self.args.grid_sizes[domain].append(input_size[i] // patch_size[i])
        # self.args.input_size = input_size
        # self.args.patch_size = patch_size
        # if "rgb" in self.input_modality:
        #     # IMPORTANT: Patch size becomes 16
        #     self.args.patch_size = 16
        #     self.args.input_size = 224

        self.model = get_model(self.args)
        self.model.output_adapters = None
        self.representation_method = representation_method
        print('Using ', self.representation_method, ' as model representation.')
        self.output_dim_factor = 2 if self.representation_method == 'concat' else 1

        if 'large' in self.args.model:
            self.output_linear = nn.Linear(1024 * self.output_dim_factor, num_classes)
            self.ln = nn.LayerNorm(normalized_shape=1024, elementwise_affine=True)
            print('Using layer norm')
        else:
            self.output_linear = nn.Linear(768 * self.output_dim_factor, num_classes) #change this back to 768
        self.all_tokens = all_tokens
        print('>> Loading weights from:', weights)
        try:
            self.model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            if 'missing key(s)' in str(e).lower():
                raise ValueError(f"Error loading model: {e}")
            if "unexpected key(s)" in str(e).lower():
                self.model.load_state_dict(model_state_dict, strict=False)

    def forward_features(self, x: Tensor):
        return self.forward(x, get_embeddings=True)

    def forward(self, x: Tensor, get_embeddings=False):
        """
        Args:
            x: (B, C, H, W) tensor. H and W are determined by the
            input_size parameter in the constructor. It expects a tensor
            in the range [0, 1].
        Returns:
            (B, C, H, W) tensor
        """
        if get_embeddings:
            self.model.output_adapters = None
        if x.device != self.device:
            x = x.to(self.device)
        if self.input_modality == "rgb":
            if x.ndim == 3:
                x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            elif x.ndim == 4 and x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
        if self.input_modality == "bscan-slo":
            x_d = {
                "bscan": x[:, 0:1],
                "slo": x[:, 1:2],
            }
            # import matplotlib.pyplot as plt
            # plt.imshow(x_d["bscan"][0,0].cpu().numpy(), cmap='gray')
            # plt.show()
            # plt.imshow(x_d["slo"][0,0].cpu().numpy(), cmap='gray')
            # plt.show()
        elif self.input_modality == "rgb-depth":
            x_d = {
                "rgb": x[:, 0:1].repeat(1, 3, 1, 1),
                "depth": x[:, 1:2],
            }
        else:
            x_d = {self.input_modality: x}
        out, _masks = self.model(x_d, mask_inputs=False)
#        if self.all_tokens:
#            out = out.mean(dim=1)
#        else:
#            # Exclude cls token:
#            if self.cls_token_only:
#                out = out[:, -self.args.num_global_tokens, :]
#            else:
#                out = out[:, :-self.args.num_global_tokens, :].mean(dim=1)

        if 'large' in self.args.model:
            out = self.ln(out)
            
        if self.representation_method == 'patch_features':
            out = out[:, :-self.args.num_global_tokens, :].mean(dim=1)
        elif self.representation_method == 'cls_token':
            out = out[:, -self.args.num_global_tokens, :]
        elif self.representation_method == 'all_tokens':
            out = out.mean(dim=1)
        elif self.representation_method == 'concat':
            cls = out[:, -self.args.num_global_tokens, :]
            pooling = out[:, :-self.args.num_global_tokens, :].mean(dim=1)
            out = torch.concat((cls, pooling), 1)
            
        if get_embeddings:
            return out
        return self.output_linear(out)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}



if __name__ == "__main__":

    weights = "./_weights/bscan-slo_checkpoint-1599.pth"

    # model.load_state_dict(torch.load(weights, map_location='cuda')['model'], strict=True)
    model = MultiMAEWrapper(input_size=(512, 512), num_classes=1)
    model.load_weights(weights, map_location="cuda")

    oct = np.load(
        "./_sample_data/3doct/67115194-25-QAFNDVAJSCAAWUMEAEZRIAYYI+CULRTGPAJTDZZAIGWXFHVFTVVCZKFEOKVRVYH+OSYKOR.npy"
    )
    bscan = (
        torch.from_numpy(oct[oct.shape[0] // 2])
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .cuda()
    )
    # normalize to range [0, 1]
    bscan = (bscan - bscan.min()) / (bscan.max() - bscan.min())
    print(bscan.shape, bscan.min(), bscan.max())
    save_image(bscan, "./_output/in_bscan.png", normalize=True)
    with torch.no_grad():
        out = model(bscan)
        print(out.shape)
        save_image(out, f"./_output/out_bscan.png", normalize=True)
