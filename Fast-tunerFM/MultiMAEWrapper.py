from torch import nn
from torch import Tensor
import torch
from util_funcs import get_model

class MultiMAEWrapper(nn.Module):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        weights=None,
    ):
        super().__init__()

        assert weights is not None
        state_dict = torch.load(weights, map_location="cpu")

        self.args = state_dict["args"]

        modalities = self.args.in_domains
        self.args.input_size = {}
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        for domain in modalities:
            self.args.input_size[domain] = input_size
        self.args.patch_size = {}
        for domain in modalities:
            self.args.patch_size[domain] = patch_size
        self.args.grid_sizes = {}
        for domain in modalities:
            self.args.grid_sizes[domain] = []
            for i in range(len(input_size)):
                self.args.grid_sizes[domain].append(input_size[i] // patch_size[i])

        self.model = get_model(self.args)
        print('>> Loading weights from:', weights)
        self.model.load_state_dict(state_dict["model"], strict=True)

        self.model.output_adapters = None

    def forward(self, x: Tensor):
        """
        Args:
            x: (B, C, H, W) tensor. H and W are determined by the
            input_size parameter in the constructor. It expects a tensor
            in the range [0, 1].
        Returns:
            (B, C, H, W) tensor
        """
        x = x.to(self.device)
        x_d = {'bscan': x}
        preds, _masks = self.model(x_d, mask_inputs=False)
        return preds

    @property
    def device(self):
        return next(self.parameters()).device
