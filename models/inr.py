"""Implicit Neural Representation for Photography Retouching (Kinli et al., 2024).

Architecture: CNNDWSplitSiren
- Position branch (2 -> n/2): 1x1 convs with Sine activation
- Signal/RGB branch (3 -> n/2): 1x1 convs with Sine activation
- Merge branch (n -> 3): CNN_DW depthwise-separable blocks + 1x1 output
- Global skip: output = residual + input_RGB
- Default: 11,491 params (n_neurons=64, n_hidden_m=1)

Reference: https://arxiv.org/abs/2412.03848
"""

import math

import torch
import torch.nn as nn


class Sine(nn.Module):
    """Sine activation (SIREN-style)."""
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def _siren_init(layer, w0=1.0, is_first=False):
    """Apply SIREN weight initialization to a Conv2d layer.

    From Sitzmann et al. (2020):
    - First layer: U(-1/n, 1/n) where n = fan_in
    - Hidden layers: U(-sqrt(6/n)/w0, sqrt(6/n)/w0)
    """
    fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / w0
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)


def _cnn_dw_block(channels, sin_w):
    """1x1 Conv → Sine → DW 3x3 Conv → Sine → 1x1 Conv."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, 1),
        Sine(sin_w),
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        Sine(sin_w),
        nn.Conv2d(channels, channels, 1),
    )


class INRetouch(nn.Module):
    """Faithful reproduction of CNNDWSplitSiren from INRetouch paper.

    Args:
        n_neurons: hidden dimension (64 in paper)
        n_hidden_p: number of hidden layers in position branch (1 in paper)
        n_hidden_s: number of hidden layers in signal branch (1 in paper)
        n_hidden_m: number of CNN_DW blocks in merge branch (1 in paper)
        sin_w: Sine activation frequency (1.0 in paper)
        siren_init: whether to apply SIREN weight initialization
    """

    def __init__(self, n_neurons=64, n_hidden_p=1, n_hidden_s=1,
                 n_hidden_m=1, sin_w=1.0, siren_init=False):
        super().__init__()
        half = n_neurons // 2

        # Position branch: coords (2ch) → half channels
        layers_p = [nn.Conv2d(2, half, 1), Sine(sin_w)]
        for _ in range(n_hidden_p):
            layers_p.extend([nn.Conv2d(half, half, 1), Sine(sin_w)])
        self.model_p = nn.Sequential(*layers_p)

        # Signal/RGB branch: RGB (3ch) → half channels
        layers_s = [nn.Conv2d(3, half, 1), Sine(sin_w)]
        for _ in range(n_hidden_s):
            layers_s.extend([nn.Conv2d(half, half, 1), Sine(sin_w)])
        self.model_s = nn.Sequential(*layers_s)

        # Merge branch: concatenated (n_neurons) → CNN_DW → output
        layers_m = []
        for _ in range(n_hidden_m):
            layers_m.append(_cnn_dw_block(n_neurons, sin_w))
            layers_m.append(Sine(sin_w))
        layers_m.append(nn.Conv2d(n_neurons, 3, 1))
        self.model_m = nn.Sequential(*layers_m)

        if siren_init:
            self._apply_siren_init(sin_w)

    def _apply_siren_init(self, w0):
        """Apply SIREN-proper weight initialization to all Conv2d layers."""
        for branch in [self.model_p, self.model_s, self.model_m]:
            first = True
            for module in branch.modules():
                if isinstance(module, nn.Conv2d):
                    _siren_init(module, w0=w0, is_first=first)
                    first = False

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 5, H, W) where channels are [coords_w, coords_h, R, G, B]

        Returns:
            (B, 3, H, W) output image
        """
        p = x[:, :2]  # position (2ch)
        s = x[:, 2:]  # signal/RGB (3ch)

        out_p = self.model_p(p)
        out_s = self.model_s(s)
        update = self.model_m(torch.cat([out_p, out_s], dim=1))

        # Global skip connection: output = update + input_RGB
        output = update + s
        return output

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def make_coord_grid(H, W, device="cpu"):
    """Create coordinate grid in [-1, 1] range (matching INRetouch convention).

    Returns:
        (1, 2, H, W) tensor with (w, h) coords in [-1, 1]
    """
    w = torch.linspace(-1, 1, W, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    grid_w, grid_h = torch.meshgrid(w, h, indexing="xy")
    return torch.stack([grid_w, grid_h], dim=0).unsqueeze(0)
