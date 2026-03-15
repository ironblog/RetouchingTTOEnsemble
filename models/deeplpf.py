"""Parametric local filters (DeepLPF-style).

Filters:
1. Global tone curve: per-channel 1D curves (K control points each)
2. Graduated filter: linear spatial gradient with exposure/color params
3. Radial filter: circular vignette with exposure/color params

Each filter has very few parameters (10-30), making them transfer-friendly.
The ref-conditioned predictor outputs all filter parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ToneCurve(nn.Module):
    """Per-channel 1D tone curves via control points + monotonic interpolation.

    Each channel has K control points. The curve maps [0,1] → [0,1].
    Initialized to identity.
    """

    def __init__(self, n_points=17):
        super().__init__()
        self.n_points = n_points
        # Parameters: per-channel control point offsets (initialized to 0 = identity)
        # 3 channels × n_points = total params
        self.n_params = 3 * n_points

    def apply(self, img, params):
        """Apply tone curves to image.

        Args:
            img: (B, 3, H, W) or (H, W, 3)
            params: (B, 3*n_points) or (3*n_points,) control point offsets

        Returns:
            Same shape as img
        """
        is_hwc = img.dim() == 3
        if is_hwc:
            img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            params = params.unsqueeze(0)

        B, C, H, W = img.shape
        K = self.n_points

        # Control points: identity + offset
        offsets = params.reshape(B, 3, K)
        identity_y = torch.linspace(0, 1, K, device=img.device).unsqueeze(0).unsqueeze(0)
        ctrl_y = (identity_y + offsets).clamp(0, 1)  # (B, 3, K)

        # Ensure monotonicity by cumulative softmax of differences
        # Actually, let's use simple linear interpolation with soft clamping
        ctrl_x = torch.linspace(0, 1, K, device=img.device)  # (K,)

        # Apply per-channel: lookup + linear interpolation
        out_channels = []
        for c in range(3):
            channel = img[:, c:c+1]  # (B, 1, H, W)
            cy = ctrl_y[:, c]  # (B, K)

            # Scale to LUT indices
            scaled = channel * (K - 1)  # (B, 1, H, W)
            lo = scaled.floor().long().clamp(0, K - 2)
            frac = scaled - lo.float()

            # Gather low and high values
            lo_flat = lo.reshape(B, -1)  # (B, H*W)
            hi_flat = (lo_flat + 1).clamp(max=K - 1)

            # cy: (B, K), lo_flat: (B, H*W) — use gather along K dim
            val_lo = cy.gather(1, lo_flat)  # (B, H*W)
            val_hi = cy.gather(1, hi_flat)

            frac_flat = frac.reshape(B, -1)
            interp = val_lo + (val_hi - val_lo) * frac_flat
            out_channels.append(interp.reshape(B, 1, H, W))

        result = torch.cat(out_channels, dim=1)

        if is_hwc:
            return result.squeeze(0).permute(1, 2, 0)
        return result


class GraduatedFilter(nn.Module):
    """Linear spatial gradient filter.

    Parameters per filter:
    - angle: rotation of gradient direction
    - center: position of 50% transition
    - width: transition width
    - exposure_shift: brightness adjustment in affected region (3ch)

    Total: 6 params per filter
    """

    def __init__(self, n_filters=2):
        super().__init__()
        self.n_filters = n_filters
        self.n_params = n_filters * 6  # angle, center, width, exp_shift(3)

    def apply(self, img, params):
        """Apply graduated filters.

        Args:
            img: (B, 3, H, W) or (H, W, 3)
            params: (B, n_filters*6) or (n_filters*6,)
        """
        is_hwc = img.dim() == 3
        if is_hwc:
            img = img.permute(2, 0, 1).unsqueeze(0)
            params = params.unsqueeze(0)

        B, C, H, W = img.shape
        result = img.clone()

        for f in range(self.n_filters):
            offset = f * 6
            angle = params[:, offset] * math.pi  # [-π, π]
            center = torch.sigmoid(params[:, offset + 1])  # [0, 1]
            width = torch.sigmoid(params[:, offset + 2]) * 0.5 + 0.1  # [0.1, 0.6]
            exp_shift = params[:, offset + 3:offset + 6].tanh() * 0.3  # [-0.3, 0.3]

            # Create spatial gradient
            yy = torch.linspace(-0.5, 0.5, H, device=img.device)
            xx = torch.linspace(-0.5, 0.5, W, device=img.device)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")

            # Rotate coordinates
            cos_a = angle.cos().reshape(B, 1, 1)
            sin_a = angle.sin().reshape(B, 1, 1)
            rotated = gx.unsqueeze(0) * cos_a + gy.unsqueeze(0) * sin_a  # (B, H, W)

            # Sigmoid mask
            center_r = center.reshape(B, 1, 1) - 0.5
            width_r = width.reshape(B, 1, 1)
            mask = torch.sigmoid((rotated - center_r) / (width_r + 1e-6))  # (B, H, W)
            mask = mask.unsqueeze(1)  # (B, 1, H, W)

            # Apply exposure shift
            shift = exp_shift.reshape(B, 3, 1, 1)
            result = result + mask * shift

        result = result.clamp(0, 1)

        if is_hwc:
            return result.squeeze(0).permute(1, 2, 0)
        return result


class RadialFilter(nn.Module):
    """Circular vignette filter.

    Parameters:
    - cx, cy: center position
    - radius: vignette radius
    - falloff: transition sharpness
    - exposure_shift: per-channel adjustment (3)

    Total: 7 params
    """

    def __init__(self):
        super().__init__()
        self.n_params = 7

    def apply(self, img, params):
        is_hwc = img.dim() == 3
        if is_hwc:
            img = img.permute(2, 0, 1).unsqueeze(0)
            params = params.unsqueeze(0)

        B, C, H, W = img.shape

        cx = torch.sigmoid(params[:, 0])  # [0, 1]
        cy = torch.sigmoid(params[:, 1])
        radius = torch.sigmoid(params[:, 2]) * 0.8 + 0.1  # [0.1, 0.9]
        falloff = torch.sigmoid(params[:, 3]) * 5 + 1  # [1, 6]
        exp_shift = params[:, 4:7].tanh() * 0.3  # [-0.3, 0.3]

        yy = torch.linspace(0, 1, H, device=img.device)
        xx = torch.linspace(0, 1, W, device=img.device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")

        cx_r = cx.reshape(B, 1, 1)
        cy_r = cy.reshape(B, 1, 1)
        r_r = radius.reshape(B, 1, 1)
        f_r = falloff.reshape(B, 1, 1)

        dist = ((gx.unsqueeze(0) - cx_r).pow(2) + (gy.unsqueeze(0) - cy_r).pow(2)).sqrt()
        mask = torch.sigmoid(f_r * (dist - r_r))  # (B, H, W)
        mask = mask.unsqueeze(1)  # (B, 1, H, W)

        shift = exp_shift.reshape(B, 3, 1, 1)
        result = (img + mask * shift).clamp(0, 1)

        if is_hwc:
            return result.squeeze(0).permute(1, 2, 0)
        return result


class ParametricFilterModel(nn.Module):
    """DeepLPF model: ref-conditioned parametric filter predictor."""

    def __init__(self, n_curve_points=17, n_grad_filters=2,
                 style_dim=128, ref_size=256):
        super().__init__()
        self.ref_size = ref_size

        self.tone_curve = ToneCurve(n_points=n_curve_points)
        self.grad_filter = GraduatedFilter(n_filters=n_grad_filters)
        self.radial_filter = RadialFilter()

        self.total_params = (self.tone_curve.n_params +
                             self.grad_filter.n_params +
                             self.radial_filter.n_params)

        # Ref encoder
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.param_head = nn.Sequential(
            nn.Linear(128, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, self.total_params),
        )

        # Initialize param_head bias to zero (identity filters)
        nn.init.zeros_(self.param_head[-1].weight)
        nn.init.zeros_(self.param_head[-1].bias)

    def predict_params(self, ref_before, ref_after):
        """Predict all filter parameters from ref pair."""
        x = torch.cat([ref_before, ref_after], dim=1)
        feat = self.ref_encoder(x).squeeze(-1).squeeze(-1)
        return self.param_head(feat)

    def apply_filters(self, img, params):
        """Apply all filters sequentially.

        Args:
            img: (B, 3, H, W)
            params: (B, total_params)
        """
        idx = 0
        # Tone curve
        n = self.tone_curve.n_params
        out = self.tone_curve.apply(img, params[:, idx:idx+n])
        idx += n

        # Graduated filters
        n = self.grad_filter.n_params
        out = self.grad_filter.apply(out, params[:, idx:idx+n])
        idx += n

        # Radial filter
        n = self.radial_filter.n_params
        out = self.radial_filter.apply(out, params[:, idx:idx+n])

        return out

    def forward(self, input_img, ref_before, ref_after):
        params = self.predict_params(ref_before, ref_after)
        output = self.apply_filters(input_img, params)
        return output, params

    def forward_with_ref_recon(self, input_img, ref_before, ref_after):
        params = self.predict_params(ref_before, ref_after)
        output = self.apply_filters(input_img, params)
        ref_recon = self.apply_filters(ref_before, params)
        return output, ref_recon, params

    def param_count(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "filter_params": self.total_params,
        }
