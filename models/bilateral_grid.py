"""Bilateral Grid for spatially-aware color transforms.

A bilateral grid extends 3D LUT by indexing over (x, y, guide) instead of
(R, G, B). At each grid cell, it stores a 3x4 affine color transform matrix.
This captures spatial-dependent effects like vignetting and local contrast
while maintaining smoothness through the grid structure.

The guide dimension can be:
  - luma: standard luminance (default, original behavior)
  - proj_ls: learnable projection of luma + saturation
  - proj_lc: learnable projection of luma + local contrast
  - proj_lsc: learnable projection of luma + saturation + local contrast

Reference: Gharbi et al., "Deep Bilateral Learning for Real-Time Image Enhancement", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def compute_luma(rgb: torch.Tensor) -> torch.Tensor:
    """Compute luminance from RGB. Input (..., 3), output (...)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def compute_saturation(rgb: torch.Tensor) -> torch.Tensor:
    """Compute saturation (chroma magnitude) from RGB. Input (..., 3), output (...)."""
    maxc = rgb.max(dim=-1).values
    minc = rgb.min(dim=-1).values
    denom = maxc.clamp(min=1e-6)
    return (maxc - minc) / denom


def compute_contrast(rgb: torch.Tensor, mode: str = "sobel", downscale: int = 4) -> torch.Tensor:
    """Compute local contrast from RGB image.

    Args:
        rgb: (H, W, 3) float tensor
        mode: "sobel" or "laplacian"
        downscale: compute at 1/downscale resolution then upsample (prevents overfitting)

    Returns:
        (H, W) contrast map normalized to [0, 1]
    """
    H, W = rgb.shape[:2]
    gray = compute_luma(rgb)  # (H, W)

    # Downsample for computation
    if downscale > 1:
        dh, dw = max(H // downscale, 4), max(W // downscale, 4)
        gray_down = F.interpolate(
            gray.unsqueeze(0).unsqueeze(0), size=(dh, dw),
            mode="bilinear", align_corners=False
        )  # (1, 1, dh, dw)
    else:
        gray_down = gray.unsqueeze(0).unsqueeze(0)
        dh, dw = H, W

    if mode == "sobel":
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=rgb.dtype, device=rgb.device).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=rgb.dtype, device=rgb.device).reshape(1, 1, 3, 3)
        gx = F.conv2d(gray_down, sobel_x, padding=1)
        gy = F.conv2d(gray_down, sobel_y, padding=1)
        contrast = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(0).squeeze(0)
    elif mode == "laplacian":
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=rgb.dtype, device=rgb.device).reshape(1, 1, 3, 3)
        contrast = F.conv2d(gray_down, lap, padding=1).abs().squeeze(0).squeeze(0)
    else:
        raise ValueError(f"Unknown contrast mode: {mode}")

    # Upsample back to full resolution
    if contrast.shape[0] != H or contrast.shape[1] != W:
        contrast = F.interpolate(
            contrast.unsqueeze(0).unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

    # Normalize to [0, 1] using percentile-robust normalization
    p99 = torch.quantile(contrast, 0.99).clamp(min=1e-6)
    contrast = (contrast / p99).clamp(0, 1)

    return contrast


def compute_guide(
    rgb: torch.Tensor,
    mode: str = "luma",
    proj_params: torch.Tensor = None,
    contrast_mode: str = "sobel",
    contrast_downscale: int = 4,
) -> torch.Tensor:
    """Compute guide value for bilateral grid indexing.

    Args:
        rgb: (H, W, 3) or (N, 3) float tensor
        mode: guide mode string:
            "luma" - standard luminance
            "sat" - saturation only
            "fixed_ls" - 0.7*luma + 0.3*sat (fixed coefficients)
            "fixed_lc" - 0.7*luma + 0.3*contrast
            "fixed_lsc" - 0.6*luma + 0.2*sat + 0.2*contrast
            "proj_ls" - learnable sigmoid(a*luma + b*sat + d)
            "proj_lc" - learnable sigmoid(a*luma + b*contrast + d)
            "proj_lsc" - learnable sigmoid(a*luma + b*sat + c*contrast + d)
        proj_params: learnable projection parameters (for proj_* modes)
        contrast_mode: "sobel" or "laplacian"
        contrast_downscale: downscale factor for contrast computation

    Returns:
        guide tensor same spatial shape as input (without channel dim)
    """
    if mode == "luma":
        return compute_luma(rgb).clamp(0, 1)

    is_spatial = (rgb.dim() == 3)  # (H, W, 3) vs (N, 3)

    luma = compute_luma(rgb)

    if mode == "sat":
        return compute_saturation(rgb).clamp(0, 1)

    elif mode == "fixed_ls":
        sat = compute_saturation(rgb)
        return (0.7 * luma + 0.3 * sat).clamp(0, 1)

    elif mode == "fixed_lc":
        if is_spatial:
            contrast = compute_contrast(rgb, contrast_mode, contrast_downscale)
        else:
            contrast = torch.zeros_like(luma)
        return (0.7 * luma + 0.3 * contrast).clamp(0, 1)

    elif mode == "fixed_lsc":
        sat = compute_saturation(rgb)
        if is_spatial:
            contrast = compute_contrast(rgb, contrast_mode, contrast_downscale)
        else:
            contrast = torch.zeros_like(luma)
        return (0.6 * luma + 0.2 * sat + 0.2 * contrast).clamp(0, 1)

    elif mode == "proj_ls":
        sat = compute_saturation(rgb)
        a, b, d = proj_params[0], proj_params[1], proj_params[2]
        g = torch.sigmoid(a * luma + b * sat + d)

    elif mode == "proj_lc":
        if is_spatial:
            contrast = compute_contrast(rgb, contrast_mode, contrast_downscale)
        else:
            contrast = torch.zeros_like(luma)
        a, b, d = proj_params[0], proj_params[1], proj_params[2]
        g = torch.sigmoid(a * luma + b * contrast + d)

    elif mode == "proj_lsc":
        sat = compute_saturation(rgb)
        if is_spatial:
            contrast = compute_contrast(rgb, contrast_mode, contrast_downscale)
        else:
            contrast = torch.zeros_like(luma)
        a, b, c, d = proj_params[0], proj_params[1], proj_params[2], proj_params[3]
        g = torch.sigmoid(a * luma + b * sat + c * contrast + d)

    else:
        raise ValueError(f"Unknown guide mode: {mode}")

    return g


def init_proj_params(mode: str, device: str = "cuda") -> torch.Tensor:
    """Initialize projection parameters for a given guide mode.

    Initialized so that the projection starts close to pure luma:
    sigmoid(2*luma + 0*sat + 0*contrast - 1) ~ luma-like.
    Fixed modes (luma, sat, fixed_*) return None (no learnable params).
    """
    if mode == "proj_ls":
        return torch.tensor([2.0, 0.0, -1.0], device=device, requires_grad=True)
    elif mode == "proj_lc":
        return torch.tensor([2.0, 0.0, -1.0], device=device, requires_grad=True)
    elif mode == "proj_lsc":
        return torch.tensor([2.0, 0.0, 0.0, -1.0], device=device, requires_grad=True)
    elif mode in ("luma", "sat", "fixed_ls", "fixed_lc", "fixed_lsc"):
        return None
    else:
        raise ValueError(f"Unknown guide mode: {mode}")


# ---------------------------------------------------------------------------
# Bilateral Grid Model
# ---------------------------------------------------------------------------

class BilateralGrid(nn.Module):
    """Learnable bilateral grid with affine color transforms.

    Grid dimensions: (spatial_h, spatial_w, guide_bins, 12)
    where 12 = 3 output channels x (3 input + 1 bias) affine coefficients.

    Initialized to identity: output = input.
    """

    def __init__(
        self,
        spatial_h: int = 16,
        spatial_w: int = 16,
        luma_bins: int = 8,
    ):
        super().__init__()
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.luma_bins = luma_bins

        # Initialize to identity affine transform
        identity = torch.zeros(spatial_h, spatial_w, luma_bins, 12)
        identity[..., 0] = 1.0   # r_r = 1
        identity[..., 5] = 1.0   # g_g = 1
        identity[..., 10] = 1.0  # b_b = 1

        self.grid = nn.Parameter(identity)

    def forward(
        self,
        pixels: torch.Tensor,
        coords: torch.Tensor,
        guide: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply bilateral grid via trilinear interpolation.

        Args:
            pixels: (..., 3) RGB float [0, 1]
            coords: (..., 2) normalized spatial coordinates [0, 1]
            guide: (...,) guide values in [0, 1]. If None, uses luminance.

        Returns:
            (..., 3) transformed pixels
        """
        orig_shape = pixels.shape[:-1]
        flat_pixels = pixels.reshape(-1, 3)
        flat_coords = coords.reshape(-1, 2)
        N = flat_pixels.shape[0]

        if guide is not None:
            flat_guide = guide.reshape(-1)
        else:
            # Default: luminance
            flat_guide = compute_luma(flat_pixels)

        # Normalize to grid coordinates
        gx = flat_coords[:, 0] * (self.spatial_w - 1)
        gy = flat_coords[:, 1] * (self.spatial_h - 1)
        gz = flat_guide.clamp(0, 1) * (self.luma_bins - 1)

        # Trilinear interpolation of the 12 affine coefficients
        coeffs = self._trilinear_sample_flat(gy, gx, gz)  # (N, 12)

        # Apply affine transform: out_c = sum(A_cr * in_r) + bias_c
        A = coeffs.reshape(N, 3, 4)  # (N, 3, 4)
        rgb_1 = torch.cat([flat_pixels, torch.ones(N, 1, device=pixels.device)], dim=-1)  # (N, 4)
        output = torch.einsum('nij,nj->ni', A, rgb_1)  # (N, 3)

        return output.reshape(*orig_shape, 3)

    def _trilinear_sample_flat(
        self, gy: torch.Tensor, gx: torch.Tensor, gz: torch.Tensor
    ) -> torch.Tensor:
        """Trilinear interpolation from the grid (flat version)."""
        Sh, Sw, Sl = self.spatial_h, self.spatial_w, self.luma_bins

        gy = gy.clamp(0, Sh - 1.001)
        gx = gx.clamp(0, Sw - 1.001)
        gz = gz.clamp(0, Sl - 1.001)

        y0 = gy.floor().long()
        x0 = gx.floor().long()
        z0 = gz.floor().long()
        y1 = (y0 + 1).clamp(max=Sh - 1)
        x1 = (x0 + 1).clamp(max=Sw - 1)
        z1 = (z0 + 1).clamp(max=Sl - 1)

        fy = (gy - y0.float()).unsqueeze(-1)
        fx = (gx - x0.float()).unsqueeze(-1)
        fz = (gz - z0.float()).unsqueeze(-1)

        # 8 corners
        c000 = self.grid[y0, x0, z0]
        c001 = self.grid[y0, x0, z1]
        c010 = self.grid[y0, x1, z0]
        c011 = self.grid[y0, x1, z1]
        c100 = self.grid[y1, x0, z0]
        c101 = self.grid[y1, x0, z1]
        c110 = self.grid[y1, x1, z0]
        c111 = self.grid[y1, x1, z1]

        c00 = c000 + (c001 - c000) * fz
        c01 = c010 + (c011 - c010) * fz
        c10 = c100 + (c101 - c100) * fz
        c11 = c110 + (c111 - c110) * fz

        c0 = c00 + (c01 - c00) * fx
        c1 = c10 + (c11 - c10) * fx

        result = c0 + (c1 - c0) * fy
        return result

    def tv_loss(self) -> torch.Tensor:
        """Total-variation smoothness on the grid."""
        dy = (self.grid[1:, :, :, :] - self.grid[:-1, :, :, :]).pow(2).mean()
        dx = (self.grid[:, 1:, :, :] - self.grid[:, :-1, :, :]).pow(2).mean()
        dz = (self.grid[:, :, 1:, :] - self.grid[:, :, :-1, :]).pow(2).mean()
        return dy + dx + dz


# ---------------------------------------------------------------------------
# Coordinate grid helper
# ---------------------------------------------------------------------------

def make_coord_grid(H: int, W: int, device: str = "cpu") -> torch.Tensor:
    """Create normalized spatial coordinate grid.

    Returns:
        (H, W, 2) tensor with (x, y) in [0, 1] where x=col/W, y=row/H
    """
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


# ---------------------------------------------------------------------------
# Fitting (per-sample optimization)
# ---------------------------------------------------------------------------

def fit_bilateral_grid(
    rb: torch.Tensor,
    ra: torch.Tensor,
    spatial_h: int = 16,
    spatial_w: int = 16,
    luma_bins: int = 8,
    fit_steps: int = 500,
    lr: float = 0.02,
    pixel_samples_k: int = 200,
    early_stop_patience: int = 30,
    early_stop_min_delta: float = 1e-6,
    tv_weight: float = 0.001,
    multires: bool = True,
    multires_lowres: int = 512,
    multires_steps_low: int = 200,
    multires_steps_high: int = 300,
    device: str = "cuda",
    guide_mode: str = "luma",
    contrast_mode: str = "sobel",
    contrast_downscale: int = 4,
    proj_reg_weight: float = 1e-4,
) -> tuple:
    """Fit a bilateral grid to map rb -> ra.

    Args:
        rb: (H, W, 3) reference before, float [0, 1]
        ra: (H, W, 3) reference after, float [0, 1]
        guide_mode: "luma", "proj_ls", "proj_lc", "proj_lsc"
        contrast_mode: "sobel" or "laplacian"
        contrast_downscale: downscale factor for contrast computation
        proj_reg_weight: L2 regularization for projection params

    Returns:
        (BilateralGrid, proj_params or None)
    """
    rb = rb.to(device)
    ra = ra.to(device)
    H, W, _ = rb.shape

    grid = BilateralGrid(spatial_h, spatial_w, luma_bins).to(device)
    proj_params = init_proj_params(guide_mode, device)

    coords_full = make_coord_grid(H, W, device)

    if multires and max(H, W) > multires_lowres:
        # Phase 1: low-res fit
        scale = multires_lowres / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        rb_low = F.interpolate(
            rb.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        ra_low = F.interpolate(
            ra.permute(2, 0, 1).unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        coords_low = make_coord_grid(new_h, new_w, device)

        proj_params = _optimize_grid_guided(
            grid, rb_low, ra_low, coords_low,
            multires_steps_low, lr, pixel_samples_k,
            early_stop_patience, early_stop_min_delta,
            tv_weight, guide_mode, contrast_mode, contrast_downscale,
            proj_params, proj_reg_weight,
        )

        # Phase 2: full-res refinement
        proj_params = _optimize_grid_guided(
            grid, rb, ra, coords_full,
            multires_steps_high, lr * 0.5, pixel_samples_k,
            early_stop_patience, early_stop_min_delta,
            tv_weight, guide_mode, contrast_mode, contrast_downscale,
            proj_params, proj_reg_weight,
        )
    else:
        proj_params = _optimize_grid_guided(
            grid, rb, ra, coords_full,
            fit_steps, lr, pixel_samples_k,
            early_stop_patience, early_stop_min_delta,
            tv_weight, guide_mode, contrast_mode, contrast_downscale,
            proj_params, proj_reg_weight,
        )

    grid.eval()
    if proj_params is not None:
        proj_params = proj_params.detach()
    return grid, proj_params


def _optimize_grid_guided(
    grid: BilateralGrid,
    rb: torch.Tensor,
    ra: torch.Tensor,
    coords: torch.Tensor,
    steps: int,
    lr: float,
    pixel_samples_k: int,
    patience: int,
    min_delta: float,
    tv_weight: float,
    guide_mode: str,
    contrast_mode: str,
    contrast_downscale: int,
    proj_params: torch.Tensor,
    proj_reg_weight: float,
) -> torch.Tensor:
    """Internal optimization loop with guide support."""
    params = list(grid.parameters())
    if proj_params is not None:
        params.append(proj_params)

    optimizer = torch.optim.Adam(params, lr=lr)
    H, W, _ = rb.shape
    n_pixels = H * W
    K = min(pixel_samples_k * 1000, n_pixels)

    # Precompute full-res guide (spatial features need full image)
    guide_full = compute_guide(rb, guide_mode, proj_params, contrast_mode, contrast_downscale)
    # For proj modes, we need to recompute guide each step since proj_params change
    # For fixed/luma modes, guide doesn't change so compute once
    needs_guide_recompute = guide_mode.startswith("proj_")

    # Precompute features that don't change
    sat_modes = ("sat", "fixed_ls", "fixed_lsc", "proj_ls", "proj_lsc")
    contrast_modes = ("fixed_lc", "fixed_lsc", "proj_lc", "proj_lsc")
    needs_features = guide_mode in sat_modes or guide_mode in contrast_modes

    luma_full = None
    sat_full = None
    contrast_full = None
    if needs_features or needs_guide_recompute:
        luma_full = compute_luma(rb)
        if guide_mode in sat_modes:
            sat_full = compute_saturation(rb)
        if guide_mode in contrast_modes:
            contrast_full = compute_contrast(rb, contrast_mode, contrast_downscale)

    rb_flat = rb.reshape(-1, 3)
    ra_flat = ra.reshape(-1, 3)
    coords_flat = coords.reshape(-1, 2)

    best_loss = float("inf")
    wait = 0

    # Only backprop through proj_params every N steps to save time
    proj_grad_interval = 3

    for step in range(steps):
        if needs_guide_recompute:
            guide_full = _compute_guide_from_features(
                luma_full, sat_full, contrast_full, guide_mode, proj_params
            )
            # Only let gradients flow to proj_params every N steps
            if step % proj_grad_interval != 0:
                guide_full = guide_full.detach()

        guide_flat = guide_full.reshape(-1)

        if K < n_pixels:
            idx = torch.randint(0, n_pixels, (K,), device=rb.device)
            rb_batch = rb_flat[idx]
            ra_batch = ra_flat[idx]
            coords_batch = coords_flat[idx]
            guide_batch = guide_flat[idx]
        else:
            rb_batch = rb_flat
            ra_batch = ra_flat
            coords_batch = coords_flat
            guide_batch = guide_flat

        optimizer.zero_grad()
        pred = grid(rb_batch, coords_batch, guide_batch)
        mse = F.mse_loss(pred, ra_batch)
        tv = grid.tv_loss()
        loss = mse + tv_weight * tv

        # L2 regularization on projection params
        if proj_params is not None and proj_reg_weight > 0:
            loss = loss + proj_reg_weight * proj_params.pow(2).sum()

        loss.backward()
        optimizer.step()

        loss_val = mse.item()
        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                break

    return proj_params


def _compute_guide_from_features(
    luma: torch.Tensor,
    sat: torch.Tensor,
    contrast: torch.Tensor,
    guide_mode: str,
    proj_params: torch.Tensor,
) -> torch.Tensor:
    """Compute guide from precomputed features (avoids recomputing spatial ops)."""
    if guide_mode == "sat":
        return sat.clamp(0, 1)
    elif guide_mode == "fixed_ls":
        return (0.7 * luma + 0.3 * sat).clamp(0, 1)
    elif guide_mode == "fixed_lc":
        c = contrast if contrast is not None else torch.zeros_like(luma)
        return (0.7 * luma + 0.3 * c).clamp(0, 1)
    elif guide_mode == "fixed_lsc":
        c = contrast if contrast is not None else torch.zeros_like(luma)
        return (0.6 * luma + 0.2 * sat + 0.2 * c).clamp(0, 1)
    elif guide_mode == "proj_ls":
        a, b, d = proj_params[0], proj_params[1], proj_params[2]
        return torch.sigmoid(a * luma + b * sat + d)
    elif guide_mode == "proj_lc":
        a, b, d = proj_params[0], proj_params[1], proj_params[2]
        return torch.sigmoid(a * luma + b * contrast + d)
    elif guide_mode == "proj_lsc":
        a, b, c, d = proj_params[0], proj_params[1], proj_params[2], proj_params[3]
        return torch.sigmoid(a * luma + b * sat + c * contrast + d)
    else:
        return luma.clamp(0, 1)
