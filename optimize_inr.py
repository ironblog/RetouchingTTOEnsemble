"""Per-sample INR optimization on a single reference pair.

Optimizes an INRetouch model from scratch to learn the mapping
ref_before -> ref_after, then applies the learned transform to a new image.

Key design choices (following Kinli et al., 2024):
- L1 loss
- Separate AdamW optimizers per branch with different weight_decay
- Gradient clipping at 0.01
- Cosine annealing LR schedule
- Random window sampling for memory-efficient training

Usage:
    model, info = fit_inretouch(rb_t, ra_t, device="cuda")
    output = apply_inretouch(model, inp_t, device="cuda")
"""

import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.inr import INRetouch, make_coord_grid


def _sample_windows(H, W, window_size, n_windows, device):
    """Sample random window positions.

    Returns:
        ys, xs: (n_windows,) int tensors of top-left corners
    """
    max_y = H - window_size
    max_x = W - window_size
    ys = torch.randint(0, max_y + 1, (n_windows,), device=device)
    xs = torch.randint(0, max_x + 1, (n_windows,), device=device)
    return ys, xs


def _extract_windows(img, ys, xs, window_size):
    """Extract windows from (1, C, H, W) tensor.

    Returns:
        (K, C, n, n) patches
    """
    n = window_size
    K = len(ys)
    src = img.squeeze(0)  # (C, H, W)
    dy = torch.arange(n, device=img.device)
    dx = torch.arange(n, device=img.device)
    row_idx = (ys.unsqueeze(1) + dy.unsqueeze(0)).unsqueeze(2).expand(K, n, n)
    col_idx = (xs.unsqueeze(1) + dx.unsqueeze(0)).unsqueeze(1).expand(K, n, n)
    patches = src[:, row_idx, col_idx]  # (C, K, n, n)
    return patches.permute(1, 0, 2, 3)  # (K, C, n, n)


def fit_inretouch(
    rb_t, ra_t,
    base_t=None,
    n_neurons=64,
    n_hidden_p=1,
    n_hidden_s=1,
    n_hidden_m=1,
    sin_w=1.0,
    siren_init=False,
    steps=1000,
    lr=1e-2,
    lr_min=1e-4,
    window_size=12,
    batch_windows=484,
    weight_decay_p=0.1,
    weight_decay_s=0.0001,
    weight_decay_m=0.001,
    betas=(0.9, 0.9),
    grad_clip=0.01,
    early_stop_patience=100,
    loss_type="l1",
    device="cuda",
):
    """Fit INRetouch on a single reference pair.

    Args:
        rb_t: (H, W, 3) ref before, float [0, 1]
        ra_t: (H, W, 3) ref after, float [0, 1]
        base_t: (H, W, 3) optional base image for skip (default: rb_t)
        steps: optimization iterations
        lr: initial learning rate
        lr_min: minimum learning rate for cosine annealing
        window_size: patch size
        batch_windows: windows per iteration
        weight_decay_p/s/m: per-branch weight decay
        grad_clip: gradient clipping norm

    Returns:
        (model, info_dict)
    """
    t0 = time.time()
    H, W = rb_t.shape[:2]
    window_size = min(window_size, H, W)

    if base_t is None:
        base_t = rb_t

    # Prepare data as (1, C, H, W)
    base_chw = base_t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    ra_chw = ra_t.permute(2, 0, 1).unsqueeze(0).to(device)

    # Coordinate grid in [-1, 1]
    coords = make_coord_grid(H, W, device)  # (1, 2, H, W)

    # Full input: [coords, RGB]
    inp_full = torch.cat([coords, base_chw], dim=1)  # (1, 5, H, W)

    # Model
    model = INRetouch(
        n_neurons=n_neurons,
        n_hidden_p=n_hidden_p,
        n_hidden_s=n_hidden_s,
        n_hidden_m=n_hidden_m,
        sin_w=sin_w,
        siren_init=siren_init,
    ).to(device)

    # Separate AdamW optimizers per branch (faithful to paper)
    # Critical: heavy weight_decay on position branch (0.1) prevents spatial overfitting
    opt_p = torch.optim.AdamW(model.model_p.parameters(), lr=lr, weight_decay=weight_decay_p, betas=betas)
    opt_s = torch.optim.AdamW(model.model_s.parameters(), lr=lr, weight_decay=weight_decay_s, betas=betas)
    opt_m = torch.optim.AdamW(model.model_m.parameters(), lr=lr, weight_decay=weight_decay_m, betas=betas)

    # Cosine annealing for each
    sch_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=steps, eta_min=lr_min)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=steps, eta_min=lr_min)
    sch_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=steps, eta_min=lr_min)

    best_loss = float("inf")
    best_state = None
    wait = 0
    actual_steps = 0

    model.train()
    for step in range(steps):
        actual_steps = step + 1
        opt_p.zero_grad()
        opt_s.zero_grad()
        opt_m.zero_grad()

        # Sample windows
        ys, xs = _sample_windows(H, W, window_size, batch_windows, device)
        inp_patches = _extract_windows(inp_full, ys, xs, window_size)   # (K, 5, n, n)
        base_patches = _extract_windows(base_chw, ys, xs, window_size)  # (K, 3, n, n)
        tgt_patches = _extract_windows(ra_chw, ys, xs, window_size)     # (K, 3, n, n)

        # Forward
        pred = model(inp_patches)  # (K, 3, n, n) — includes skip

        # Loss
        if loss_type == "l2":
            loss = F.mse_loss(pred, tgt_patches)
        else:
            loss = F.l1_loss(pred, tgt_patches)

        loss.backward()

        # Gradient clipping (paper: 0.01)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt_p.step()
        opt_s.step()
        opt_m.step()
        sch_p.step()
        sch_s.step()
        sch_m.step()

        loss_val = loss.item()
        if loss_val < best_loss - 1e-7:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if early_stop_patience > 0 and wait >= early_stop_patience:
                break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Compute PSNR_ref on full image
    with torch.no_grad():
        inp_eval = torch.cat([coords, base_chw], dim=1)
        pred_full = model(inp_eval)  # (1, 3, H, W)
        mse = F.mse_loss(pred_full, ra_chw).item()
        psnr_ref = -10 * math.log10(mse + 1e-10)

    return model, {
        "psnr_ref": psnr_ref,
        "best_loss": best_loss,
        "actual_steps": actual_steps,
        "params": model.param_count(),
        "time": time.time() - t0,
    }


@torch.no_grad()
def apply_inretouch(model, img_t, base_t=None, device="cuda"):
    """Apply trained INRetouch to a new image.

    Args:
        model: trained INRetouch model
        img_t: (H, W, 3) input image
        base_t: (H, W, 3) optional base for skip (default: img_t)

    Returns:
        (H, W, 3) output tensor [0, 1]
    """
    H, W = img_t.shape[:2]
    if base_t is None:
        base_t = img_t

    base_chw = base_t.permute(2, 0, 1).unsqueeze(0).to(device)
    coords = make_coord_grid(H, W, device)
    inp = torch.cat([coords, base_chw], dim=1)

    pred = model(inp)  # (1, 3, H, W)
    return pred.squeeze(0).permute(1, 2, 0).clamp(0, 1)
