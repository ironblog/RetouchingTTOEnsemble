"""Two-stage Bilateral Grid + INR residual inference with TTA.

Pipeline per sample:
1. Fit BG(ref_before -> ref_after) — global tone mapping (~2s)
2. Compute BG(ref_before) and BG(input)
3. Fit INR: BG(ref_before) -> ref_after (learns residual on top of BG)
4. Apply: output = INR(BG(input))
5. Average with TTA-augmented prediction

Usage:
    python infer_bg_inr.py --root . \
        --variant bg_inr_hflip \
        --input_root dataset/Automatic_Evaluation_Data \
        --out_dir outputs/bg_inr_hflip \
        --device cuda
"""

import argparse
import os
import re
import time

import cv2
import numpy as np
import torch

from optimize_inr import fit_inretouch, apply_inretouch
from models.bilateral_grid import (
    BilateralGrid, fit_bilateral_grid, compute_guide, make_coord_grid
)


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_png(img, path):
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def apply_bg_model(bg_model, img_t, device="cuda"):
    """Apply a fitted BG model to an image."""
    H, W = img_t.shape[:2]
    img_d = img_t.to(device)
    coords = make_coord_grid(H, W, device)
    guide = compute_guide(img_d)
    with torch.no_grad():
        out = bg_model(img_d, coords, guide)
    return out.clamp(0, 1)


VARIANT_CONFIGS = {
    # BG + INR(m=2) residual + hflip TTA
    "bg_inr_hflip": dict(n_hidden_m=2, tta="hflip"),
    # BG + INR(m=2) residual + vflip TTA
    "bg_inr_vflip": dict(n_hidden_m=2, tta="vflip"),
}


def _fit_and_apply_inr(base_rb_t, ra_t, base_inp_t, device, seed=42, **kwargs):
    """Fit INR on BG(rb)->ra residual, apply to BG(input)."""
    torch.manual_seed(seed)

    fit_kwargs = dict(
        n_neurons=64,
        n_hidden_p=kwargs.get("n_hidden_p", 1),
        n_hidden_s=kwargs.get("n_hidden_s", 1),
        n_hidden_m=kwargs.get("n_hidden_m", 1),
        sin_w=kwargs.get("sin_w", 1.0),
        siren_init=kwargs.get("siren_init", False),
        steps=kwargs.get("steps", 1000),
        lr=kwargs.get("lr", 1e-2),
        lr_min=1e-4,
        window_size=kwargs.get("window_size", 12),
        batch_windows=kwargs.get("batch_windows", 484),
        weight_decay_p=0.1,
        weight_decay_s=0.0001,
        weight_decay_m=0.001,
        betas=kwargs.get("betas", (0.9, 0.9)),
        grad_clip=0.01,
        early_stop_patience=100,
    )

    model, fit_info = fit_inretouch(
        base_rb_t, ra_t, base_t=base_rb_t, device=device, **fit_kwargs)

    with torch.no_grad():
        out = apply_inretouch(model, base_inp_t, base_t=base_inp_t, device=device)
    return out.cpu().numpy(), fit_info


def infer_single(rb, ra, inp, device, variant="bg_inr_hflip"):
    """Run two-stage BG+INR inference with TTA on a single sample."""
    t0 = time.time()
    cfg = VARIANT_CONFIGS[variant].copy()
    tta = cfg.pop("tta", "none")
    info = {"variant": variant, "tta": tta}

    if rb.shape != ra.shape:
        ra = cv2.resize(ra, (rb.shape[1], rb.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    rb_t = torch.from_numpy(rb).float()
    ra_t = torch.from_numpy(ra).float()
    inp_t = torch.from_numpy(inp).float()

    # Stage 1: Fit bilateral grid
    bg_model, proj = fit_bilateral_grid(
        rb_t, ra_t, device=device,
        spatial_h=4, spatial_w=4, luma_bins=12,
        tv_weight=0.02, fit_steps=500, lr=0.02)

    # Stage 2: Apply BG to get base images
    bg_rb = apply_bg_model(bg_model, rb_t, device)
    bg_inp = apply_bg_model(bg_model, inp_t, device)
    bg_rb_cpu = bg_rb.cpu()
    bg_inp_cpu = bg_inp.cpu()

    # Stage 3: Fit INR on BG(rb) -> ra residual, apply to BG(input)
    out1, fit_info1 = _fit_and_apply_inr(
        bg_rb_cpu, ra_t, bg_inp_cpu, device, seed=42, **cfg)
    info["psnr_ref"] = fit_info1["psnr_ref"]
    info["params"] = fit_info1["params"]
    info["actual_steps"] = fit_info1["actual_steps"]

    outputs = [out1]

    if tta == "hflip":
        bg_rb_hflip = torch.flip(bg_rb_cpu, [1])
        ra_hflip = torch.flip(ra_t, [1])
        bg_inp_hflip = torch.flip(bg_inp_cpu, [1])
        out_hflip, _ = _fit_and_apply_inr(
            bg_rb_hflip, ra_hflip, bg_inp_hflip, device, seed=42, **cfg)
        out2 = np.flip(out_hflip, axis=1).copy()
        outputs.append(out2)

    if tta == "vflip":
        bg_rb_vflip = torch.flip(bg_rb_cpu, [0])
        ra_vflip = torch.flip(ra_t, [0])
        bg_inp_vflip = torch.flip(bg_inp_cpu, [0])
        out_vflip, _ = _fit_and_apply_inr(
            bg_rb_vflip, ra_vflip, bg_inp_vflip, device, seed=42, **cfg)
        out_v = np.flip(out_vflip, axis=0).copy()
        outputs.append(out_v)

    output = np.mean(outputs, axis=0)
    output = np.clip(output, 0, 1)
    info["n_tta"] = len(outputs)
    info["time"] = time.time() - t0
    torch.cuda.empty_cache()
    return output, info


def run_competition(args):
    os.makedirs(args.out_dir, exist_ok=True)
    input_root = args.input_root
    if not os.path.isabs(input_root):
        input_root = os.path.join(args.root, input_root)

    sample_dirs = []
    for fn in os.listdir(input_root):
        m = re.match(r"^sample(\d+)$", fn)
        if m:
            sample_dirs.append((int(m.group(1)), fn))
    sample_dirs.sort()

    n = len(sample_dirs)
    print(f"BG+INR ({args.variant}) Competition: {n} samples")

    refs = []
    for i, (num, name) in enumerate(sample_dirs):
        sd = os.path.join(input_root, name)
        rb = read_rgb(os.path.join(sd, f"sample{num}_before.jpg"))
        ra = read_rgb(os.path.join(sd, f"sample{num}_after.jpg"))
        inp = read_rgb(os.path.join(sd, f"sample{num}_input.jpg"))

        out, info = infer_single(rb, ra, inp, device=args.device,
                                  variant=args.variant)
        refs.append(info.get("psnr_ref", 0))
        save_png(out, os.path.join(args.out_dir, f"sample{num}_output.png"))

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] ref={info.get('psnr_ref',0):.2f}"
                  f" (avg_ref={np.mean(refs):.2f}) ({info['time']:.1f}s)"
                  f" tta={info['n_tta']}x",
                  flush=True)

    print(f"\n  PSNR_ref: {np.mean(refs):.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Two-stage BG+INR inference")
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=["competition"], default="competition")
    parser.add_argument("--variant",
                        choices=list(VARIANT_CONFIGS.keys()),
                        default="bg_inr_hflip")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(args.root, args.out_dir)

    run_competition(args)


if __name__ == "__main__":
    main()
