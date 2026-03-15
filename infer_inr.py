"""INR inference with Test-Time Augmentation (TTA).

Fits an implicit neural representation on the reference pair (before, after),
then applies the learned transform to a new input image. TTA averages
predictions from geometrically augmented inputs to reduce optimization noise.

Usage:
    python infer_inr.py --root . \
        --variant inr_siren_vflip \
        --input_root dataset/Automatic_Evaluation_Data \
        --out_dir outputs/inr_siren_vflip \
        --device cuda --resume
"""

import argparse
import os
import re
import time

import cv2
import numpy as np
import torch

from optimize_inr import fit_inretouch, apply_inretouch


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


# ---------------------------------------------------------------------------
# Variant configurations: architecture + TTA strategy + optimization schedule
# ---------------------------------------------------------------------------
VARIANT_CONFIGS = {
    # Standard INR (m=2, 64 neurons) with hflip TTA
    "inr_deep_hflip": dict(n_hidden_m=2, tta="hflip"),
    # Standard INR (m=2, 64 neurons) with vflip TTA
    "inr_deep_vflip": dict(n_hidden_m=2, tta="vflip"),
    # SIREN-initialized INR (m=1) with vflip TTA
    "inr_siren_vflip": dict(n_hidden_m=1, siren_init=True, betas=(0.9, 0.999), tta="vflip"),
    # SIREN-initialized INR (m=1) with hflip TTA, extended optimization
    "inr_siren_hflip_strong": dict(n_hidden_m=1, siren_init=True, betas=(0.9, 0.999), tta="hflip", steps=1500, batch_windows=784),
    # Standard INR with vflip, seed=0 (decorrelated)
    "inr_deep_vflip_s0": dict(n_hidden_m=2, tta="vflip", base_seed=0),
    # Standard INR with vflip, seed=7 (decorrelated)
    "inr_deep_vflip_s7": dict(n_hidden_m=2, tta="vflip", base_seed=7),
    # Extended optimization: vflip, 1500 steps, 784 windows
    "inr_strong_vflip": dict(n_hidden_m=2, tta="vflip", steps=1500, batch_windows=784),
    # Extended optimization: hflip, 1500 steps, 784 windows
    "inr_strong_hflip": dict(n_hidden_m=2, tta="hflip", steps=1500, batch_windows=784),
}


def _fit_and_apply(rb_t, ra_t, inp_t, device, seed=42, **kwargs):
    """Fit INR on ref pair and apply to input."""
    torch.manual_seed(seed)

    fit_kwargs = dict(
        n_neurons=kwargs.get("n_neurons", 64),
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
        early_stop_patience=kwargs.get("early_stop_patience", 100),
    )

    model, fit_info = fit_inretouch(
        rb_t, ra_t, base_t=rb_t, device=device, **fit_kwargs)

    with torch.no_grad():
        out = apply_inretouch(model, inp_t, base_t=inp_t, device=device)
    return out.cpu().numpy(), fit_info


def infer_single(rb, ra, inp, device, variant="inr_siren_vflip"):
    """Run INR inference with TTA on a single sample."""
    t0 = time.time()
    cfg = VARIANT_CONFIGS[variant].copy()
    tta = cfg.pop("tta", "none")
    info = {"variant": variant, "tta": tta}

    base_seed = cfg.pop("base_seed", 42)

    if rb.shape != ra.shape:
        ra = cv2.resize(ra, (rb.shape[1], rb.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    rb_t = torch.from_numpy(rb).float()
    ra_t = torch.from_numpy(ra).float()
    inp_t = torch.from_numpy(inp).float()

    out1, fit_info1 = _fit_and_apply(rb_t, ra_t, inp_t, device, seed=base_seed, **cfg)
    info["psnr_ref"] = fit_info1["psnr_ref"]
    info["params"] = fit_info1["params"]
    info["actual_steps"] = fit_info1["actual_steps"]

    outputs = [out1]

    if tta in ("hflip", "hvflip"):
        rb_flip = torch.flip(rb_t, [1])
        ra_flip = torch.flip(ra_t, [1])
        inp_flip = torch.flip(inp_t, [1])
        out_flip, _ = _fit_and_apply(rb_flip, ra_flip, inp_flip, device, seed=base_seed, **cfg)
        out2 = np.flip(out_flip, axis=1).copy()
        outputs.append(out2)

    if tta in ("vflip", "hvflip"):
        rb_vflip = torch.flip(rb_t, [0])
        ra_vflip = torch.flip(ra_t, [0])
        inp_vflip = torch.flip(inp_t, [0])
        out_vflip, _ = _fit_and_apply(rb_vflip, ra_vflip, inp_vflip, device, seed=base_seed, **cfg)
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
    print(f"INR ({args.variant}) Competition: {n} samples")

    refs = []
    skipped = 0
    for i, (num, name) in enumerate(sample_dirs):
        out_path = os.path.join(args.out_dir, f"sample{num}_output.png")
        if getattr(args, 'resume', False) and os.path.exists(out_path):
            skipped += 1
            continue

        sd = os.path.join(input_root, name)
        rb = read_rgb(os.path.join(sd, f"sample{num}_before.jpg"))
        ra = read_rgb(os.path.join(sd, f"sample{num}_after.jpg"))
        inp = read_rgb(os.path.join(sd, f"sample{num}_input.jpg"))

        out, info = infer_single(rb, ra, inp, device=args.device,
                                  variant=args.variant)
        refs.append(info.get("psnr_ref", 0))
        save_png(out, out_path)

        if (i + 1 - skipped) % 10 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] ref={info.get('psnr_ref',0):.2f}"
                  f" (avg_ref={np.mean(refs):.2f}) ({info['time']:.1f}s)"
                  f" tta={info['n_tta']}x skipped={skipped}",
                  flush=True)

    if refs:
        print(f"\n  PSNR_ref: {np.mean(refs):.2f} dB (skipped {skipped})")
    else:
        print(f"\n  All {skipped} samples already exist.")


def main():
    parser = argparse.ArgumentParser(description="INR per-sample inference with TTA")
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=["competition"], default="competition")
    parser.add_argument("--variant",
                        choices=list(VARIANT_CONFIGS.keys()),
                        default="inr_siren_vflip")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Skip existing output files")
    args = parser.parse_args()

    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(args.root, args.out_dir)

    run_competition(args)


if __name__ == "__main__":
    main()
