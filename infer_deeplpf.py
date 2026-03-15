"""DeepLPF-style parametric filter inference with test-time adaptation.

Pre-trained encoder predicts parameters for interpretable photographic filters
(tone curves, graduated filter, radial vignette). At test time, the encoder
is optionally fine-tuned on the reference pair for improved per-sample accuracy.

Usage:
    python infer_deeplpf.py --root . \
        --ckpt checkpoints/deeplpf.pt \
        --input_root dataset/Automatic_Evaluation_Data \
        --out_dir outputs/deeplpf \
        --device cuda --tta
"""

import argparse
import os
import re
import time

import cv2
import numpy as np
import torch

from models.deeplpf import ParametricFilterModel


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


def tta_params(model, rb_chw, ra_chw, device, steps=40, lr=0.05):
    """Fine-tune encoder and parameter head on the reference pair."""
    tta_params_list = []
    for name, param in model.named_parameters():
        if "ref_encoder" in name or "param_head" in name:
            param.requires_grad_(True)
            tta_params_list.append(param)
        else:
            param.requires_grad_(False)

    if not tta_params_list:
        return 0.0

    opt = torch.optim.Adam(tta_params_list, lr=lr)

    rb_batch = rb_chw.unsqueeze(0)
    ra_batch = ra_chw.unsqueeze(0)

    best_loss = float("inf")
    best_state = {n: p.data.clone() for n, p in model.named_parameters()
                  if p.requires_grad}

    for step in range(steps):
        output, _ = model(rb_batch, rb_batch, ra_batch)
        loss = (output - ra_batch).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {n: p.data.clone() for n, p in model.named_parameters()
                          if p.requires_grad}

    for n, p in model.named_parameters():
        if n in best_state:
            p.data.copy_(best_state[n])

    for p in model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        recon, _ = model(rb_batch, rb_batch, ra_batch)
        mse = (recon - ra_batch).pow(2).mean().item()
        psnr_ref = -10.0 * np.log10(mse + 1e-10)

    return psnr_ref


def infer_single(model, rb, ra, inp, device, tta=False, tta_steps=40, tta_lr=0.05):
    H, W = inp.shape[:2]
    ref_size = model.ref_size

    rb_rsz = cv2.resize(rb, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    ra_rsz = cv2.resize(ra, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    rb_chw = torch.from_numpy(rb_rsz.transpose(2, 0, 1)).float().to(device)
    ra_chw = torch.from_numpy(ra_rsz.transpose(2, 0, 1)).float().to(device)
    inp_chw = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(device)

    with torch.no_grad():
        rb_batch = rb_chw.unsqueeze(0)
        ra_batch = ra_chw.unsqueeze(0)
        recon_base, _ = model(rb_batch, rb_batch, ra_batch)
        mse_base = (recon_base - ra_batch).pow(2).mean().item()
        psnr_ref_base = -10.0 * np.log10(mse_base + 1e-10)

    psnr_ref_tta = psnr_ref_base
    if tta:
        orig_state = {n: p.data.clone() for n, p in model.named_parameters()}
        psnr_ref_tta = tta_params(model, rb_chw, ra_chw, device,
                                  steps=tta_steps, lr=tta_lr)
        if psnr_ref_tta < psnr_ref_base:
            for n, p in model.named_parameters():
                p.data.copy_(orig_state[n])
            psnr_ref_tta = psnr_ref_base

    with torch.no_grad():
        inp_batch = inp_chw.unsqueeze(0)
        rb_batch = rb_chw.unsqueeze(0)
        ra_batch = ra_chw.unsqueeze(0)
        output, _ = model(inp_batch, rb_batch, ra_batch)
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np, 0, 1)

    if tta:
        for n, p in model.named_parameters():
            p.data.copy_(orig_state[n])

    torch.cuda.empty_cache()

    return output_np, {
        "psnr_ref_base": psnr_ref_base,
        "psnr_ref_tta": psnr_ref_tta,
    }


def run_competition(args, model):
    os.makedirs(args.out_dir, exist_ok=True)
    input_root = args.input_root
    if not os.path.isabs(input_root):
        input_root = os.path.join(args.root, input_root)

    samples = []
    for fn in os.listdir(input_root):
        m = re.match(r"^sample(\d+)$", fn)
        if m:
            samples.append((int(m.group(1)), fn))
    samples.sort()

    n = len(samples)
    print(f"DeepLPF Competition: {n} samples")

    refs = []
    for i, (num, name) in enumerate(samples):
        t0 = time.time()
        sd = os.path.join(input_root, name)
        rb = read_rgb(os.path.join(sd, f"sample{num}_before.jpg"))
        ra = read_rgb(os.path.join(sd, f"sample{num}_after.jpg"))
        inp = read_rgb(os.path.join(sd, f"sample{num}_input.jpg"))

        out, info = infer_single(model, rb, ra, inp, args.device,
                                 tta=args.tta, tta_steps=args.tta_steps,
                                 tta_lr=args.tta_lr)
        elapsed = time.time() - t0
        refs.append(info["psnr_ref_tta"])
        save_png(out, os.path.join(args.out_dir, f"sample{num}_output.png"))

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] ref={info['psnr_ref_tta']:.2f} "
                  f"(avg={np.mean(refs):.2f})", flush=True)

    print(f"\n  PSNR_ref: {np.mean(refs):.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="DeepLPF parametric filter inference")
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=["competition"], default="competition")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta_steps", type=int, default=40)
    parser.add_argument("--tta_lr", type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(args.root, args.out_dir)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = ParametricFilterModel(
        n_curve_points=cfg.get("n_curve_points", 17),
        n_grad_filters=cfg.get("n_grad_filters", 2),
        style_dim=cfg.get("style_dim", 128),
        ref_size=cfg.get("ref_size", 256),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    run_competition(args, model)


if __name__ == "__main__":
    main()
