"""Weighted ensemble blending of component outputs.

Reads pre-generated PNG outputs from each component directory,
computes a fixed weighted average in BGR space, and saves the result.

Usage:
    python blend_ensemble.py \
        --components_dir outputs/components \
        --out_dir outputs/ensemble \
        --zip_path submissions/submission.zip
"""
import argparse
import os
import cv2
import numpy as np
import zipfile


# 11-way heterogeneous ensemble configuration
# Maps component name -> (output subdirectory, weight)
ENSEMBLE_11WAY = {
    "inr_deep_hflip":          5,
    "inr_strong_hflip":       10,
    "inr_deep_vflip":          5,
    "inr_strong_vflip":        8,
    "inr_siren_vflip":        15,
    "inr_siren_hflip_strong":  5,
    "deeplpf":                10,
    "bg_inr_hflip":            7,
    "bg_inr_vflip":            5,
    "inr_deep_vflip_s0":       5,
    "inr_deep_vflip_s7":      15,
}


def blend_weighted(imgs, weights):
    result = np.zeros_like(imgs[0], dtype=np.float64)
    for img, w in zip(imgs, weights):
        result += img.astype(np.float64) * w
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Weighted ensemble blending")
    parser.add_argument("--components_dir", required=True,
                        help="Base directory containing per-component subdirs")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for blended images")
    parser.add_argument("--zip_path", default=None,
                        help="Optional: create submission zip at this path")
    args = parser.parse_args()

    config = ENSEMBLE_11WAY
    total_w = sum(config.values())
    comp_names = list(config.keys())
    weights = [config[k] / total_w for k in comp_names]
    comp_dirs = [os.path.join(args.components_dir, k) for k in comp_names]

    # Verify all components exist
    for name, d in zip(comp_names, comp_dirs):
        if not os.path.isdir(d):
            print(f"ERROR: missing component directory: {d}")
            return
        count = len([f for f in os.listdir(d) if f.endswith("_output.png")])
        print(f"  {name}: {count} files (weight={config[name]})")

    os.makedirs(args.out_dir, exist_ok=True)

    # Get file list from first component
    files = sorted([f for f in os.listdir(comp_dirs[0]) if f.endswith("_output.png")])
    print(f"\nBlending {len(files)} samples from {len(comp_names)} components...")

    for fname in files:
        imgs = [cv2.imread(os.path.join(d, fname), cv2.IMREAD_COLOR) for d in comp_dirs]
        imgs = [i for i in imgs if i is not None]
        if len(imgs) != len(comp_names):
            continue
        out = blend_weighted(imgs, weights)
        cv2.imwrite(os.path.join(args.out_dir, fname), out)

    # Create zip if requested
    if args.zip_path:
        os.makedirs(os.path.dirname(args.zip_path) or ".", exist_ok=True)
        with zipfile.ZipFile(args.zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for fname in sorted(os.listdir(args.out_dir)):
                if fname.endswith("_output.png"):
                    zf.write(os.path.join(args.out_dir, fname), fname)
        size_mb = os.path.getsize(args.zip_path) / 1024 / 1024
        print(f"\nCreated {args.zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
