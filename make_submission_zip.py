"""Create submission zip with images at the top level (no folders).

Usage:
    python tools/make_submission_zip.py --pred_dir outputs\\dev_predictions --zip_path submissions\\Submission.zip
"""

import argparse
import os
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Make submission zip")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing predicted PNG files")
    parser.add_argument("--zip_path", type=str, required=True,
                        help="Output zip file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.zip_path) or ".", exist_ok=True)

    # Collect all image files
    image_files = sorted([
        f for f in os.listdir(args.pred_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not image_files:
        print(f"ERROR: No image files found in {args.pred_dir}")
        return

    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in image_files:
            fpath = os.path.join(args.pred_dir, fname)
            # Add at top level (no folder prefix)
            zf.write(fpath, arcname=fname)

    print(f"Created {args.zip_path} with {len(image_files)} files")
    print(f"  Size: {os.path.getsize(args.zip_path) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
