#!/usr/bin/env bash
# =============================================================================
# Pipeline verification: creates dummy data and tests all import chains.
# Run from the github/ directory (or inside Docker container at /workspace).
# =============================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DUMMY="test_dummy"
mkdir -p "${DUMMY}/sample1"

echo "=== Step 1: Create dummy test images ==="
python - << 'PYEOF'
import numpy as np, cv2, os
d = "test_dummy/sample1"
for name in ["sample1_before.jpg", "sample1_after.jpg", "sample1_input.jpg"]:
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(d, name), img)
print("  Dummy images created.")
PYEOF

echo "=== Step 2: Test INR import chain ==="
python -c "
from optimize_inr import fit_inretouch, apply_inretouch
from models.inr import INRetouch, make_coord_grid
print('  optimize_inr + models.inr: OK')
"

echo "=== Step 3: Test BG+INR import chain ==="
python -c "
from models.bilateral_grid import BilateralGrid, fit_bilateral_grid, compute_guide, make_coord_grid
print('  models.bilateral_grid: OK')
"

echo "=== Step 4: Test DeepLPF import chain ==="
python -c "
from models.deeplpf import ParametricFilterModel
print('  models.deeplpf: OK')
"

echo "=== Step 5: Test blend_ensemble import ==="
python -c "
import blend_ensemble
print('  blend_ensemble: OK')
"

echo "=== Step 6: Quick INR inference (1 sample, CPU) ==="
python infer_inr.py \
    --root . \
    --variant inr_deep_hflip \
    --input_root "${DUMMY}" \
    --out_dir "${DUMMY}/out_inr" \
    --device cpu 2>&1 | tail -2

if [ -f "${DUMMY}/out_inr/sample1_output.png" ]; then
    echo "  INR output: OK"
else
    echo "  INR output: FAILED"
    exit 1
fi

echo "=== Step 7: Verify checkpoint path ==="
if [ -f "checkpoints/deeplpf.pt" ]; then
    echo "  checkpoints/deeplpf.pt: OK"
else
    echo "  checkpoints/deeplpf.pt: MISSING"
    exit 1
fi

echo "=== Step 8: Quick DeepLPF inference (1 sample, CPU) ==="
python infer_deeplpf.py \
    --root . \
    --ckpt checkpoints/deeplpf.pt \
    --input_root "${DUMMY}" \
    --out_dir "${DUMMY}/out_deeplpf" \
    --device cpu --tta 2>&1 | tail -2

if [ -f "${DUMMY}/out_deeplpf/sample1_output.png" ]; then
    echo "  DeepLPF output: OK"
else
    echo "  DeepLPF output: FAILED"
    exit 1
fi

echo ""
echo "=== ALL TESTS PASSED ==="
rm -rf "${DUMMY}"
