#!/usr/bin/env bash
# =============================================================================
# Heterogeneous 11-Way Ensemble Inference
#
# Usage:
#   bash run_inference.sh <input_dir> <output_zip> [--gpus 0,1]
#
# Example:
#   bash run_inference.sh dataset/Automatic_Evaluation_Data submissions/output.zip --gpus 0,1
# =============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_zip> [--gpus 0,1]"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_ZIP="$2"
GPUS="0,1"

shift 2
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

COMP_DIR="outputs/_components"
BLEND_DIR="outputs/_blended"
LOG_DIR="logs"
mkdir -p "$COMP_DIR" "$BLEND_DIR" "$LOG_DIR" "$(dirname "$OUTPUT_ZIP")"

echo "================================================================="
echo "  Heterogeneous 11-Way Ensemble"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_ZIP"
echo "  GPUs:   ${GPU_LIST[*]} ($N_GPUS total)"
echo "  Started: $(date)"
echo "================================================================="

run_inr() {
    local variant=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] GPU${gpu} ${variant}"
    CUDA_VISIBLE_DEVICES=${gpu} CUDA_LAUNCH_BLOCKING=1 \
        python infer_inr.py \
        --root . \
        --variant "$variant" \
        --input_root "$INPUT_DIR" \
        --out_dir "${COMP_DIR}/${variant}" \
        --device cuda --resume \
        >> "${LOG_DIR}/${variant}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU${gpu} ${variant} done"
}

run_bg_inr() {
    local variant=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] GPU${gpu} ${variant}"
    CUDA_VISIBLE_DEVICES=${gpu} CUDA_LAUNCH_BLOCKING=1 \
        python infer_bg_inr.py \
        --root . \
        --variant "$variant" \
        --input_root "$INPUT_DIR" \
        --out_dir "${COMP_DIR}/${variant}" \
        --device cuda \
        >> "${LOG_DIR}/${variant}.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU${gpu} ${variant} done"
}

run_deeplpf() {
    local gpu=$1
    echo "[$(date +%H:%M:%S)] GPU${gpu} deeplpf"
    CUDA_VISIBLE_DEVICES=${gpu} CUDA_LAUNCH_BLOCKING=1 \
        python infer_deeplpf.py \
        --root . \
        --ckpt checkpoints/deeplpf.pt \
        --input_root "$INPUT_DIR" \
        --out_dir "${COMP_DIR}/deeplpf" \
        --device cuda --tta \
        >> "${LOG_DIR}/deeplpf.log" 2>&1
    echo "[$(date +%H:%M:%S)] GPU${gpu} deeplpf done"
}

echo ""
echo "--- Phase 1: Component generation ---"

if [ "$N_GPUS" -ge 2 ]; then
    (
        run_inr inr_strong_hflip "${GPU_LIST[0]}"
        run_inr inr_strong_vflip "${GPU_LIST[0]}"
        run_inr inr_siren_vflip "${GPU_LIST[0]}"
        run_bg_inr bg_inr_hflip "${GPU_LIST[0]}"
        run_deeplpf "${GPU_LIST[0]}"
        run_inr inr_deep_hflip "${GPU_LIST[0]}"
    ) &
    PID0=$!
    (
        run_inr inr_deep_vflip_s7 "${GPU_LIST[1]}"
        run_inr inr_siren_hflip_strong "${GPU_LIST[1]}"
        run_inr inr_deep_vflip_s0 "${GPU_LIST[1]}"
        run_inr inr_deep_vflip "${GPU_LIST[1]}"
        run_bg_inr bg_inr_vflip "${GPU_LIST[1]}"
    ) &
    PID1=$!
    wait $PID0 $PID1
else
    for v in inr_strong_hflip inr_strong_vflip inr_siren_vflip \
             inr_deep_vflip_s7 inr_siren_hflip_strong inr_deep_vflip_s0 \
             inr_deep_vflip inr_deep_hflip; do
        run_inr "$v" "${GPU_LIST[0]}"
    done
    run_bg_inr bg_inr_hflip "${GPU_LIST[0]}"
    run_bg_inr bg_inr_vflip "${GPU_LIST[0]}"
    run_deeplpf "${GPU_LIST[0]}"
fi

echo ""
echo "--- Phase 2: Blend & zip ---"

python blend_ensemble.py \
    --components_dir "$COMP_DIR" \
    --out_dir "$BLEND_DIR" \
    --zip_path "$OUTPUT_ZIP"

echo ""
echo "================================================================="
echo "  Done: $(date)"
echo "  Submission: $OUTPUT_ZIP"
echo "================================================================="
