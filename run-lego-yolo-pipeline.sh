#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$PROJECT_ROOT/artifacts}"
EXPORT_ROOT="${EXPORT_ROOT:-$ARTIFACTS_ROOT/exports}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-$SCRIPT_DIR/export-cvat-dataset.sh}"
PREPARE_DATASET_SCRIPT="${PREPARE_DATASET_SCRIPT:-$SCRIPT_DIR/scripts/prepare_yolo_dataset.py}"
REPORT_ACCELERATOR_SCRIPT="${REPORT_ACCELERATOR_SCRIPT:-$SCRIPT_DIR/scripts/report_accelerator.py}"
SOURCE_DATASET="${SOURCE_DATASET:-$EXPORT_ROOT/latest}"
TRAINING_ROOT="${TRAINING_ROOT:-$ARTIFACTS_ROOT/training-data}"
RUNS_ROOT="${RUNS_ROOT:-$ARTIFACTS_ROOT/runs}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-$ARTIFACTS_ROOT/predictions}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ARTIFACTS_ROOT/weights}"
YOLO_VENV="${YOLO_VENV:-$PROJECT_ROOT/.venv}"

MODEL="${MODEL:-$WEIGHTS_ROOT/yolo26n.pt}"
EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:--1}"
DEVICE="${DEVICE:-0}"
PATIENCE="${PATIENCE:-30}"
WORKERS="${WORKERS:-0}"
CACHE="${CACHE:-False}"
VAL_FRACTION="${VAL_FRACTION:-0.20}"
SPLIT_SEED="${SPLIT_SEED:-42}"

# Keep relative paths and third-party downloads anchored to this repository,
# regardless of the directory from which the script was launched.
cd "$PROJECT_ROOT"

for command_name in python3 cp; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Error: required command not found: %s\n' "$command_name" >&2
        exit 1
    fi
done

while true; do
    read -rp 'Export a fresh dataset from CVAT first? [y/n]: ' answer
    case "${answer,,}" in
        y|yes)
            if [[ ! -x "$EXPORT_SCRIPT" ]]; then
                printf 'Error: export script is missing or not executable: %s\n' "$EXPORT_SCRIPT" >&2
                exit 1
            fi
            PROJECT_ROOT="$PROJECT_ROOT" \
                ARTIFACTS_ROOT="$ARTIFACTS_ROOT" \
                OUTPUT_ROOT="$EXPORT_ROOT" \
                "$EXPORT_SCRIPT"
            break
            ;;
        n|no)
            break
            ;;
        *)
            printf 'Please answer y or n.\n'
            ;;
    esac
done

if [[ ! -d "$SOURCE_DATASET/images/train" || ! -d "$SOURCE_DATASET/labels/train" ]]; then
    printf 'Error: no usable CVAT export found at %s\n' "$SOURCE_DATASET" >&2
    printf 'Run the export or set SOURCE_DATASET to an extracted CVAT dataset.\n' >&2
    exit 1
fi

for helper_script in "$PREPARE_DATASET_SCRIPT" "$REPORT_ACCELERATOR_SCRIPT"; do
    if [[ ! -f "$helper_script" ]]; then
        printf 'Error: required helper script not found: %s\n' "$helper_script" >&2
        exit 1
    fi
done

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
dataset_dir="$TRAINING_ROOT/lego-$timestamp"
run_name="lego-$timestamp"

mkdir -p "$TRAINING_ROOT" "$RUNS_ROOT" "$PREDICTIONS_ROOT" "$WEIGHTS_ROOT"
cp -aL "$SOURCE_DATASET" "$dataset_dir"

printf 'Creating a class-preserving train/validation split...\n'
python3 "$PREPARE_DATASET_SCRIPT" \
    "$dataset_dir" \
    --val-fraction "$VAL_FRACTION" \
    --seed "$SPLIT_SEED"

if [[ ! -x "$YOLO_VENV/bin/python" ]]; then
    printf 'Creating YOLO virtual environment at %s...\n' "$YOLO_VENV"
    python3 -m venv "$YOLO_VENV"
    "$YOLO_VENV/bin/python" -m pip install --upgrade pip
    "$YOLO_VENV/bin/python" -m pip install ultralytics
elif [[ ! -x "$YOLO_VENV/bin/yolo" ]]; then
    printf 'Installing Ultralytics in the existing YOLO environment...\n'
    "$YOLO_VENV/bin/python" -m pip install ultralytics
fi

printf 'Checking accelerator availability...\n'
"$YOLO_VENV/bin/python" "$REPORT_ACCELERATOR_SCRIPT"

printf 'Fine-tuning %s...\n' "$MODEL"
"$YOLO_VENV/bin/yolo" detect train \
    model="$MODEL" \
    data="$dataset_dir/data.yaml" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device="$DEVICE" \
    patience="$PATIENCE" \
    workers="$WORKERS" \
    cache="$CACHE" \
    project="$RUNS_ROOT" \
    name="$run_name"

best_model="$RUNS_ROOT/$run_name/weights/best.pt"
if [[ ! -f "$best_model" ]]; then
    printf 'Error: training finished without producing %s\n' "$best_model" >&2
    exit 1
fi

printf 'Generating validation predictions...\n'
"$YOLO_VENV/bin/yolo" detect predict \
    model="$best_model" \
    source="$dataset_dir/images/val" \
    conf=0.25 \
    save=True \
    project="$PREDICTIONS_ROOT" \
    name="$run_name"

printf '\nPipeline complete.\n'
printf 'Dataset:     %s\n' "$dataset_dir"
printf 'Best model:  %s\n' "$best_model"
printf 'Predictions: %s/%s\n' "$PREDICTIONS_ROOT" "$run_name"
