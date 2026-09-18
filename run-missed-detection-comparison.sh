#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$PROJECT_ROOT/artifacts}"
RUNS_ROOT="${RUNS_ROOT:-$ARTIFACTS_ROOT/runs}"
TRAINING_ROOT="${TRAINING_ROOT:-$ARTIFACTS_ROOT/training-data}"
COMPARISONS_ROOT="${COMPARISONS_ROOT:-$ARTIFACTS_ROOT/missed-comparisons}"
YOLO_VENV="${YOLO_VENV:-$PROJECT_ROOT/.venv}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-$PROJECT_ROOT/install.sh}"
COMPARISON_SCRIPT="${COMPARISON_SCRIPT:-$PROJECT_ROOT/scripts/compare_missed_detections.py}"

CONF="${CONF:-0.25}"
MATCH_IOU="${MATCH_IOU:-0.50}"
NMS_IOU="${NMS_IOU:-0.70}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-}"

cd "$PROJECT_ROOT"

newest_file() {
    local search_root="$1"
    local filename="$2"
    local newest=""
    local candidate

    [[ -d "$search_root" ]] || return 1
    while IFS= read -r -d '' candidate; do
        if [[ -z "$newest" || "$candidate" -nt "$newest" ]]; then
            newest="$candidate"
        fi
    done < <(find "$search_root" -type f -name "$filename" -print0)

    [[ -n "$newest" ]] || return 1
    printf '%s\n' "$newest"
}

if [[ ! -x "$YOLO_VENV/bin/python" || ! -x "$YOLO_VENV/bin/yolo" ]]; then
    YOLO_VENV="$YOLO_VENV" "$INSTALL_SCRIPT"
fi
if [[ ! -f "$COMPARISON_SCRIPT" ]]; then
    printf 'Error: comparison helper not found: %s\n' "$COMPARISON_SCRIPT" >&2
    exit 1
fi

if [[ -n "${MODEL:-}" ]]; then
    model_path="$MODEL"
else
    if ! model_path="$(newest_file "$RUNS_ROOT" best.pt)"; then
        printf 'Error: no best.pt found below %s\n' "$RUNS_ROOT" >&2
        printf 'Set MODEL to select one explicitly.\n' >&2
        exit 1
    fi
fi

if [[ -n "${DATASET:-}" ]]; then
    dataset_dir="$DATASET"
else
    if ! data_yaml="$(newest_file "$TRAINING_ROOT" data.yaml)"; then
        printf 'Error: no prepared dataset found below %s\n' "$TRAINING_ROOT" >&2
        printf 'Set DATASET to select one explicitly.\n' >&2
        exit 1
    fi
    dataset_dir="$(dirname -- "$data_yaml")"
fi

[[ "$model_path" = /* ]] || model_path="$PROJECT_ROOT/$model_path"
[[ "$dataset_dir" = /* ]] || dataset_dir="$PROJECT_ROOT/$dataset_dir"

if [[ ! -f "$model_path" ]]; then
    printf 'Error: model does not exist: %s\n' "$model_path" >&2
    exit 1
fi
if [[ ! -f "$dataset_dir/data.yaml" ]]; then
    printf 'Error: dataset has no data.yaml: %s\n' "$dataset_dir" >&2
    exit 1
fi

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
output_dir="${OUTPUT_DIR:-$COMPARISONS_ROOT/comparison-$timestamp}"
mkdir -p "$output_dir"

printf 'Model:   %s\n' "$model_path"
printf 'Dataset: %s\n' "$dataset_dir"
printf 'Output:  %s\n' "$output_dir"

arguments=(
    "$COMPARISON_SCRIPT"
    --model "$model_path"
    --dataset "$dataset_dir"
    --output "$output_dir"
    --conf "$CONF"
    --match-iou "$MATCH_IOU"
    --nms-iou "$NMS_IOU"
    --imgsz "$IMGSZ"
)
if [[ -n "$DEVICE" ]]; then
    arguments+=(--device "$DEVICE")
fi

"$YOLO_VENV/bin/python" "${arguments[@]}"

printf '\nComparison complete.\n'
printf 'Review examples: %s\n' "$output_dir"
printf 'Summary:         %s/summary.csv\n' "$output_dir"
