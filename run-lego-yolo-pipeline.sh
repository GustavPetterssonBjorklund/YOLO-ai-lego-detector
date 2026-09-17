#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
EXPORT_DIR="${EXPORT_DIR:-$SCRIPT_DIR}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-$EXPORT_DIR/export-cvat-dataset.sh}"
SOURCE_DATASET="${SOURCE_DATASET:-$EXPORT_DIR/latest}"
TRAINING_ROOT="${TRAINING_ROOT:-$PROJECT_ROOT/training-data}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-$PROJECT_ROOT/predictions}"
YOLO_VENV="${YOLO_VENV:-$PROJECT_ROOT/yolo-venv}"

MODEL="${MODEL:-yolo26n.pt}"
EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:--1}"
DEVICE="${DEVICE:-0}"
PATIENCE="${PATIENCE:-30}"
WORKERS="${WORKERS:-0}"
CACHE="${CACHE:-False}"
VAL_FRACTION="${VAL_FRACTION:-0.20}"
SPLIT_SEED="${SPLIT_SEED:-42}"

for command_name in python3 cp find; do
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
            OUTPUT_ROOT="$EXPORT_DIR" "$EXPORT_SCRIPT"
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

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
dataset_dir="$TRAINING_ROOT/lego-$timestamp"
run_name="lego-$timestamp"

mkdir -p "$TRAINING_ROOT" "$RUNS_ROOT" "$PREDICTIONS_ROOT"
cp -aL "$SOURCE_DATASET" "$dataset_dir"

printf 'Creating a class-preserving train/validation split...\n'
python3 - "$dataset_dir" "$VAL_FRACTION" "$SPLIT_SEED" <<'PY'
import random
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
val_fraction = float(sys.argv[2])
seed = int(sys.argv[3])

if not 0 < val_fraction < 1:
    raise SystemExit("VAL_FRACTION must be between 0 and 1")

image_dir = root / "images" / "train"
label_dir = root / "labels" / "train"
images = sorted(
    path for path in image_dir.iterdir()
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if len(images) < 2:
    raise SystemExit("At least two images are required")

labels_by_image = {}
all_classes = set()
negative_images = 0
for image in images:
    label = label_dir / f"{image.stem}.txt"
    classes = set()
    if label.is_file():
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                classes.add(int(fields[0]))
    else:
        # CVAT/YOLO may omit the label file for a valid negative image that
        # contains no annotated objects. Ultralytics supports this directly.
        negative_images += 1

    labels_by_image[image] = classes
    all_classes.update(classes)

if not all_classes:
    raise SystemExit("No labeled objects were found")

val_count = max(1, min(len(images) - 1, round(len(images) * val_fraction)))
selected = None

for attempt in range(1000):
    rng = random.Random(seed + attempt)
    candidate = set(rng.sample(images, val_count))
    train_candidate = set(images) - candidate
    val_classes = set().union(*(labels_by_image[p] for p in candidate))
    train_classes = set().union(*(labels_by_image[p] for p in train_candidate))
    if val_classes == all_classes and train_classes == all_classes:
        selected = candidate
        used_seed = seed + attempt
        break

if selected is None:
    raise SystemExit(
        "Could not create a split containing every class in both subsets. "
        "Add more examples of rare classes or choose the split manually."
    )

val_image_dir = root / "images" / "val"
val_label_dir = root / "labels" / "val"
val_image_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)

for image in sorted(selected):
    label = label_dir / f"{image.stem}.txt"
    shutil.move(str(image), val_image_dir / image.name)
    if label.is_file():
        shutil.move(str(label), val_label_dir / label.name)

def image_paths(directory: Path):
    return sorted(
        path.relative_to(root).as_posix()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

train_paths = image_paths(image_dir)
val_paths = image_paths(val_image_dir)
(root / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
(root / "val.txt").write_text("\n".join(val_paths) + "\n", encoding="utf-8")

yaml_path = root / "data.yaml"
if not yaml_path.is_file():
    raise SystemExit(f"Missing dataset configuration: {yaml_path}")

original_lines = yaml_path.read_text(encoding="utf-8").splitlines()
updated_lines = []
seen = set()
for line in original_lines:
    key = line.split(":", 1)[0].strip() if ":" in line else ""
    if key == "path":
        updated_lines.append(f'path: "{root}"')
        seen.add("path")
    elif key == "train":
        updated_lines.append("train: train.txt")
        seen.add("train")
    elif key == "val":
        updated_lines.append("val: val.txt")
        seen.add("val")
    else:
        updated_lines.append(line)

for key, value in (
    ("path", f'"{root}"'),
    ("train", "train.txt"),
    ("val", "val.txt"),
):
    if key not in seen:
        updated_lines.append(f"{key}: {value}")

yaml_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

print(f"Split seed: {used_seed}")
print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Unlabelled negative images: {negative_images}")
print(f"Classes in both subsets: {sorted(all_classes)}")
PY

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
"$YOLO_VENV/bin/python" - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

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

