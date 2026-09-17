#!/usr/bin/env python3
"""Prepare a copied CVAT YOLO dataset for an Ultralytics training run."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_SPLIT_ATTEMPTS = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root of the copied YOLO dataset")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def classes_in_label(label: Path) -> set[int]:
    classes: set[int] = set()
    for line_number, line in enumerate(
        label.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        try:
            classes.add(int(fields[0]))
        except ValueError as error:
            raise SystemExit(
                f"Invalid class ID in {label} on line {line_number}: {fields[0]!r}"
            ) from error
    return classes


def read_labels(
    images: list[Path], label_dir: Path
) -> tuple[dict[Path, set[int]], set[int], int]:
    labels_by_image: dict[Path, set[int]] = {}
    all_classes: set[int] = set()
    negative_images = 0

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if label.is_file():
            classes = classes_in_label(label)
        else:
            # A missing label is a valid negative image in Ultralytics YOLO.
            classes = set()
            negative_images += 1

        labels_by_image[image] = classes
        all_classes.update(classes)

    return labels_by_image, all_classes, negative_images


def choose_validation_images(
    images: list[Path],
    labels_by_image: dict[Path, set[int]],
    all_classes: set[int],
    val_fraction: float,
    seed: int,
) -> tuple[set[Path], int]:
    validation_size = max(1, min(len(images) - 1, round(len(images) * val_fraction)))

    for attempt in range(MAX_SPLIT_ATTEMPTS):
        used_seed = seed + attempt
        candidate = set(random.Random(used_seed).sample(images, validation_size))
        training = set(images) - candidate
        validation_classes = set().union(*(labels_by_image[path] for path in candidate))
        training_classes = set().union(*(labels_by_image[path] for path in training))
        if validation_classes == all_classes and training_classes == all_classes:
            return candidate, used_seed

    raise SystemExit(
        "Could not create a split containing every class in both subsets. "
        "Add more examples of rare classes or choose the split manually."
    )


def move_to_validation(
    selected: set[Path],
    label_dir: Path,
    validation_image_dir: Path,
    validation_label_dir: Path,
) -> None:
    validation_image_dir.mkdir(parents=True, exist_ok=True)
    validation_label_dir.mkdir(parents=True, exist_ok=True)

    for image in sorted(selected):
        label = label_dir / f"{image.stem}.txt"
        shutil.move(image, validation_image_dir / image.name)
        if label.is_file():
            shutil.move(label, validation_label_dir / label.name)


def remove_stale_path_files(root: Path) -> None:
    # Manifests and caches can contain paths from a different working directory
    # or machine. Directory-based data.yaml entries make them unnecessary.
    for manifest in (root / "train.txt", root / "val.txt"):
        manifest.unlink(missing_ok=True)
    for cache in root.rglob("*.cache"):
        cache.unlink()


def update_data_yaml(root: Path) -> None:
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise SystemExit(f"Missing dataset configuration: {yaml_path}")

    values = {
        "path": f'"{root}"',
        "train": "images/train",
        "val": "images/val",
    }
    seen: set[str] = set()
    updated_lines: list[str] = []

    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        key = line.split(":", 1)[0].strip() if ":" in line else ""
        if key in values:
            updated_lines.append(f"{key}: {values[key]}")
            seen.add(key)
        else:
            updated_lines.append(line)

    for key, value in values.items():
        if key not in seen:
            updated_lines.append(f"{key}: {value}")

    yaml_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def prepare_dataset(root: Path, val_fraction: float, seed: int) -> None:
    root = root.resolve()
    if not 0 < val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1")

    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"Missing images/train or labels/train under {root}")
    if not (root / "data.yaml").is_file():
        raise SystemExit(f"Missing dataset configuration: {root / 'data.yaml'}")

    images = find_images(image_dir)
    if len(images) < 2:
        raise SystemExit("At least two images are required")

    labels_by_image, all_classes, negative_images = read_labels(images, label_dir)
    if not all_classes:
        raise SystemExit("No labeled objects were found")

    selected, used_seed = choose_validation_images(
        images, labels_by_image, all_classes, val_fraction, seed
    )
    validation_image_dir = root / "images" / "val"
    validation_label_dir = root / "labels" / "val"
    move_to_validation(selected, label_dir, validation_image_dir, validation_label_dir)
    remove_stale_path_files(root)
    update_data_yaml(root)

    print(f"Split seed: {used_seed}")
    print(f"Training images: {len(find_images(image_dir))}")
    print(f"Validation images: {len(find_images(validation_image_dir))}")
    print(f"Unlabelled negative images: {negative_images}")
    print(f"Classes in both subsets: {sorted(all_classes)}")


def main() -> None:
    args = parse_args()
    prepare_dataset(args.dataset, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
