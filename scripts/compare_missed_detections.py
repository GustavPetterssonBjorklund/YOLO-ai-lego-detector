#!/usr/bin/env python3
"""Render ground truth beside model output for images with missed objects."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device")
    return parser.parse_args()


def validate_fraction(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise SystemExit(f"{name} must be between 0 and 1")


def intersection_over_union(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = left.xyxy
    rx1, ry1, rx2, ry2 = right.xyxy
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def missed_ground_truth_indices(
    labels: list[Box], predictions: list[Box], minimum_iou: float
) -> set[int]:
    """Class-aware, one-to-one greedy matching, highest IoU first."""
    candidates = sorted(
        (
            (intersection_over_union(label, prediction), label_index, prediction_index)
            for label_index, label in enumerate(labels)
            for prediction_index, prediction in enumerate(predictions)
            if label.class_id == prediction.class_id
        ),
        reverse=True,
    )
    matched_labels: set[int] = set()
    matched_predictions: set[int] = set()
    for overlap, label_index, prediction_index in candidates:
        if overlap < minimum_iou:
            break
        if label_index in matched_labels or prediction_index in matched_predictions:
            continue
        matched_labels.add(label_index)
        matched_predictions.add(prediction_index)
    return set(range(len(labels))) - matched_labels


def load_class_names(data_yaml: Path) -> dict[int, str]:
    import yaml

    contents = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = contents.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise SystemExit(f"Invalid names entry in {data_yaml}")


def load_labels(label_path: Path, width: int, height: int) -> list[Box]:
    if not label_path.is_file():
        return []

    labels: list[Box] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise SystemExit(
                f"Expected YOLO detection labels with 5 fields in {label_path} "
                f"on line {line_number}, found {len(fields)}"
            )
        try:
            class_id = int(fields[0])
            center_x, center_y, box_width, box_height = map(float, fields[1:])
        except ValueError as error:
            raise SystemExit(
                f"Invalid label in {label_path} on line {line_number}: {line!r}"
            ) from error
        labels.append(
            Box(
                class_id,
                (
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                ),
            )
        )
    return labels


def predictions_from_result(result: Any) -> list[Box]:
    boxes = result.boxes
    if boxes is None:
        return []
    coordinates = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    return [
        Box(int(class_id), tuple(map(float, xyxy)), float(confidence))
        for xyxy, class_id, confidence in zip(coordinates, classes, confidences)
    ]


def class_name(class_id: int, names: dict[int, str]) -> str:
    return names.get(class_id, f"class {class_id}")


def draw_boxes(
    image: Any,
    boxes: list[Box],
    names: dict[int, str],
    *,
    missed: set[int] | None = None,
) -> Any:
    import cv2

    rendered = image.copy()
    missed = missed or set()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (round(value) for value in box.xyxy)
        color = (0, 0, 255) if index in missed else (0, 190, 0)
        caption = class_name(box.class_id, names)
        if box.confidence is not None:
            caption = f"{caption} {box.confidence:.2f}"
            color = (255, 140, 0)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_y = max(text_height + baseline + 2, y1)
        cv2.rectangle(
            rendered,
            (x1, text_y - text_height - baseline - 2),
            (x1 + text_width + 4, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            rendered,
            caption,
            (x1 + 2, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def add_header(image: Any, title: str) -> Any:
    import cv2
    import numpy as np

    header_height = 42
    header = np.full((header_height, image.shape[1], 3), 35, dtype=image.dtype)
    cv2.putText(
        header,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def find_images(dataset: Path) -> list[tuple[str, Path]]:
    images: list[tuple[str, Path]] = []
    for split in ("train", "val"):
        split_dir = dataset / "images" / split
        if not split_dir.is_dir():
            continue
        images.extend(
            (split, path)
            for path in sorted(split_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return images


def prediction_results(model: Any, images: Iterable[Path], args: argparse.Namespace):
    options: dict[str, Any] = {
        "source": [str(path) for path in images],
        "conf": args.conf,
        "iou": args.nms_iou,
        "imgsz": args.imgsz,
        "stream": True,
        "verbose": False,
    }
    if args.device:
        options["device"] = args.device
    return model.predict(**options)


def run(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    validate_fraction("--conf", args.conf)
    validate_fraction("--match-iou", args.match_iou)
    validate_fraction("--nms-iou", args.nms_iou)
    dataset = args.dataset.resolve()
    images = find_images(dataset)
    if not images:
        raise SystemExit(f"No train or validation images found under {dataset}")

    names = load_class_names(dataset / "data.yaml")
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.resolve()))
    results = prediction_results(model, (path for _, path in images), args)

    rows: list[dict[str, str | int]] = []
    total_labels = 0
    total_missed = 0
    affected_images = 0

    for (split, image_path), result in zip(images, results, strict=True):
        image = result.orig_img
        height, width = image.shape[:2]
        relative_path = image_path.relative_to(dataset / "images" / split)
        label_path = dataset / "labels" / split / relative_path.with_suffix(".txt")
        labels = load_labels(label_path, width, height)
        predictions = predictions_from_result(result)
        missed = missed_ground_truth_indices(labels, predictions, args.match_iou)
        total_labels += len(labels)
        total_missed += len(missed)
        if not missed:
            continue

        affected_images += 1
        ground_truth = add_header(
            draw_boxes(image, labels, names, missed=missed),
            f"DATASET LABELS - {len(missed)} MISSED IN RED",
        )
        model_output = add_header(
            draw_boxes(image, predictions, names),
            f"MODEL OUTPUT - conf >= {args.conf:g}",
        )
        comparison = np.hstack((ground_truth, model_output))
        destination = args.output / split / relative_path.with_suffix(".jpg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), comparison):
            raise SystemExit(f"Could not write comparison image: {destination}")

        missed_names = sorted(class_name(labels[index].class_id, names) for index in missed)
        rows.append(
            {
                "split": split,
                "image": str(relative_path),
                "label_count": len(labels),
                "prediction_count": len(predictions),
                "missed_count": len(missed),
                "missed_classes": "; ".join(missed_names),
                "comparison": str(destination.relative_to(args.output)),
            }
        )

    summary_path = args.output / "summary.csv"
    fieldnames = [
        "split",
        "image",
        "label_count",
        "prediction_count",
        "missed_count",
        "missed_classes",
        "comparison",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images checked: {len(images)}")
    print(f"Ground-truth objects: {total_labels}")
    print(f"Missed objects: {total_missed}")
    print(f"Images with misses: {affected_images}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
