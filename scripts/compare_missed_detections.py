#!/usr/bin/env python3
"""Render ground truth beside model output for images with missed objects."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None


@dataclass(frozen=True)
class Provenance:
    task_id: str = ""
    task_name: str = ""
    job_ids: str = ""


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


def unmatched_indices(
    labels: list[Box], predictions: list[Box], minimum_iou: float
) -> tuple[set[int], set[int]]:
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
    return (
        set(range(len(labels))) - matched_labels,
        set(range(len(predictions))) - matched_predictions,
    )


def missed_ground_truth_indices(
    labels: list[Box], predictions: list[Box], minimum_iou: float
) -> set[int]:
    """Return labels without a matching prediction."""
    return unmatched_indices(labels, predictions, minimum_iou)[0]


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
    unmatched: set[int] | None = None,
) -> Any:
    import cv2

    rendered = image.copy()
    unmatched = unmatched or set()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (round(value) for value in box.xyxy)
        color = (0, 0, 255) if index in unmatched else (0, 190, 0)
        caption = class_name(box.class_id, names)
        if box.confidence is not None:
            caption = f"{caption} {box.confidence:.2f}"
            if index not in unmatched:
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


def add_filename_header(image: Any, filename: str, provenance: Provenance | None) -> Any:
    import cv2
    import numpy as np

    header_height = 76
    header = np.full((header_height, image.shape[1], 3), 20, dtype=image.dtype)
    cv2.putText(
        header,
        f"IMAGE: {filename}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if provenance is None:
        provenance_text = "CVAT TASK / JOB: unavailable (re-export dataset for provenance)"
    else:
        task = f"{provenance.task_id} ({provenance.task_name})".strip()
        provenance_text = f"CVAT TASK: {task}    JOB: {provenance.job_ids or 'unknown'}"
    cv2.putText(
        header,
        provenance_text,
        (12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def load_provenance(dataset: Path) -> dict[str, Provenance]:
    manifest = dataset / "provenance.csv"
    if not manifest.is_file():
        return {}

    provenance: dict[str, Provenance] = {}
    basenames: dict[str, list[Provenance]] = {}
    with manifest.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            image = row.get("image", "").replace("\\", "/").lstrip("./")
            if not image or not row.get("task_id"):
                continue
            value = Provenance(
                task_id=row.get("task_id", ""),
                task_name=row.get("task_name", ""),
                job_ids=row.get("job_ids", ""),
            )
            provenance[image] = value
            basenames.setdefault(PurePath(image).name, []).append(value)

    for basename, values in basenames.items():
        if len(values) == 1:
            provenance.setdefault(basename, values[0])
    return provenance


def provenance_for_image(
    provenance: dict[str, Provenance], split: str, relative_path: Path
) -> Provenance | None:
    candidates = (
        f"{split}/{relative_path.as_posix()}",
        relative_path.as_posix(),
        relative_path.name,
    )
    return next((provenance[key] for key in candidates if key in provenance), None)


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
    provenance = load_provenance(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.resolve()))
    results = prediction_results(model, (path for _, path in images), args)

    rows: list[dict[str, str | int]] = []
    total_labels = 0
    total_missed = 0
    total_extra = 0
    affected_images = 0
    provenance_matches = 0

    for (split, image_path), result in zip(images, results, strict=True):
        image = result.orig_img
        height, width = image.shape[:2]
        relative_path = image_path.relative_to(dataset / "images" / split)
        label_path = dataset / "labels" / split / relative_path.with_suffix(".txt")
        labels = load_labels(label_path, width, height)
        predictions = predictions_from_result(result)
        missed, extra = unmatched_indices(labels, predictions, args.match_iou)
        image_provenance = provenance_for_image(provenance, split, relative_path)
        if image_provenance is not None:
            provenance_matches += 1
        total_labels += len(labels)
        total_missed += len(missed)
        total_extra += len(extra)
        if not missed and not extra:
            continue

        affected_images += 1
        ground_truth = add_header(
            draw_boxes(image, labels, names, unmatched=missed),
            f"DATASET LABELS - {len(missed)} UNDETECTED IN RED",
        )
        model_output = add_header(
            draw_boxes(image, predictions, names, unmatched=extra),
            f"MODEL OUTPUT - {len(extra)} UNLABELLED IN RED",
        )
        comparison = np.hstack((ground_truth, model_output))
        comparison = add_filename_header(
            comparison, f"{split}/{relative_path}", image_provenance
        )
        destination = args.output / split / relative_path.with_suffix(".jpg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), comparison):
            raise SystemExit(f"Could not write comparison image: {destination}")

        missed_names = sorted(class_name(labels[index].class_id, names) for index in missed)
        extra_names = sorted(
            class_name(predictions[index].class_id, names) for index in extra
        )
        rows.append(
            {
                "split": split,
                "image": str(relative_path),
                "label_count": len(labels),
                "prediction_count": len(predictions),
                "missed_count": len(missed),
                "missed_classes": "; ".join(missed_names),
                "unlabelled_prediction_count": len(extra),
                "unlabelled_prediction_classes": "; ".join(extra_names),
                "cvat_task_id": image_provenance.task_id if image_provenance else "",
                "cvat_task_name": image_provenance.task_name if image_provenance else "",
                "cvat_job_ids": image_provenance.job_ids if image_provenance else "",
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
        "unlabelled_prediction_count",
        "unlabelled_prediction_classes",
        "cvat_task_id",
        "cvat_task_name",
        "cvat_job_ids",
        "comparison",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images checked: {len(images)}")
    print(f"Ground-truth objects: {total_labels}")
    print(f"Missed objects: {total_missed}")
    print(f"Predictions without matching labels: {total_extra}")
    print(f"Images requiring review: {affected_images}")
    if (dataset / "provenance.csv").is_file():
        print(f"CVAT provenance matched: {provenance_matches}/{len(images)} images")
    else:
        print("CVAT provenance: unavailable (provenance.csv not found)")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
