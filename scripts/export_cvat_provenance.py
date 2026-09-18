#!/usr/bin/env python3
"""Write CVAT task and job provenance for images in an exported dataset."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class FrameOrigin:
    source_name: str
    task_id: int
    task_name: str
    job_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_name(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().lstrip("./")


def frame_numbers(meta: Any, count: int) -> list[int]:
    included = getattr(meta, "included_frames", None)
    if included is not None and len(included) == count:
        return [int(number) for number in included]

    start = int(getattr(meta, "start_frame", None) or 0)
    frame_filter = str(getattr(meta, "frame_filter", None) or "")
    step = 1
    if frame_filter.startswith("step="):
        try:
            step = int(frame_filter.removeprefix("step="))
        except ValueError:
            step = 1
    return [start + index * step for index in range(count)]


def origin_for_frame(task: Any, jobs: list[Any], frame: int, source_name: str) -> FrameOrigin:
    matching_jobs = [
        job
        for job in jobs
        if getattr(job, "start_frame", None) is not None
        and getattr(job, "stop_frame", None) is not None
        and int(job.start_frame) <= frame <= int(job.stop_frame)
    ]
    annotation_jobs = [
        job for job in matching_jobs if str(getattr(job, "type", "annotation")) == "annotation"
    ]
    if annotation_jobs:
        matching_jobs = annotation_jobs
    return FrameOrigin(
        source_name=normalized_name(source_name),
        task_id=int(task.id),
        task_name=str(task.name),
        job_ids=tuple(sorted(int(job.id) for job in matching_jobs)),
    )


def collect_origins(client: Any, project_id: int) -> list[FrameOrigin]:
    project = client.projects.retrieve(project_id)
    origins: list[FrameOrigin] = []
    for task in project.get_tasks():
        meta = task.get_meta()
        frames = list(meta.frames or [])
        jobs = task.get_jobs()
        for frame, frame_number in zip(frames, frame_numbers(meta, len(frames)), strict=True):
            origins.append(
                origin_for_frame(task, jobs, frame_number, str(frame.name))
            )
    return origins


def match_origin(
    relative_image: Path, origins: list[FrameOrigin]
) -> FrameOrigin | None:
    relative_name = normalized_name(relative_image.as_posix())
    exact = [origin for origin in origins if origin.source_name == relative_name]
    if len(exact) == 1:
        return exact[0]

    suffix = [
        origin
        for origin in origins
        if relative_name.endswith(f"/{origin.source_name}")
        or origin.source_name.endswith(f"/{relative_name}")
    ]
    if len(suffix) == 1:
        return suffix[0]

    basename = relative_image.name
    by_basename = [
        origin for origin in origins if PurePosixPath(origin.source_name).name == basename
    ]
    return by_basename[0] if len(by_basename) == 1 else None


def dataset_images(dataset: Path) -> list[Path]:
    image_root = dataset / "images"
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_manifest(dataset: Path, output: Path, origins: list[FrameOrigin]) -> None:
    images = dataset_images(dataset)
    matched = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["image", "source_name", "task_id", "task_name", "job_ids"],
        )
        writer.writeheader()
        for image in images:
            relative = image.relative_to(dataset / "images")
            origin = match_origin(relative, origins)
            if origin is None:
                writer.writerow({"image": relative.as_posix()})
                continue
            matched += 1
            writer.writerow(
                {
                    "image": relative.as_posix(),
                    "source_name": origin.source_name,
                    "task_id": origin.task_id,
                    "task_name": origin.task_name,
                    "job_ids": ";".join(map(str, origin.job_ids)),
                }
            )
    print(f"CVAT provenance matched: {matched}/{len(images)} images")


def main() -> None:
    from cvat_sdk import make_client

    args = parse_args()
    token = os.environ.get("CVAT_ACCESS_TOKEN", "").strip()
    password = os.environ.get("PASS", "")
    client_options: dict[str, Any]
    if token:
        client_options = {"access_token": token}
    elif password:
        client_options = {"credentials": (args.username, password)}
    else:
        raise SystemExit("CVAT_ACCESS_TOKEN or PASS is required for provenance export")

    with make_client(args.host, **client_options) as client:
        client.organization_slug = args.organization
        origins = collect_origins(client, args.project_id)
    write_manifest(args.dataset.resolve(), args.output.resolve(), origins)


if __name__ == "__main__":
    main()
