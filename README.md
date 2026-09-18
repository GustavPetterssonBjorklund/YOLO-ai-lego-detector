# LEGO YOLO detector

This repository exports labeled LEGO images from CVAT, fine-tunes an
Ultralytics YOLO model, and can serve live predictions from one camera or RTSP
source.

## Live annotated video

Install the dependencies and start the server with a trained weights file:

```bash
./install.sh
MODEL=artifacts/runs/lego-example/weights/best.pt \
VIDEO_SOURCE=0 \
./run-video-server.sh
```

Open `http://localhost:8000` in a browser. The displayed MJPEG stream contains
YOLO boxes and labels. **Capture raw frame** saves the corresponding unmodified
camera image beneath `artifacts/captures/<UTC-date>/`; captured images have no
generated labels and are intended for later import and annotation in CVAT.

`VIDEO_SOURCE` may be a webcam index or an RTSP URL. Optional settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONF` | `0.25` | YOLO confidence threshold from 0 to 1 |
| `DEVICE` | automatic | Ultralytics device such as `0` or `cpu` |
| `CAPTURE_ROOT` | `artifacts/captures` | Raw-frame output directory |
| `HOST` | `0.0.0.0` | HTTP bind address |
| `PORT` | `8000` | HTTP port |

The server deliberately has no authentication or TLS. Use it only on a trusted
LAN, or place it behind an authenticated HTTPS reverse proxy. It must run with
one Uvicorn worker because that process owns the camera and model.

The HTTP surface is `GET /`, `GET /stream.mjpg`, `POST /api/snapshot`, and
`GET /api/health`. If a camera or RTSP source disconnects, the service reports a
degraded health state and retries it every two seconds.

## Training pipeline

Run `./run-lego-yolo-pipeline.sh` to optionally export the configured CVAT
project, prepare a class-preserving train/validation split, train YOLO, and
write predictions below `artifacts/`. Configuration values at the top of the
script can be overridden with environment variables.

## Missed-detection comparisons

Run `./run-missed-detection-comparison.sh` to find the newest trained
`best.pt` and prepared dataset, evaluate every train and validation image, and
write side-by-side ground-truth/model images for examples containing missed
objects or model predictions without matching dataset labels. The output and a
`summary.csv` are written below
`artifacts/missed-comparisons/`.

A ground-truth object and prediction count as matched when they have the same
class and at least `MATCH_IOU=0.50` overlap. Unmatched boxes on either side are
shown in red, making both missed detections and potentially missing/incorrect
dataset annotations visible. `CONF`, `MATCH_IOU`, `DEVICE`, `MODEL`, `DATASET`,
and `OUTPUT_DIR` can be overridden through environment variables.
