#!/usr/bin/env python3
"""Serve one YOLO-annotated video source as MJPEG and capture raw frames."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJPEG_MEDIA_TYPE = "multipart/x-mixed-replace; boundary=frame"


def parse_video_source(value: str) -> int | str:
    """Turn a non-negative integer into a camera index; leave URLs/paths alone."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("VIDEO_SOURCE must not be empty")
    if stripped.isdecimal():
        return int(stripped)
    return stripped


@dataclass(frozen=True)
class Settings:
    model_path: Path
    video_source: int | str = 0
    confidence: float = 0.25
    device: str | None = None
    capture_root: Path = PROJECT_ROOT / "artifacts" / "captures"
    jpeg_quality: int = 85
    reconnect_delay: float = 2.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if environ is None else environ
        model_value = values.get("MODEL", "").strip()
        if not model_value:
            raise RuntimeError("MODEL must point to a trained .pt weights file")

        model_path = Path(model_value).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise RuntimeError(f"MODEL does not exist or is not a file: {model_path}")
        if model_path.suffix.lower() != ".pt":
            raise RuntimeError(f"MODEL must be a .pt weights file: {model_path}")

        try:
            confidence = float(values.get("CONF", "0.25"))
        except ValueError as error:
            raise RuntimeError("CONF must be a number between 0 and 1") from error
        if not 0 <= confidence <= 1:
            raise RuntimeError("CONF must be between 0 and 1")

        capture_root = Path(
            values.get("CAPTURE_ROOT", str(PROJECT_ROOT / "artifacts" / "captures"))
        ).expanduser()
        if not capture_root.is_absolute():
            capture_root = PROJECT_ROOT / capture_root

        device = values.get("DEVICE", "").strip() or None
        return cls(
            model_path=model_path,
            video_source=parse_video_source(values.get("VIDEO_SOURCE", "0")),
            confidence=confidence,
            device=device,
            capture_root=capture_root.resolve(),
        )


def load_yolo_model(model_path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(model_path))


def open_video_source(source: int | str) -> Any:
    import cv2

    return cv2.VideoCapture(source)


def encode_jpeg(frame: Any, quality: int) -> bytes:
    import cv2

    success, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        raise RuntimeError("OpenCV could not encode the annotated frame as JPEG")
    return encoded.tobytes()


def write_image(path: Path, frame: Any) -> bool:
    import cv2

    return bool(cv2.imwrite(str(path), frame))


class VideoInferenceService:
    """Own the single video reader/inference loop shared by every HTTP client."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_loader: Callable[[Path], Any] = load_yolo_model,
        capture_factory: Callable[[int | str], Any] = open_video_source,
        jpeg_encoder: Callable[[Any, int], bytes] = encode_jpeg,
        image_writer: Callable[[Path, Any], bool] = write_image,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self._model_loader = model_loader
        self._capture_factory = capture_factory
        self._jpeg_encoder = jpeg_encoder
        self._image_writer = image_writer
        self._now = now or (lambda: datetime.now(timezone.utc))

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._model: Any | None = None
        self._raw_frame: Any | None = None
        self._annotated_jpeg: bytes | None = None
        self._frame_sequence = 0
        self._capture_sequence = 0
        self._model_ready = False
        self._source_connected = False
        self._latest_frame_at: str | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._model = self._model_loader(self.settings.model_path)
        with self._condition:
            self._model_ready = True
            self._last_error = None
        self._thread = threading.Thread(
            target=self._run, name="yolo-video-inference", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            capture = self._capture
            self._condition.notify_all()
        if capture is not None:
            capture.release()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._condition:
            self._source_connected = False
            self._capture = None

    def _set_connection(self, connected: bool) -> None:
        with self._condition:
            self._source_connected = connected

    def _set_error(self, error: str) -> None:
        with self._condition:
            self._last_error = error

    def _publish_frame(self, raw_frame: Any, annotated_jpeg: bytes) -> None:
        with self._condition:
            self._raw_frame = raw_frame.copy()
            self._annotated_jpeg = annotated_jpeg
            self._frame_sequence += 1
            self._latest_frame_at = self._now().isoformat()
            self._last_error = None
            self._condition.notify_all()

    def _annotate(self, frame: Any) -> Any:
        predict_args: dict[str, Any] = {
            "source": frame,
            "conf": self.settings.confidence,
            "verbose": False,
        }
        if self.settings.device is not None:
            predict_args["device"] = self.settings.device
        results = self._model.predict(**predict_args)
        if not results:
            raise RuntimeError("YOLO returned no result for a video frame")
        return results[0].plot()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            capture = None
            try:
                capture = self._capture_factory(self.settings.video_source)
                with self._condition:
                    self._capture = capture
                if not capture.isOpened():
                    raise RuntimeError("Could not open VIDEO_SOURCE")
                self._set_connection(True)

                while not self._stop_event.is_set():
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError("Video source stopped returning frames")
                    try:
                        annotated = self._annotate(frame)
                        jpeg = self._jpeg_encoder(
                            annotated, self.settings.jpeg_quality
                        )
                        self._publish_frame(frame, jpeg)
                    except Exception as error:  # Keep a transient inference error alive.
                        self._set_error(f"Inference error: {error}")
                        if self._stop_event.wait(0.1):
                            break
            except Exception as error:
                if not self._stop_event.is_set():
                    self._set_error(str(error))
            finally:
                self._set_connection(False)
                if capture is not None:
                    capture.release()
                with self._condition:
                    if self._capture is capture:
                        self._capture = None

            if not self._stop_event.is_set():
                self._stop_event.wait(self.settings.reconnect_delay)

    def iter_mjpeg(self) -> Iterator[bytes]:
        seen_sequence = -1
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frame_sequence != seen_sequence
                    or self._stop_event.is_set(),
                    timeout=10,
                )
                if self._stop_event.is_set():
                    return
                if self._annotated_jpeg is None:
                    continue
                seen_sequence = self._frame_sequence
                jpeg = self._annotated_jpeg
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                + jpeg
                + b"\r\n"
            )

    def snapshot(self) -> str | None:
        with self._condition:
            if self._raw_frame is None:
                return None
            frame = self._raw_frame.copy()
            self._capture_sequence += 1
            capture_sequence = self._capture_sequence

        captured_at = self._now()
        day_directory = self.settings.capture_root / captured_at.strftime("%Y-%m-%d")
        day_directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f"frame-{captured_at.strftime('%Y%m%dT%H%M%S.%fZ')}"
            f"-{capture_sequence:06d}.jpg"
        )
        destination = day_directory / filename
        temporary = day_directory / f".{filename}.tmp.jpg"
        try:
            if not self._image_writer(temporary, frame):
                raise RuntimeError(f"Could not write captured frame to {temporary}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination.relative_to(self.settings.capture_root).as_posix()

    def health(self) -> dict[str, Any]:
        with self._condition:
            if self._source_connected and self._frame_sequence > 0:
                status = "ok"
            elif self._last_error:
                status = "degraded"
            else:
                status = "starting"
            return {
                "status": status,
                "model_ready": self._model_ready,
                "source_connected": self._source_connected,
                "frame_sequence": self._frame_sequence,
                "latest_frame_at": self._latest_frame_at,
                "last_error": self._last_error,
            }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LEGO YOLO camera</title>
  <style>
    body { max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #111;
      color: #eee; font: 16px system-ui, sans-serif; }
    img { display: block; width: 100%; background: #222; border-radius: .5rem; }
    footer { display: flex; gap: 1rem; align-items: center; margin-top: 1rem; }
    button { padding: .65rem 1rem; font: inherit; cursor: pointer; }
    #status { color: #bbb; }
  </style>
</head>
<body>
  <h1>LEGO YOLO camera</h1>
  <img src="/stream.mjpg" alt="Live annotated camera stream">
  <footer><button id="capture">Capture raw frame</button><span id="status">Connecting…</span></footer>
  <script>
    const status = document.querySelector('#status');
    document.querySelector('#capture').addEventListener('click', async () => {
      status.textContent = 'Capturing…';
      try {
        const response = await fetch('/api/snapshot', {method: 'POST'});
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || 'Capture failed');
        status.textContent = `Saved ${body.path}`;
      } catch (error) { status.textContent = error.message; }
    });
    async function updateHealth() {
      try {
        const health = await (await fetch('/api/health')).json();
        if (!status.textContent.startsWith('Saved')) {
          status.textContent = health.status === 'ok' ? 'Live' : (health.last_error || health.status);
        }
      } catch (_) { status.textContent = 'Server unavailable'; }
    }
    setInterval(updateHealth, 2000); updateHealth();
  </script>
</body>
</html>
"""


def create_app(
    settings: Settings | None = None,
    service: VideoInferenceService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_service = service or VideoInferenceService(settings or Settings.from_env())
        app.state.video_service = active_service
        active_service.start()
        try:
            yield
        finally:
            active_service.stop()

    application = FastAPI(title="LEGO YOLO stream", lifespan=lifespan)

    def active_service() -> VideoInferenceService:
        return application.state.video_service

    @application.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @application.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        return StreamingResponse(
            active_service().iter_mjpeg(), media_type=MJPEG_MEDIA_TYPE
        )

    @application.post("/api/snapshot")
    def snapshot() -> dict[str, str]:
        try:
            path = active_service().snapshot()
        except OSError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        if path is None:
            raise HTTPException(
                status_code=503, detail="No video frame is available yet"
            )
        return {"path": path}

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return active_service().health()

    return application


app = create_app()
