import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


try:
    from scripts.stream_server import Settings, VideoInferenceService, parse_video_source
except ModuleNotFoundError as error:
    if error.name == "fastapi":
        Settings = VideoInferenceService = parse_video_source = None
    else:
        raise


@unittest.skipIf(Settings is None, "stream server dependencies are not installed")
class StreamServerTest(unittest.TestCase):
    def make_settings(self, root: Path, **overrides):
        values = {
            "model_path": root / "model.pt",
            "capture_root": root / "captures",
            "reconnect_delay": 0.01,
        }
        values.update(overrides)
        return Settings(**values)

    def test_parse_video_source_supports_camera_indices_and_urls(self) -> None:
        self.assertEqual(0, parse_video_source(" 0 "))
        self.assertEqual(12, parse_video_source("12"))
        self.assertEqual(
            "rtsp://camera.example/stream",
            parse_video_source("rtsp://camera.example/stream"),
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            parse_video_source("  ")

    def test_settings_require_existing_pt_model_and_validate_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "lego.pt"
            model.touch()
            settings = Settings.from_env(
                {
                    "MODEL": str(model),
                    "VIDEO_SOURCE": "2",
                    "CONF": "0.4",
                    "CAPTURE_ROOT": str(root / "frames"),
                }
            )
            self.assertEqual(model, settings.model_path)
            self.assertEqual(2, settings.video_source)
            self.assertEqual(0.4, settings.confidence)

            with self.assertRaisesRegex(RuntimeError, "between 0 and 1"):
                Settings.from_env({"MODEL": str(model), "CONF": "2"})
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                Settings.from_env({"MODEL": str(root / "missing.pt")})

    def test_snapshot_saves_raw_frame_and_mjpeg_uses_annotated_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            written = {}
            instant = datetime(2026, 9, 17, 12, 30, tzinfo=timezone.utc)

            def writer(path, frame):
                written["path"] = path
                written["frame"] = frame
                path.write_bytes(b"raw-jpeg")
                return True

            service = VideoInferenceService(
                self.make_settings(root), image_writer=writer, now=lambda: instant
            )
            service._publish_frame(["raw"], b"annotated-jpeg")

            relative_path = service.snapshot()
            self.assertEqual(
                "2026-09-17/frame-20260917T123000.000000Z-000001.jpg",
                relative_path,
            )
            self.assertEqual(["raw"], written["frame"])
            self.assertEqual(
                b"raw-jpeg", (root / "captures" / relative_path).read_bytes()
            )

            part = next(service.iter_mjpeg())
            self.assertIn(b"Content-Type: image/jpeg", part)
            self.assertIn(b"annotated-jpeg", part)
            self.assertNotIn(b"raw-jpeg", part)
            service.stop()

    def test_snapshot_is_unavailable_before_first_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = VideoInferenceService(
                self.make_settings(Path(temporary_directory))
            )
            self.assertIsNone(service.snapshot())
            self.assertEqual("starting", service.health()["status"])

    def test_reader_reconnects_and_recovers_after_failed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            captures = []
            created = 0

            class Capture:
                def __init__(self, frames, opened=True):
                    self.frames = iter(frames)
                    self.opened = opened
                    self.released = False

                def isOpened(self):
                    return self.opened

                def read(self):
                    return next(self.frames, (False, None))

                def release(self):
                    self.released = True

            class Result:
                def plot(self):
                    return ["annotated"]

            class Model:
                def predict(self, **_kwargs):
                    return [Result()]

            def capture_factory(_source):
                nonlocal created
                created += 1
                capture = (
                    Capture([], opened=False)
                    if created == 1
                    else Capture([(True, ["raw"])])
                )
                captures.append(capture)
                return capture

            service = VideoInferenceService(
                self.make_settings(root),
                model_loader=lambda _path: Model(),
                capture_factory=capture_factory,
                jpeg_encoder=lambda _frame, _quality: b"jpeg",
            )
            service.start()
            deadline = time.monotonic() + 2
            while service.health()["frame_sequence"] == 0:
                if time.monotonic() >= deadline:
                    self.fail("service did not recover after reconnecting")
                time.sleep(0.01)
            service.stop()

            self.assertGreaterEqual(created, 2)
            self.assertTrue(all(capture.released for capture in captures))
            self.assertEqual(1, service.health()["frame_sequence"])
            self.assertFalse(service.health()["source_connected"])


if __name__ == "__main__":
    unittest.main()
