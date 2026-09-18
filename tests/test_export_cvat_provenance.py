import unittest
from types import SimpleNamespace
from pathlib import Path

from scripts.export_cvat_provenance import FrameOrigin, frame_numbers, match_origin


class ExportCvatProvenanceTest(unittest.TestCase):
    def test_frame_numbers_honor_start_and_step(self) -> None:
        meta = SimpleNamespace(
            included_frames=None, start_frame=10, frame_filter="step=3"
        )
        self.assertEqual([10, 13, 16], frame_numbers(meta, 3))

    def test_exact_source_path_wins(self) -> None:
        origins = [
            FrameOrigin("camera-a/frame.jpg", 1, "A", (10,)),
            FrameOrigin("camera-b/frame.jpg", 2, "B", (20,)),
        ]
        result = match_origin(Path("camera-b/frame.jpg"), origins)
        self.assertIsNotNone(result)
        self.assertEqual(2, result.task_id)

    def test_duplicate_basename_is_not_guessed(self) -> None:
        origins = [
            FrameOrigin("camera-a/frame.jpg", 1, "A", (10,)),
            FrameOrigin("camera-b/frame.jpg", 2, "B", (20,)),
        ]
        self.assertIsNone(match_origin(Path("frame.jpg"), origins))


if __name__ == "__main__":
    unittest.main()
