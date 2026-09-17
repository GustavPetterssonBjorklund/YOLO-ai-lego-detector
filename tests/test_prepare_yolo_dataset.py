import tempfile
import unittest
from pathlib import Path

from scripts.prepare_yolo_dataset import prepare_dataset


class PrepareDatasetTest(unittest.TestCase):
    def test_split_uses_directories_and_removes_stale_path_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            for index in range(4):
                (image_dir / f"image-{index}.jpg").touch()
            for index in range(3):
                (label_dir / f"image-{index}.txt").write_text(
                    "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
                )

            (root / "data.yaml").write_text(
                "train: train.txt\nval: val.txt\nnames:\n  0: brick\n",
                encoding="utf-8",
            )
            (root / "train.txt").write_text("images/train/missing.jpg\n")
            (root / "val.txt").touch()
            (root / "images" / "train.cache").touch()

            prepare_dataset(root, val_fraction=0.5, seed=42)

            data_yaml = (root / "data.yaml").read_text(encoding="utf-8")
            self.assertIn(f'path: "{root}"', data_yaml)
            self.assertIn("train: images/train", data_yaml)
            self.assertIn("val: images/val", data_yaml)
            self.assertFalse((root / "train.txt").exists())
            self.assertFalse((root / "val.txt").exists())
            self.assertFalse((root / "images" / "train.cache").exists())
            self.assertEqual(2, len(list((root / "images" / "train").glob("*.jpg"))))
            self.assertEqual(2, len(list((root / "images" / "val").glob("*.jpg"))))


if __name__ == "__main__":
    unittest.main()
