import csv
import tempfile
import unittest
from pathlib import Path

from scripts.compare_missed_detections import Box, intersection_over_union
from scripts.compare_missed_detections import load_provenance, provenance_for_image
from scripts.compare_missed_detections import missed_ground_truth_indices
from scripts.compare_missed_detections import unmatched_indices


class CompareMissedDetectionsTest(unittest.TestCase):
    def test_iou(self) -> None:
        left = Box(0, (0, 0, 10, 10))
        right = Box(0, (5, 0, 15, 10))
        self.assertAlmostEqual(1 / 3, intersection_over_union(left, right))

    def test_matching_is_class_aware(self) -> None:
        label = Box(0, (0, 0, 10, 10))
        wrong_class = Box(1, (0, 0, 10, 10), 0.99)
        self.assertEqual({0}, missed_ground_truth_indices([label], [wrong_class], 0.5))

    def test_one_prediction_cannot_match_two_labels(self) -> None:
        labels = [Box(0, (0, 0, 10, 10)), Box(0, (0, 0, 10, 10))]
        prediction = Box(0, (0, 0, 10, 10), 0.99)
        self.assertEqual(
            1, len(missed_ground_truth_indices(labels, [prediction], 0.5))
        )

    def test_matching_accepts_overlap_at_threshold(self) -> None:
        label = Box(0, (0, 0, 10, 10))
        prediction = Box(0, (0, 0, 10, 10), 0.99)
        self.assertEqual(set(), missed_ground_truth_indices([label], [prediction], 1.0))

    def test_prediction_without_a_label_is_reported(self) -> None:
        prediction = Box(0, (0, 0, 10, 10), 0.99)
        missed, extra = unmatched_indices([], [prediction], 0.5)
        self.assertEqual(set(), missed)
        self.assertEqual({0}, extra)

    def test_wrong_class_is_unmatched_on_both_sides(self) -> None:
        label = Box(0, (0, 0, 10, 10))
        prediction = Box(1, (0, 0, 10, 10), 0.99)
        self.assertEqual(({0}, {0}), unmatched_indices([label], [prediction], 0.5))

    def test_provenance_survives_train_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            with (dataset / "provenance.csv").open(
                "w", encoding="utf-8", newline=""
            ) as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=["image", "task_id", "task_name", "job_ids"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image": "train/frame-1.jpg",
                        "task_id": "12",
                        "task_name": "LEGO batch",
                        "job_ids": "34",
                    }
                )

            provenance = load_provenance(dataset)
            result = provenance_for_image(
                provenance, "val", Path("frame-1.jpg")
            )
            self.assertIsNotNone(result)
            self.assertEqual("12", result.task_id)
            self.assertEqual("34", result.job_ids)


if __name__ == "__main__":
    unittest.main()
