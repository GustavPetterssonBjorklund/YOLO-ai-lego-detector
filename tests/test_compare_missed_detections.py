import unittest

from scripts.compare_missed_detections import Box, intersection_over_union
from scripts.compare_missed_detections import missed_ground_truth_indices


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


if __name__ == "__main__":
    unittest.main()
