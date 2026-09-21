"""Synthetic classification data must carry a learnable signal.

Both synthetic generators drew targets from torch.randint, independent of the
features stored beside them. y was independent of X, so the Bayes-optimal
classifier was the constant majority class and the ceiling was 1/num_classes.
The quickstart chapter points a new user at this data first, and
configs/dev/synthetic_label_skew.yaml selects a best checkpoint on it by
maximising val_accuracy_sample_weighted_avg -- selecting on noise.
"""

from __future__ import annotations

import unittest
from collections import Counter

import pytest
import torch

from fedbrew.data.generate import _generate_synthetic_tensors
from fedbrew.data.synthetic_classification import (
    SyntheticClassificationDataset,
    synthetic_labels,
    synthetic_teacher,
)

pytestmark = pytest.mark.fast


def _fit_probe(features: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Least-squares linear probe, returned as a scoring function."""

    design = torch.cat([features, torch.ones(len(features), 1)], dim=1)
    one_hot = torch.nn.functional.one_hot(targets, num_classes).float()
    weights = torch.linalg.lstsq(design, one_hot).solution

    def score(x: torch.Tensor, y: torch.Tensor) -> float:
        padded = torch.cat([x, torch.ones(len(x), 1)], dim=1)
        return float(((padded @ weights).argmax(dim=1) == y).float().mean())

    return score


def _class_separation(features: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Between-class mean spread in units of the within-class spread.

    Under exact independence this is sqrt(num_classes / n): the sampling noise
    of a mean over n / num_classes draws, and nothing else.
    """

    means = torch.stack([features[targets == c].mean(dim=0) for c in range(num_classes)])
    return float(means.std(dim=0).mean() / features.std(dim=0).mean())


class GeneratedTensorsCarrySignalTests(unittest.TestCase):
    """fedbrew/data/generate.py:_generate_synthetic_tensors."""

    NUM_SAMPLES = 20000
    INPUT_DIM = 8
    NUM_CLASSES = 4

    def setUp(self) -> None:
        self.features, self.targets = _generate_synthetic_tensors(
            num_samples=self.NUM_SAMPLES,
            input_dim=self.INPUT_DIM,
            num_classes=self.NUM_CLASSES,
            seed=0,
        )

    def test_class_conditional_means_are_separated(self) -> None:
        independence = (self.NUM_CLASSES / self.NUM_SAMPLES) ** 0.5
        separation = _class_separation(self.features, self.targets, self.NUM_CLASSES)
        # Measured 0.0135 before the fix against a 0.0141 independence baseline.
        self.assertGreater(separation, 10 * independence)

    def test_a_probe_generalises_to_held_out_examples(self) -> None:
        # Scored out of sample: fitting and scoring on the same rows cleared
        # chance by 0.9 points on the old data, which was the overfitting
        # floor rather than signal.
        half = self.NUM_SAMPLES // 2
        score = _fit_probe(self.features[:half], self.targets[:half], self.NUM_CLASSES)
        accuracy = score(self.features[half:], self.targets[half:])
        self.assertGreater(accuracy, 3 * (1.0 / self.NUM_CLASSES))

    def test_every_class_is_represented(self) -> None:
        counts = Counter(self.targets.tolist())
        self.assertEqual(set(counts), set(range(self.NUM_CLASSES)))
        # A teacher makes the classes unequal, but none may vanish.
        self.assertGreater(min(counts.values()), 0.05 * self.NUM_SAMPLES)

    def test_the_same_seed_gives_the_same_data(self) -> None:
        again = _generate_synthetic_tensors(num_samples=64, input_dim=4, num_classes=3, seed=11)
        once = _generate_synthetic_tensors(num_samples=64, input_dim=4, num_classes=3, seed=11)
        self.assertTrue(torch.equal(once[0], again[0]))
        self.assertTrue(torch.equal(once[1], again[1]))
        other = _generate_synthetic_tensors(num_samples=64, input_dim=4, num_classes=3, seed=12)
        self.assertFalse(torch.equal(once[1], other[1]))


class SyntheticDatasetCarriesSignalTests(unittest.TestCase):
    """fedbrew/data/synthetic_classification.py:SyntheticClassificationDataset."""

    def setUp(self) -> None:
        self.dataset = SyntheticClassificationDataset(
            num_clients=3, samples_per_client=2000, input_dim=5, num_classes=2
        )
        self.clients = self.dataset.list_clients()

    def _pooled_train(self) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [self.dataset.get_client_data(c)["train"] for c in self.clients]
        return (
            torch.cat([p["X"] for p in parts]),
            torch.cat([p["y"] for p in parts]),
        )

    def test_a_probe_fit_on_clients_generalises_to_the_global_test_set(self) -> None:
        # The clients and the global test set are drawn by two different
        # generators. If they did not share one teacher, a probe fit on the
        # clients would score at chance on the test set.
        features, targets = self._pooled_train()
        score = _fit_probe(features, targets, 2)
        global_data = self.dataset.get_global_data("test")
        self.assertGreater(score(global_data["X"], global_data["y"]), 0.75)

    def test_no_client_collapses_to_one_class(self) -> None:
        # The +index covariate shift must stay covariate shift. Applied through
        # a teacher that is not blind to it, clients 1 and 2 came out 19:1 on a
        # two-class problem and a majority-class predictor scored 95%.
        for client_id in self.clients:
            with self.subTest(client=client_id):
                targets = self.dataset.get_client_data(client_id)["train"]["y"]
                counts = Counter(targets.tolist())
                self.assertEqual(set(counts), {0, 1})
                majority = max(counts.values()) / len(targets)
                self.assertLess(majority, 0.65)

    def test_the_per_client_covariate_shift_is_still_there(self) -> None:
        means = [float(self.dataset.get_client_data(c)["train"]["X"].mean()) for c in self.clients]
        for index, mean in enumerate(means):
            self.assertAlmostEqual(mean, float(index), delta=0.1)


class TeacherTests(unittest.TestCase):
    def test_the_teacher_ignores_a_constant_added_to_every_dimension(self) -> None:
        teacher = synthetic_teacher(input_dim=6, num_classes=3, seed=5)
        features = torch.randn(500, 6, generator=torch.Generator().manual_seed(1))
        for shift in (1.0, 2.0, 7.5):
            with self.subTest(shift=shift):
                plain = synthetic_labels(features, teacher, torch.Generator().manual_seed(3))
                shifted = synthetic_labels(
                    features + shift, teacher, torch.Generator().manual_seed(3)
                )
                self.assertTrue(torch.equal(plain, shifted))

    def test_the_teacher_is_deterministic_in_its_seed(self) -> None:
        first = synthetic_teacher(4, 3, seed=9)
        self.assertTrue(torch.equal(first, synthetic_teacher(4, 3, seed=9)))
        self.assertFalse(torch.equal(first, synthetic_teacher(4, 3, seed=10)))

    def test_labels_depend_on_the_features(self) -> None:
        teacher = synthetic_teacher(input_dim=4, num_classes=3, seed=2)
        features = torch.randn(200, 4, generator=torch.Generator().manual_seed(0))
        one = synthetic_labels(features, teacher, torch.Generator().manual_seed(4))
        other = synthetic_labels(
            torch.randn(200, 4, generator=torch.Generator().manual_seed(1)),
            teacher,
            torch.Generator().manual_seed(4),
        )
        self.assertFalse(torch.equal(one, other))


if __name__ == "__main__":
    unittest.main()
