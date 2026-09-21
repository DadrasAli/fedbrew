"""Deterministic synthetic federated classification data."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from fedbrew.data.dataset import FederatedDataset

#: Standard deviation of the noise added to the teacher's logits, in the same
#: units as the logits themselves. Non-zero so the task is not perfectly
#: separable and a model can be seen to improve rather than solve it in one
#: step; small enough that the Bayes accuracy is far above chance.
TEACHER_LOGIT_NOISE = 0.5

#: Offset that separates the teacher's draw from the feature draws. The client
#: features and the global test features come from two different generators
#: (seed and seed + 1000), so the teacher cannot be drawn from either without
#: the two halves of the dataset disagreeing about what the labels mean.
TEACHER_SEED_OFFSET = 7919


def synthetic_teacher(input_dim: int, num_classes: int, seed: int) -> Tensor:
    """The fixed linear rule that turns features into labels.

    One teacher for every client and for the global test set: a federated
    problem needs a single solution for the clients to be converging on.

    Each column sums to zero, which makes the teacher blind to a constant added
    to every input dimension. SyntheticClassificationDataset shifts client i's
    features by +i, and a shift c contributes c * sum(W[:, k]) to class k's
    logit -- a per-class constant that grows with the client index and swamps
    the per-example term. With a raw teacher, clients 1 and 2 of the shipped
    3-client fixture came out 19:1 and 19:1 on a two-class problem, so a
    majority-class predictor scored 95% and the accuracy metric stopped meaning
    anything. Zero-sum columns leave the shift as pure covariate heterogeneity,
    which is what it is documented to be.
    """

    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(input_dim, num_classes, generator=generator)
    return teacher - teacher.mean(dim=0, keepdim=True)


def synthetic_labels(
    features: Tensor,
    teacher: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Label features with the teacher, so y depends on X.

    Both synthetic generators used to draw targets from torch.randint,
    independent of the features they were stored beside. The Bayes-optimal
    classifier for that data is the constant majority class and the ceiling is
    1/num_classes, so every run on it sat at chance -- including the
    "guaranteed local synthetic experiment" in docs/03-quickstart.md
    and a 20-round config selecting a best checkpoint by validation accuracy.
    """

    logits = features @ teacher
    noise = torch.randn(logits.shape, generator=generator)
    return (logits + TEACHER_LOGIT_NOISE * noise).argmax(dim=1)


class SyntheticClassificationDataset(FederatedDataset):
    """Small deterministic tensor dataset split across fake clients."""

    def __init__(
        self,
        num_clients: int = 3,
        samples_per_client: int = 20,
        input_dim: int = 5,
        num_classes: int = 2,
        seed: int = 42,
    ) -> None:
        """Generate the whole dataset in memory from a seeded linear teacher.

        Args:
            num_clients: Number of synthetic clients, named ``client_0`` up.
            samples_per_client: Training examples per client.
            input_dim: Features per example; every client's ``X`` is
                (samples_per_client, input_dim) float.
            num_classes: Label count; ``y`` holds int64 indices in
                [0, num_classes).
            seed: Base seed. Fixes both the teacher's weights (offset by
                TEACHER_SEED_OFFSET so the teacher and the samples do not share
                a stream) and every drawn example, so the dataset is a pure
                function of these five arguments.

        Nothing is written to disk and nothing is downloaded, which is what
        makes this the fallback when the environment cannot reach the network.
        Labels come from a linear teacher, so the task is learnable by design
        and a run that fails to learn it points at the setup, not the data.
        """

        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seed = seed
        self._teacher = synthetic_teacher(input_dim, num_classes, seed + TEACHER_SEED_OFFSET)
        self._clients = [f"client_{index}" for index in range(num_clients)]
        self._client_data = self._build_client_data()
        self._global_test_data = self._build_global_test_data()
        self._client_test_data = self._split_global_test_data()

    def list_clients(self) -> list[str]:
        """Return synthetic client identifiers."""

        return list(self._clients)

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        """Return cloned train, eval, and explicit test tensors for one client."""

        if client_id not in self._client_data:
            raise KeyError(f"Unknown client_id: {client_id}")
        train_data = self._client_data[client_id]
        test_data = self._client_test_data[client_id]
        return {
            "train": {
                "X": train_data["X"].clone(),
                "y": train_data["y"].clone(),
            },
            "eval": {
                "X": test_data["X"].clone(),
                "y": test_data["y"].clone(),
            },
            "test": {
                "X": test_data["X"].clone(),
                "y": test_data["y"].clone(),
            },
            # Every split: train, eval and test are one samples_per_client
            # block each, and num_examples is their sum everywhere else.
            "num_examples": self.samples_per_client * 3,
            "num_train_examples": self.samples_per_client,
            "num_eval_examples": self.samples_per_client,
            "num_test_examples": self.samples_per_client,
        }

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        """Return split counts without materializing client tensors."""

        if client_id not in self._client_data:
            raise KeyError(f"Unknown client_id: {client_id}")
        return {
            "client_id": client_id,
            "num_examples": self.samples_per_client * 3,
            "num_train_examples": self.samples_per_client,
            "num_eval_examples": self.samples_per_client,
            "num_test_examples": self.samples_per_client,
        }

    def get_global_data(self, split: str | None = "test") -> dict[str, Any]:
        """Return deterministic global evaluation tensors."""

        if split == "train":
            features = [self._client_data[client_id]["X"] for client_id in self._clients]
            targets = [self._client_data[client_id]["y"] for client_id in self._clients]
            return {
                "X": torch.cat(features, dim=0).clone(),
                "y": torch.cat(targets, dim=0).clone(),
                "split": "train",
            }

        return {
            "X": self._global_test_data["X"].clone(),
            "y": self._global_test_data["y"].clone(),
            "split": split or "test",
        }

    def get_metadata(self) -> dict[str, Any]:
        """Return deterministic synthetic dataset metadata."""

        return {
            "name": "synthetic_classification",
            "num_clients": self.num_clients,
            "samples_per_client": self.samples_per_client,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "seed": self.seed,
            "client_ids": list(self._clients),
            "client_test_source": "partitioned_global_test",
            "label_rule": "linear_teacher",
        }

    def _build_client_data(self) -> dict[str, dict[str, Tensor]]:
        generator = torch.Generator().manual_seed(self.seed)
        client_data: dict[str, dict[str, Tensor]] = {}
        for index, client_id in enumerate(self._clients):
            features = torch.randn(
                self.samples_per_client,
                self.input_dim,
                generator=generator,
            ) + float(index)
            # Labelled from the features as stored, shift included, so one rule
            # holds for every client and for the global test set. The teacher is
            # blind to the shift by construction, so it stays what it claims to
            # be -- covariate heterogeneity -- instead of collapsing the client's
            # label distribution.
            targets = synthetic_labels(features, self._teacher, generator)
            client_data[client_id] = {"X": features, "y": targets}
        return client_data

    def _build_global_test_data(self) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.seed + 1000)
        num_examples = self.num_clients * self.samples_per_client
        features = torch.randn(num_examples, self.input_dim, generator=generator)
        targets = synthetic_labels(features, self._teacher, generator)
        return {"X": features, "y": targets}

    def _split_global_test_data(self) -> dict[str, dict[str, Tensor]]:
        client_test_data: dict[str, dict[str, Tensor]] = {}
        for index, client_id in enumerate(self._clients):
            start = index * self.samples_per_client
            stop = start + self.samples_per_client
            client_test_data[client_id] = {
                "X": self._global_test_data["X"][start:stop],
                "y": self._global_test_data["y"][start:stop],
            }
        return client_test_data
