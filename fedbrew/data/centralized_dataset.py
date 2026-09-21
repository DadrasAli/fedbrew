"""Centralized-baseline view over a federated dataset.

Wraps any :class:`FederatedDataset` and exposes it as a single client holding
the concatenation of every source client's splits. Training against this view
is the centralized upper bound for a federated run: identical data, identical
model, identical local-update engine, but no partitioning and no aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from fedbrew.core.refusal import RunRefused
from fedbrew.data.cached_payload import CachedPayload
from fedbrew.data.dataset import FederatedDataset

#: Client ID the pooled view reports. It appears in client_metrics.csv and in
#: every round's selected-client list, so it is readable rather than cryptic.
CENTRALIZED_CLIENT_ID = "centralized"

#: Splits carried through from the source clients, in shard order.
POOLED_SPLITS = ("train", "eval", "test")


class CentralizedFederatedDataset(FederatedDataset):
    """Expose a federated dataset as one client owning all clients' data.

    Source shards are read exactly once, on the first access, and concatenated
    per split. Construct the wrapped dataset without a shard cache: caching
    would hold a second copy of data this view already keeps resident.
    """

    def __init__(
        self,
        base: FederatedDataset,
        client_id: str = CENTRALIZED_CLIENT_ID,
    ) -> None:
        """Wrap a federated dataset as a single pooled client.

        Args:
            base: The dataset to pool. Construct it with
                ``shard_cache_bytes=0``: this view reads every shard once and
                keeps the concatenation resident, so a shard cache would only
                hold a second copy.
            client_id: Identifier for the single pooled client.

        Raises:
            RunRefused: If ``base`` has no clients to pool.

        Pooling is deferred: the source shards are read and concatenated on
        first access, not here. Averaging one client's result is the identity,
        so a ``centralized`` arm differs from a FedAvg arm only in how the data
        is partitioned -- the local-update code path is the same one, which is
        what stops the baseline drifting from the arms it is a baseline for.
        """

        self.base = base
        self.client_id = str(client_id)
        self.source_client_ids = list(base.list_clients())
        if not self.source_client_ids:
            raise RunRefused("centralized training requires at least one source client")
        self._pooled: CachedPayload | None = None

    def list_clients(self) -> list[str]:
        """Return the single pooled client ID."""

        return [self.client_id]

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        """Return every source client's data pooled into one client payload.

        The pool is read once and served on every round, so it is served as a
        private structure over the pooled tensors rather than as itself; see
        :class:`CachedPayload`.
        """

        self._check_client_id(client_id)
        return self._pool().serve(f"centralized client {self.client_id!r}")

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        """Return pooled split sizes for the single client."""

        self._check_client_id(client_id)
        pooled = self._pool().payload
        return {
            "client_id": self.client_id,
            "num_examples": int(pooled["num_examples"]),
            "num_train_examples": int(pooled["num_train_examples"]),
            "num_eval_examples": int(pooled["num_eval_examples"]),
            "num_test_examples": int(pooled["num_test_examples"]),
            "metadata": dict(cast(Mapping[str, Any], pooled["metadata"])),
        }

    def get_global_data(self) -> Any:
        """Return the source dataset's centralized test shard unchanged."""

        return self.base.get_global_data()

    def get_metadata(self) -> dict[str, Any]:
        """Return source metadata annotated with the pooling that was applied."""

        metadata = dict(self.base.get_metadata())
        metadata.update(
            centralized=True,
            centralized_client_id=self.client_id,
            num_source_clients=len(self.source_client_ids),
        )
        return metadata

    def _check_client_id(self, client_id: str) -> None:
        if client_id != self.client_id:
            raise KeyError(
                f"Unknown client_id: {client_id}; the centralized view exposes "
                f"only {self.client_id!r}"
            )

    def _pool(self) -> CachedPayload:
        if self._pooled is not None:
            return self._pooled

        parts: dict[str, list[Mapping[str, Any]]] = {split: [] for split in POOLED_SPLITS}
        for source_client_id in self.source_client_ids:
            client_data = self.base.get_client_data(source_client_id)
            for split in POOLED_SPLITS:
                split_data = _split_tensors(client_data, split)
                if split_data is not None:
                    parts[split].append(split_data)

        pooled: dict[str, Any] = {
            split: _concatenate_splits(parts[split], split) for split in POOLED_SPLITS
        }
        counts = {split: _num_examples(pooled[split]) for split in POOLED_SPLITS}
        if counts["train"] <= 0:
            raise RunRefused("centralized training found no pooled train examples")

        pooled.update(
            num_examples=counts["train"] + counts["eval"],
            num_train_examples=counts["train"],
            num_eval_examples=counts["eval"],
            num_test_examples=counts["test"],
            metadata={
                "centralized": True,
                "num_source_clients": len(self.source_client_ids),
            },
        )
        self._pooled = CachedPayload(pooled)
        return self._pooled


def _split_tensors(client_data: Any, split: str) -> Mapping[str, Any] | None:
    """Return one non-empty split of a client payload, if it has one."""

    if not isinstance(client_data, Mapping):
        return None

    split_data = client_data.get(split)
    if isinstance(split_data, Mapping):
        return split_data if _num_examples(split_data) > 0 else None

    # Old-format shards store x/y at the top level and carry only train data.
    if split == "train" and _num_examples(client_data) > 0:
        return client_data
    return None


def _concatenate_splits(
    parts: list[Mapping[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Concatenate one split's feature/target tensors across source clients."""

    if not parts:
        return {}

    import torch

    features = []
    targets = []
    for index, part in enumerate(parts):
        feature, target = _feature_target(part, split, index)
        features.append(feature)
        targets.append(target)
    return {"x": torch.cat(features, dim=0), "y": torch.cat(targets, dim=0)}


def _feature_target(part: Mapping[str, Any], split: str, index: int) -> tuple[Any, Any]:
    feature = part.get("x", part.get("X"))
    target = part.get("y")
    if feature is None or target is None:
        raise RunRefused(f"source client {index} is missing x/y tensors in its {split} split")
    return feature, target


def _num_examples(data: Any) -> int:
    if not isinstance(data, Mapping):
        return 0
    target = data.get("y")
    if hasattr(target, "__len__"):
        return len(cast(Any, target))
    feature = data.get("x", data.get("X"))
    if hasattr(feature, "__len__"):
        return len(cast(Any, feature))
    return 0
