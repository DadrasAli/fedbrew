"""Abstract dataset contract for federated benchmark data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sized
from typing import Any


class FederatedDataset(ABC):
    """Task-agnostic access pattern for federated datasets."""

    @abstractmethod
    def list_clients(self) -> list[str]:
        """Return available federated client identifiers."""

    @abstractmethod
    def get_client_data(self, client_id: str) -> Any:
        """Return data associated with one federated client."""

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        """Return lightweight metadata for one client.

        Datasets without a metadata index retain compatibility through this
        fallback, which may load the client's data to infer its size.
        """

        payload = self.get_client_data(client_id)
        return {
            "client_id": client_id,
            "num_examples": _infer_num_examples(payload),
        }

    @abstractmethod
    def get_global_data(self) -> Any:
        """Return global data used outside client-local training."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata."""


def _infer_num_examples(payload: Any) -> int:
    if isinstance(payload, Mapping):
        explicit = payload.get("num_examples")
        if isinstance(explicit, int) and not isinstance(explicit, bool):
            return explicit

        split_total = 0
        found_split = False
        for split_name in ("train", "eval"):
            split = payload.get(split_name)
            if isinstance(split, Mapping):
                split_count = _infer_num_examples(split)
                split_total += split_count
                found_split = found_split or split_count > 0
        if found_split:
            return split_total

        for key in ("y", "X", "x", "data"):
            value = payload.get(key)
            if isinstance(value, Sized):
                return len(value)

    if isinstance(payload, Sized):
        return len(payload)
    return 0
