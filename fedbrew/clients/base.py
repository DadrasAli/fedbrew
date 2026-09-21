"""Abstract client update contract for federated learning experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from fedbrew.core.protocol import ClientInfo, EvalRequest, EvalResult, FitRequest, FitResult


class ClientUpdate(ABC):
    """Task-agnostic contract implemented by client-side update logic."""

    @abstractmethod
    def setup(self, client_info: ClientInfo) -> None:
        """Prepare a client before it participates in rounds."""

    @abstractmethod
    def fit(self, request: FitRequest) -> FitResult:
        """Run local fitting for a fit request."""

    @abstractmethod
    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Run local evaluation for an evaluation request."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of client state."""

    @abstractmethod
    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore client state from a snapshot."""
