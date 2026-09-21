"""Abstract server strategy contract for federated learning experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fedbrew.core.protocol import ClientInfo, EvalResult, FitRequest, FitResult, RoundInfo


class ServerStrategy(ABC):
    """Task-agnostic contract implemented by server-side strategies."""

    #: Whether ``aggregate_stream`` consumes results without buffering them.
    #: Strategies that can accumulate incrementally set this to True so the loop
    #: can release each client's model state as soon as it is aggregated.
    streaming_aggregation = False

    @abstractmethod
    def initialize(self) -> dict[str, Any]:
        """Initialize server state and return the initial payload."""

    @abstractmethod
    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        """Create fit requests for the selected clients in a round."""

    @abstractmethod
    def aggregate(
        self,
        round_info: RoundInfo,
        results: Sequence[FitResult],
    ) -> dict[str, Any]:
        """Aggregate client fit results into an updated server payload."""

    def aggregate_stream(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Aggregate fit results supplied as a one-shot iterable.

        The default buffers the iterable and delegates to ``aggregate``, which
        keeps every client model state alive at once. Strategies that override
        this to accumulate incrementally must also set ``streaming_aggregation``.
        """

        return self.aggregate(round_info, list(results))

    @abstractmethod
    def evaluate(
        self,
        round_info: RoundInfo,
        results: Sequence[EvalResult],
    ) -> dict[str, float]:
        """Aggregate evaluation results into server-side metrics."""

    @abstractmethod
    def save_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of server strategy state."""

    @abstractmethod
    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore server strategy state from a snapshot."""
