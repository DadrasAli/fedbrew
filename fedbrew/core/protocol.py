"""Shared protocol objects exchanged by benchmark components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FitRequest:
    """Request sent by a server to ask a client to run local fitting."""

    round_id: int
    client_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    #: T, the rounds in the run, copied from ``RoundInfo.total_rounds`` by the
    #: server that builds the request. None for a request built outside a run.
    total_rounds: int | None = None


@dataclass(slots=True)
class FitResult:
    """Result returned by a client after local fitting."""

    round_id: int
    client_id: str
    num_examples: int
    payload: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRequest:
    """Request sent by a server to ask a client to run evaluation."""

    round_id: int
    client_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    """Result returned by a client after evaluation."""

    round_id: int
    client_id: str
    num_examples: int
    metrics: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClientInfo:
    """Task-agnostic metadata describing one federated client."""

    client_id: str
    num_examples: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoundInfo:
    """Task-agnostic metadata describing one federated round."""

    round_id: int
    payload: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    #: T, the rounds in the run. The loop sets it every round; None for a
    #: RoundInfo built outside a run.
    total_rounds: int | None = None
