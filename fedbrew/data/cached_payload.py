"""Serve a memoised shard payload without handing over the memoised object."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CachedPayload:
    """A decoded shard a dataset keeps and serves more than once.

    Two datasets memoise tensors: ``ManifestFederatedDataset`` in an LRU keyed
    by client, and ``CentralizedFederatedDataset`` as the single pooled client.
    A memoised payload served by reference is one the caller can edit for every
    later round -- ``data["x"] = data["x"].to(device)`` replaces the cached
    tensor, and an in-place normalisation rescales it. Round one would then
    train on the shard as generated and round two on whatever round one left
    behind, with nothing between the two saying so.

    So a served payload is a private structure over shared tensors:

    - Every mapping in it is rebuilt, so rebinding a key edits the caller's copy
      and leaves the cache alone. This is the half that can be prevented.
    - The tensors are the cached ones, because serving copies of them is what
      the cache exists to avoid. An in-place edit therefore still reaches the
      cache, and cannot be prevented -- torch has no read-only tensor. It is
      detected instead, on the next serve, from the per-tensor version counter.

    The detection is one serve late by construction, which is the point: an edit
    to a payload that is never served again changes no later round, and is not
    what the cache made possible.
    """

    __slots__ = ("payload", "_versions")

    def __init__(self, payload: dict[str, Any]) -> None:
        """Take ownership of a decoded payload and record its tensor versions."""

        self.payload = payload
        self._versions = _tensor_versions(payload)

    def serve(self, label: str) -> dict[str, Any]:
        """Return a private view of the payload, refusing an edited one.

        Args:
            label: What is being served, for the error message -- a manifest
                path and client, or the pooled view's client id.

        Raises:
            RuntimeError: A tensor was edited in place since the last serve.
        """

        # strict=True: nothing outside this object holds the cached structure,
        # so a differing length would mean the payload itself was restructured.
        current = _tensor_versions(self.payload)
        for (path, before), (_, after) in zip(self._versions, current, strict=True):
            if before == after:
                continue
            raise RuntimeError(
                f"{label}: a cached shard tensor was edited in place after it "
                f"was served ({path}: torch version {before} -> {after}). Every "
                "later round reads the edited tensor, so the run would train on "
                "data that changed under it. Build new tensors from a served "
                "shard rather than editing one."
            )
        return _copy_containers(self.payload)


def _copy_containers(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a payload's mappings, sharing everything they hold."""

    return {
        key: (_copy_containers(value) if isinstance(value, Mapping) else value)
        for key, value in payload.items()
    }


def _tensor_versions(payload: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    """Read torch's in-place counter for every tensor in a payload, by path.

    ``Tensor._version`` is torch's own record of how many in-place operations
    have touched a tensor's storage -- what autograd checks before it reuses a
    saved tensor -- and it is bumped through views. There is no public spelling
    of it, so a torch that stops exposing it leaves this returning fewer entries
    and the check passing, rather than raising on every shard;
    ``tests/test_shard_cache.py`` pins that today's torch still does.
    """

    recorded: list[tuple[str, int]] = []
    _collect_versions(payload, "", recorded)
    return tuple(recorded)


def _collect_versions(
    payload: Mapping[str, Any],
    prefix: str,
    recorded: list[tuple[str, int]],
) -> None:
    for key, value in payload.items():
        if isinstance(value, Mapping):
            _collect_versions(value, f"{prefix}{key}.", recorded)
            continue
        version = getattr(value, "_version", None)
        if isinstance(version, int):
            recorded.append((f"{prefix}{key}", version))
