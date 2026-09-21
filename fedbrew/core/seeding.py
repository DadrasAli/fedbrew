"""Deterministic seed derivation.

A leaf module on purpose: it depends on nothing but hashlib, so anything that
needs a derived seed -- servers, clients, data generators -- can import it
without pulling in the config layer, which imports the server registry, which
would close a cycle.

Every helper here hashes rather than adds. Adding a base seed to an index makes
neighbouring seeds the same sequence read from different offsets:
``Random(42 + 2)`` and ``Random(43 + 1)`` are one generator, so "replicates"
seeded 42/43/44 share their draws shifted by a round. Hashing has no such
structure, and the length-prefixed field encoding below keeps ("ab", "c") from
colliding with ("a", "bc").
"""

from __future__ import annotations

import hashlib
from typing import Any


def _update_seed_hash(hasher: Any, label: str, value: object) -> None:
    text = f"{label}:{type(value).__module__}.{type(value).__qualname__}:{value}"
    encoded = text.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    hasher.update(encoded)


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed from a base seed and arbitrary parts."""

    hasher = hashlib.sha256()
    _update_seed_hash(hasher, "base_seed", int(base_seed))
    for index, part in enumerate(parts):
        _update_seed_hash(hasher, f"part_{index}", part)
    return int.from_bytes(hasher.digest()[:4], byteorder="big", signed=False)


def client_seed(base_seed: int, round_id: int, client_id: str) -> int:
    """Return a deterministic seed for one client in one FL round."""

    return derive_seed(base_seed, "round", round_id, "client", client_id)


def dataloader_seed(base_seed: int, round_id: int, client_id: str, phase: str) -> int:
    """Return a deterministic seed for a client DataLoader phase."""

    return derive_seed(
        base_seed,
        "round",
        round_id,
        "client",
        client_id,
        "dataloader",
        phase,
    )
