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


def _seed_field(label: str, value: object) -> bytes:
    text = f"{label}:{type(value).__module__}.{type(value).__qualname__}:{value}"
    encoded = text.encode("utf-8")
    return len(encoded).to_bytes(8, byteorder="big", signed=False) + encoded


def _update_seed_hash(hasher: Any, label: str, value: object) -> None:
    hasher.update(_seed_field(label, value))


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


def dataloader_seeds(base_seed: int, round_id: int, client_ids: list[str], phase: str) -> list[int]:
    """``dataloader_seed`` for each client, the fields they share hashed once.

    The same digests: a hash is taken over the fields in order, so the state
    after the shared prefix is copied for each client and the rest added.
    """

    prefix = hashlib.sha256()
    prefix.update(
        _seed_field("base_seed", int(base_seed))
        + _seed_field("part_0", "round")
        + _seed_field("part_1", round_id)
        + _seed_field("part_2", "client")
    )
    suffix = _seed_field("part_4", "dataloader") + _seed_field("part_5", phase)
    seeds = []
    for client_id in client_ids:
        hasher = prefix.copy()
        hasher.update(_seed_field("part_3", client_id) + suffix)
        seeds.append(int.from_bytes(hasher.digest()[:4], byteorder="big", signed=False))
    return seeds
