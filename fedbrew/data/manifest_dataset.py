"""Manifest-backed federated dataset runtime loader."""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from fedbrew.core.refusal import RunRefused
from fedbrew.data.cached_payload import CachedPayload
from fedbrew.data.dataset import FederatedDataset
from fedbrew.data.writers.torch_shards import load_client_shard

#: Default in-memory budget for cached client shards, in bytes. Cross-device
#: rounds touch every client each round, and on a cluster the shards live on a
#: shared filesystem, so re-reading them per round leaves the GPU waiting on I/O.
#: FEMNIST's full writer partition is well under this budget.
DEFAULT_SHARD_CACHE_BYTES = 4 * 1024**3


class ManifestFederatedDataset(FederatedDataset):
    """Federated dataset that reads generated shard manifests.

    Loaded shards are kept in a least-recently-used cache bounded by
    ``shard_cache_bytes``; pass ``0`` to read from disk on every access.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        shard_cache_bytes: int = DEFAULT_SHARD_CACHE_BYTES,
    ) -> None:
        """Open a generated dataset and read its client roster.

        Args:
            manifest_path: Path to the generator's ``manifest.json``. Shard
                paths inside it are resolved relative to its parent directory,
                so the manifest and its ``shards/`` directory move together.
            shard_cache_bytes: In-memory LRU budget for decoded client shards,
                in bytes. 0 re-reads from disk on every access. The cache only
                pays off from the second round onward, since round one is the
                pass that fills it. A cached shard is served as a private
                structure over the cached tensors, so caching cannot change
                results; see :class:`CachedPayload`.

        Raises:
            RunRefused: If the manifest or its client roster cannot be read or
                decoded, a roster row lacks ``client_id`` or ``shard``, or two
                roster rows share a ``client_id``. A duplicate
                would make the round loop train one client twice under two
                separate FedAvg weights while another was never trained, so the
                roster is refused rather than silently de-duplicated.
        """

        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self._manifest = _load_json(self.manifest_path)
        self._clients = self._load_clients()
        self._clients_by_id = {str(client["client_id"]): client for client in self._clients}
        if len(self._clients_by_id) != len(self._clients):
            # list_clients returns one entry per row and this map keeps one per
            # id, so a duplicate makes the loop select a client whose shard
            # belongs to a different one -- training it twice a round under two
            # FedAvg weights. Refuse the roster rather than silently drop rows.
            counts = Counter(str(client["client_id"]) for client in self._clients)
            duplicates = sorted(client_id for client_id, count in counts.items() if count > 1)
            raise RunRefused(
                f"{self.manifest_path}: clients.jsonl repeats "
                f"{len(duplicates)} client_id(s): {', '.join(duplicates[:5])}"
                + (" ..." if len(duplicates) > 5 else "")
                + ". Each id has one shard, so the repeats would train that "
                "shard more than once a round."
            )
        self.shard_cache_bytes = max(0, int(shard_cache_bytes))
        self._shard_cache: OrderedDict[str, CachedPayload] = OrderedDict()
        self._shard_cache_bytes_used = 0

    def _load_shard_cached(self, client_id: str, shard_path: Path) -> dict[str, Any]:
        """Return a client shard, reading from disk only on a cache miss.

        The two shards this never caches -- the disabled cache and one too big
        for the budget -- are returned as loaded, because nothing else holds
        them. Everything the cache keeps is served through
        :meth:`CachedPayload.serve`.
        """

        if self.shard_cache_bytes == 0:
            return load_client_shard(shard_path)

        cached = self._shard_cache.get(client_id)
        if cached is not None:
            self._shard_cache.move_to_end(client_id)
            return cached.serve(self._shard_label(client_id))

        shard = load_client_shard(shard_path)
        shard_bytes = _shard_nbytes(shard)
        if shard_bytes > self.shard_cache_bytes:
            return shard

        entry = CachedPayload(shard)
        self._shard_cache[client_id] = entry
        self._shard_cache_bytes_used += shard_bytes
        while self._shard_cache_bytes_used > self.shard_cache_bytes and len(self._shard_cache) > 1:
            _, evicted = self._shard_cache.popitem(last=False)
            self._shard_cache_bytes_used -= _shard_nbytes(evicted.payload)
        return entry.serve(self._shard_label(client_id))

    def _shard_label(self, client_id: str) -> str:
        """Name one cached shard, for a mutation error that has to be traceable."""

        return f"{self.manifest_path}: client {client_id!r}"

    def list_clients(self) -> list[str]:
        """Return client IDs from clients.jsonl."""

        return [str(client["client_id"]) for client in self._clients]

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        """Return indexed client metadata without loading its shard."""

        if client_id not in self._clients_by_id:
            raise KeyError(f"Unknown client_id: {client_id}")
        metadata = dict(self._clients_by_id[client_id])
        # num_examples is every split. The per-split keys are authoritative;
        # the fallbacks below only fire for a legacy record that declares the
        # total alone, where all of it is train.
        num_examples = int(metadata.get("num_examples", 0))
        num_test_examples = int(metadata.get("num_test_examples", 0))
        num_train_examples = int(metadata.get("num_train_examples", num_examples))
        num_eval_examples = int(
            metadata.get(
                "num_eval_examples",
                max(0, num_examples - num_train_examples - num_test_examples),
            )
        )
        return {
            "client_id": client_id,
            "num_examples": num_examples,
            "num_train_examples": num_train_examples,
            "num_eval_examples": num_eval_examples,
            "num_test_examples": num_test_examples,
            "metadata": metadata,
        }

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        """Load one client shard with backward-compatible split handling."""

        if client_id not in self._clients_by_id:
            raise KeyError(f"Unknown client_id: {client_id}")
        metadata = self._clients_by_id[client_id]
        shard = self._load_shard_cached(client_id, self.root / str(metadata["shard"]))
        if _is_split_shard(shard):
            train_data = cast(dict[str, Any], shard["train"])
            eval_data = cast(dict[str, Any], shard.get("eval", {}))
            test_data = cast(dict[str, Any], shard.get("test", {}))
            return {
                "train": train_data,
                "eval": eval_data,
                "test": test_data,
                "num_examples": int(
                    metadata.get(
                        "num_examples",
                        _num_examples(train_data)
                        + _num_examples(eval_data)
                        + _num_examples(test_data),
                    )
                ),
                "num_train_examples": int(
                    metadata.get("num_train_examples", _num_examples(train_data))
                ),
                "num_eval_examples": int(
                    metadata.get("num_eval_examples", _num_examples(eval_data))
                ),
                "num_test_examples": int(
                    metadata.get("num_test_examples", _num_examples(test_data))
                ),
                "metadata": dict(metadata),
            }

        # Safe to write into: a cached shard is served as a private structure,
        # and an uncached one is this call's own load.
        shard["num_examples"] = int(metadata.get("num_examples", _num_examples(shard)))
        return shard

    def get_global_data(self, split: str | None = "test") -> Any:
        """Load the global test shard when the manifest provides one."""

        if split not in {None, "test"}:
            raise ValueError(f"Unsupported global split: {split}")
        global_test_path = self._manifest.get("global_test")
        if not isinstance(global_test_path, str) or not global_test_path:
            return None
        shard = load_client_shard(self.root / global_test_path)
        shard["split"] = "test"
        return shard

    def get_metadata(self) -> dict[str, Any]:
        """Return manifest metadata."""

        metadata = dict(self._manifest)
        metadata["manifest_path"] = str(self.manifest_path)
        return metadata

    def _load_clients(self) -> list[dict[str, Any]]:
        """Read the client roster the manifest names, one JSON object per line.

        Refused where it is read, naming the file and the line, like the
        manifest and the shards: a roster that is missing, truncated, or has a
        row without a key the dataset reads is input, not a defect.
        """

        if "clients_file" not in self._manifest:
            raise RunRefused(f"the dataset manifest {self.manifest_path} names no clients_file")
        clients_path = self.root / str(self._manifest["clients_file"])
        clients = []
        try:
            with clients_path.open("r", encoding="utf-8") as file:
                for number, line in enumerate(file, start=1):
                    if line.strip():
                        clients.append(_roster_row(clients_path, number, line))
        except (OSError, UnicodeDecodeError) as error:
            reason = error.strerror if isinstance(error, OSError) else None
            raise RunRefused(
                f"cannot read the client roster {clients_path}: {reason or error}"
            ) from error
        return clients


def _shard_nbytes(shard: Mapping[str, Any]) -> int:
    """Return the approximate resident size of a loaded shard in bytes."""

    total = 0
    for value in shard.values():
        if isinstance(value, Mapping):
            total += _shard_nbytes(value)
        elif hasattr(value, "nbytes"):
            total += int(cast(Any, value).nbytes)
    return total


def _is_split_shard(shard: Mapping[str, Any]) -> bool:
    train = shard.get("train")
    return isinstance(train, Mapping)


def _num_examples(data: Mapping[str, Any]) -> int:
    targets = data.get("y")
    if hasattr(targets, "__len__"):
        return len(cast(Any, targets))
    features = data.get("x", data.get("X"))
    if hasattr(features, "__len__"):
        return len(cast(Any, features))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    # Refused at the read rather than checked ahead of it: a path can go between
    # a check and the read, and a file that exists can still be truncated.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RunRefused(
            f"cannot read the dataset manifest {path}: {error.strerror or error}"
        ) from error
    except ValueError as error:
        raise RunRefused(f"the dataset manifest {path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise RunRefused(f"Expected JSON object in {path}")
    return cast(dict[str, Any], data)


#: The keys the dataset reads from every roster row: construction indexes the
#: roster by the first, and `get_client_data` opens the second.
ROSTER_ROW_KEYS = ("client_id", "shard")


def _roster_row(path: Path, number: int, line: str) -> dict[str, Any]:
    """One roster line as the dataset uses it, or a refusal naming the line."""

    try:
        row = json.loads(line)
    except ValueError as error:
        raise RunRefused(
            f"the client roster {path} line {number} is not valid JSON: {error}"
        ) from error
    if not isinstance(row, dict):
        raise RunRefused(f"the client roster {path} line {number} is not a JSON object")
    missing = [key for key in ROSTER_ROW_KEYS if key not in row]
    if missing:
        raise RunRefused(f"the client roster {path} line {number} has no {', '.join(missing)}")
    return cast(dict[str, Any], row)
