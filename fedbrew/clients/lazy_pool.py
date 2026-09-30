"""Lazy mapping for materializing federated clients on first use."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.protocol import ClientInfo
from fedbrew.core.refusal import RunRefused


class LazyClientPool(Mapping[str, ClientUpdate]):
    """Create and cache clients only when their IDs are accessed."""

    def __init__(
        self,
        client_ids: list[str],
        client_factory: Callable[[str], ClientUpdate],
        keep_resident: Callable[[str], bool] | None = None,
    ) -> None:
        """Register the roster without constructing any client.

        Args:
            client_ids: Every client in the run, in roster order. Must be
                unique.
            client_factory: Called with one client id to build that client.
                Its result is kept until the client is released.
            keep_resident: Whether a client may stay built past
                :meth:`release_client`, asked each time; also called on every
                access to a built client. ``None`` releases every client asked,
                which is what the pool always did. The factory passes the
                dataset's ``touch_shard``: a client stays built while its shard
                is in the shard cache, and the cache's eviction releases it
                (:meth:`evict_client`).

        Raises:
            ValueError: If ``client_ids`` contains a duplicate.

        Construction is deferred because at low participation most clients are
        never selected: building all of them up front would pay the per-client
        setup cost for a roster the run mostly ignores. A client's saved state
        survives here even while the client itself has not been built, so a
        resume can restore state for a client that has not yet been touched.
        """

        self._client_ids = tuple(client_ids)
        self._client_id_set = set(self._client_ids)
        if len(self._client_id_set) != len(self._client_ids):
            raise ValueError("client_ids must be unique")
        self._client_factory = client_factory
        self._keep_resident = keep_resident
        self._clients: dict[str, ClientUpdate] = {}
        self._client_infos: dict[str, ClientInfo] = {}
        self._saved_states: dict[str, dict[str, Any]] = {}

    def __getitem__(self, client_id: str) -> ClientUpdate:
        if client_id not in self._client_id_set:
            raise KeyError(client_id)
        existing = self._clients.get(client_id)
        if existing is not None:
            if self._keep_resident is None or self._keep_resident(client_id):
                return existing
            # Its shard left the cache without the eviction reaching the pool.
            self.evict_client(client_id)

        client = self._client_factory(client_id)
        client_info = self._client_infos.get(client_id)
        if client_info is not None:
            client.setup(client_info)
        saved_state = self._saved_states.get(client_id)
        if saved_state is not None:
            client.load_state(saved_state)
        self._clients[client_id] = client
        return client

    def __iter__(self) -> Iterator[str]:
        return iter(self._client_ids)

    def __len__(self) -> int:
        return len(self._client_ids)

    def __contains__(self, client_id: object) -> bool:
        return client_id in self._client_id_set

    def is_built(self, client_id: str) -> bool:
        """Whether ``client_id``'s client is built now."""

        return client_id in self._clients

    def forget(self, client_id: str) -> None:
        """Drop a built client and keep nothing of it: for one built only to be asked about.

        Unlike :meth:`evict_client`, no snapshot of it is kept, so the
        checkpoints (:meth:`get_state_snapshot`) hold what they would have held
        had it never been built. A state it was given (:meth:`load_state_snapshot`)
        stays.
        """

        self._clients.pop(client_id, None)

    @property
    def materialized_client_ids(self) -> list[str]:
        """Return client IDs already constructed, in dataset order."""

        return [client_id for client_id in self._client_ids if client_id in self._clients]

    def setup_client_infos(self, client_infos: list[ClientInfo]) -> None:
        """Store setup metadata and configure already-materialized clients."""

        self._client_infos = {
            info.client_id: info for info in client_infos if info.client_id in self._client_id_set
        }
        for client_id, client in self._clients.items():
            client_info = self._client_infos.get(client_id)
            if client_info is not None:
                client.setup(client_info)

    def load_state_snapshot(
        self,
        client_states: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Store checkpoint states without constructing untouched clients."""

        for raw_client_id, client_state in client_states.items():
            client_id = str(raw_client_id)
            if client_id not in self._client_id_set:
                continue
            if not isinstance(client_state, Mapping):
                raise RunRefused("checkpoint client state must be a mapping")
            saved_state = dict(client_state)
            self._saved_states[client_id] = saved_state
            client = self._clients.get(client_id)
            if client is not None:
                client.load_state(saved_state)

    def get_state_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return known states without materializing additional clients.

        In roster order, whichever clients are built: a checkpoint stacks the
        states in this order (``stack_client_states``), so an order that
        followed which clients happened to be resident would make the same
        states a different file.
        """

        snapshot = {}
        for client_id in self._client_ids:
            client = self._clients.get(client_id)
            if client is not None:
                snapshot[client_id] = dict(client.get_state())
            elif client_id in self._saved_states:
                snapshot[client_id] = dict(self._saved_states[client_id])
        return snapshot

    def release_client(self, client_id: str) -> None:
        """Let one built client go, unless it may stay resident.

        Every client used to be evicted here after each evaluation and rebuilt
        for its next fit: the constructor, its setup and a load_state of its own
        snapshot, per client per round -- 20% of an MNIST MLP round at 1000
        clients (measured on 2026-09-26). A client whose shard is in the shard cache
        now stays built; it holds nothing the cache does not already hold, so
        memory stays under the cache's budget.
        """

        if self._keep_resident is not None and self._keep_resident(client_id):
            return
        self.evict_client(client_id)

    def evict_client(self, client_id: str) -> None:
        """Snapshot one built client's state and drop the client, to bound memory use.

        What was built from the snapshot later is the client that was dropped:
        the state it carries across rounds is its get_state, and everything
        else it is built from the config and its shard.
        """

        client = self._clients.pop(client_id, None)
        if client is not None:
            self._saved_states[client_id] = dict(client.get_state())
