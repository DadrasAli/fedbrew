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
    ) -> None:
        """Register the roster without constructing any client.

        Args:
            client_ids: Every client in the run, in roster order. Must be
                unique.
            client_factory: Called with one client id to build that client, at
                most once per id. Its result is cached for the rest of the run.

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
        self._clients: dict[str, ClientUpdate] = {}
        self._client_infos: dict[str, ClientInfo] = {}
        self._saved_states: dict[str, dict[str, Any]] = {}

    def __getitem__(self, client_id: str) -> ClientUpdate:
        if client_id not in self._client_id_set:
            raise KeyError(client_id)
        existing = self._clients.get(client_id)
        if existing is not None:
            return existing

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
        """Return known states without materializing additional clients."""

        snapshot = {client_id: dict(state) for client_id, state in self._saved_states.items()}
        for client_id, client in self._clients.items():
            snapshot[client_id] = dict(client.get_state())
        return snapshot

    def release_client(self, client_id: str) -> None:
        """Snapshot and evict one materialized client to bound memory use."""

        client = self._clients.pop(client_id, None)
        if client is not None:
            self._saved_states[client_id] = dict(client.get_state())
