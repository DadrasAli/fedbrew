"""A client the pool builds to be asked about is not one the run used, until it is used.

``select_executor`` builds the roster's first client to ask its rule whether it
can be batched. A checkpoint holds the state of every client the run used, so
a run that falls back to the sequential executor by default must not hold that
client's state unless it fits or evaluates it: it is the run it was before
there was a default. Held here, on the pool itself:

- ``ask`` builds a client once, and any later access returns that client
  without building another;
- an asked client is not in ``get_state_snapshot`` until it is accessed or
  counted ``used``, and a state it was given by ``load_state_snapshot`` is;
- evicting an asked client keeps no state of it.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest

from fedbrew.clients.lazy_pool import LazyClientPool

pytestmark = pytest.mark.fast


class _Client:
    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.state: dict[str, Any] = {"id": client_id}

    def setup(self, info: Any) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return dict(self.state)

    def load_state(self, state: dict[str, Any]) -> None:
        self.state = dict(state)


def _pool() -> tuple[LazyClientPool, list[str]]:
    built: list[str] = []

    def build(client_id: str) -> _Client:
        built.append(client_id)
        return _Client(client_id)

    return LazyClientPool(["a", "b", "c"], build), built


class AskedClientTest(unittest.TestCase):
    def test_built_once_and_held_out_of_the_snapshot_until_used(self) -> None:
        pool, built = _pool()
        asked = pool.ask("a")
        self.assertIs(pool.ask("a"), asked)
        self.assertEqual(built, ["a"])
        self.assertEqual(pool.get_state_snapshot(), {})
        self.assertIs(pool["a"], asked)
        self.assertEqual(built, ["a"])
        self.assertEqual(pool.get_state_snapshot(), {"a": {"id": "a"}})

    def test_counted_used_it_is_in_the_snapshot(self) -> None:
        pool, _ = _pool()
        pool.ask("a")
        pool.used("a")
        self.assertEqual(list(pool.get_state_snapshot()), ["a"])

    def test_a_client_already_built_is_not_asked_about(self) -> None:
        pool, built = _pool()
        pool["b"]
        pool.ask("b")
        self.assertEqual(built, ["b"])
        self.assertEqual(list(pool.get_state_snapshot()), ["b"])

    def test_evicted_unused_it_keeps_nothing(self) -> None:
        pool, _ = _pool()
        pool.ask("a")
        pool.evict_client("a")
        self.assertEqual(pool.get_state_snapshot(), {})
        self.assertEqual(pool.materialized_client_ids, [])

    def test_a_state_it_was_given_is_kept(self) -> None:
        pool, _ = _pool()
        pool.load_state_snapshot({"a": {"id": "a", "rounds": 3}})
        pool.ask("a")
        self.assertEqual(pool.get_state_snapshot(), {"a": {"id": "a", "rounds": 3}})


if __name__ == "__main__":
    unittest.main()
