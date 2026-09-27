"""A manifest client stays built while its shard is cached, and changes nothing by it.

LazyClientPool released every client after its evaluation and rebuilt it for
its next fit -- constructor, setup and a load_state of its own snapshot, per
client per round, 20% of an MNIST MLP round at 1000 clients (measured on 2026-09-26).
A client now stays built while its shard is in the shard cache, and the
cache's eviction releases it, so the cache's budget is still the bound on
client data in memory. What is pinned here:

- each client is built once, where it was rebuilt every round;
- trajectories are bit-identical with resident clients, with every client
  released as before, and with a cache small enough to evict mid-run: every
  checkpoint's model, server state, per-client state and RNG position, and
  every non-timing CSV cell, for local SGD with momentum and for SCAFFOLD,
  whose per-client control variate is the state a leak would move;
- no state leaks between clients: each resident client holds its own shard
  and its own control variate, the one a released client rebuilt from its
  snapshot holds;
- at the cap, no client outlives its shard: every built client's shard is in
  the cache, the cache is within its budget, and with the cache off no client
  stays built;
- an in-place edit of a resident client's shard is refused on its next use,
  as a rebuild's serve refused it.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.clients.lazy_pool import LazyClientPool
from fedbrew.core import factory, runner
from fedbrew.data.manifest_dataset import ManifestFederatedDataset, _shard_nbytes
from tests.test_evaluation_cadence import _trajectory
from tests.test_reproducibility import TIMING

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 6

RULES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "local_sgd": ({}, {"momentum": 0.9}),
    "scaffold": (
        {"strategy": "scaffold"},
        {
            "update_rule": "scaffold",
            **dict.fromkeys(
                (
                    "momentum",
                    "weight_decay",
                    "nesterov",
                    "learning_rate_schedule",
                    "min_learning_rate",
                )
            ),
        },
    ),
}


def _generate(root: Path) -> Path:
    from fedbrew.data.generate import generate_from_config

    generator = yaml.safe_load((REPO_ROOT / "data/configs/synthetic_label_skew.yaml").read_text())
    generator["dataset"]["output_dir"] = str(root / "data")
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(generator), encoding="utf-8")
    return Path(generate_from_config(path))


def _write(root: Path, name: str, manifest: Path, rule: str, cache_bytes: int | None) -> Path:
    config = yaml.safe_load((REPO_ROOT / "configs/dev/synthetic_label_skew.yaml").read_text())
    server, client = RULES[rule]
    config["experiment"]["output_dir"] = str(root / name)
    config["data"]["path"] = str(manifest)
    config["server"].update(server)
    config["server"]["participation_rate"] = 0.6
    config["client"].update(client)
    config["client"] = {key: value for key, value in config["client"].items() if value is not None}
    config["model"]["dropout"] = 0.3
    config["defaults"]["global_rounds"] = ROUNDS
    runtime = config["runtime"]
    runtime["device"] = "cpu"
    runtime["checkpointing"].update({"save_every_round": True, "keep_last": None})
    if cache_bytes is not None:
        runtime["performance"]["shard_cache_bytes"] = cache_bytes
    config["evaluation"] = {
        "train": {"every": 1, "clients": "all"},
        "val": {"every": 1, "clients": "all"},
        "test": {"every": 1, "clients": "all"},
        "central_test": {"every": 1},
    }
    config["client_statistics"] = {"per_client_csv": True}
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@contextmanager
def _released_as_before() -> Iterator[None]:
    """The pool as it was: every client released after its evaluation."""

    with mock.patch.object(
        factory, "_lazy_client_pool", lambda ids, build, dataset: LazyClientPool(ids, build)
    ):
        yield


@contextmanager
def _captured(into: dict[str, Any]) -> Iterator[None]:
    real = runner.build_components

    def build(config: Any) -> Any:
        into["components"] = real(config)
        return into["components"]

    with mock.patch.object(runner, "build_components", build):
        yield


def _output(path: Path) -> Path:
    return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])


class _Generated(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._class_tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._class_tmp.name)
        cls.manifest = _generate(cls.root)
        probe = ManifestFederatedDataset(cls.manifest, shard_cache_bytes=0)
        cls.shard_bytes = max(
            _shard_nbytes(probe._load_shard_cached(c, probe.root / m["shard"]))
            for c, m in probe._clients_by_id.items()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_tmp.cleanup()


class TheTrajectoryIsTheReleasedOneTest(_Generated):
    def test_resident_released_and_evicting(self) -> None:
        small = 2 * self.shard_bytes + 1
        for rule in RULES:
            with self.subTest(rule=rule):
                runner.run(_write(self.root, f"{rule}-r", self.manifest, rule, None), None)
                resident = _trajectory(_output(self.root / f"{rule}-r.yaml"))
                with _released_as_before():
                    runner.run(_write(self.root, f"{rule}-x", self.manifest, rule, None), None)
                released = _trajectory(_output(self.root / f"{rule}-x.yaml"))
                runner.run(_write(self.root, f"{rule}-s", self.manifest, rule, small), None)
                evicting = _trajectory(_output(self.root / f"{rule}-s.yaml"))
                self.assertEqual(len(resident["checkpoints"]), ROUNDS)
                self.assertTrue(any(key not in TIMING for key in resident["rounds"][0]))
                self.assertEqual(resident, released)
                self.assertEqual(evicting, released)


class EachClientIsBuiltOnceTest(_Generated):
    def _builds(self, name: str, released: bool) -> int:
        built = 0
        real = factory._lazy_client_pool

        def counting(ids: Any, build: Any, dataset: Any) -> LazyClientPool:
            def counted(client_id: str) -> Any:
                nonlocal built
                built += 1
                return build(client_id)

            if released:
                return LazyClientPool(ids, counted)
            return real(ids, counted, dataset)

        with mock.patch.object(factory, "_lazy_client_pool", counting):
            runner.run(_write(self.root, name, self.manifest, "local_sgd", None), None)
        return built

    def test_once_resident_and_every_round_released(self) -> None:
        self.assertEqual(self._builds("once-r", released=False), 5)
        self.assertGreater(self._builds("once-x", released=True), 5 * (ROUNDS - 1))


class NoStateLeaksTest(_Generated):
    def test_each_client_holds_its_own_shard_and_control(self) -> None:
        resident: dict[str, Any] = {}
        with _captured(resident):
            runner.run(_write(self.root, "leak-r", self.manifest, "scaffold", None), None)
        released: dict[str, Any] = {}
        with _released_as_before(), _captured(released):
            runner.run(_write(self.root, "leak-x", self.manifest, "scaffold", None), None)

        pool = resident["components"].clients
        self.assertIsInstance(pool, LazyClientPool)
        built = pool.materialized_client_ids
        self.assertEqual(len(built), 5, "every client stays built with every shard cached")
        dataset = resident["components"].dataset
        for client_id in built:
            client = pool._clients[client_id]
            own = dataset.get_client_data(client_id)
            self.assertTrue(torch.equal(client.client_data["train"]["y"], own["train"]["y"]))
        controls = [pool._clients[c]._client_control for c in built]
        self.assertEqual(len({id(control) for control in controls}), len(controls))

        before = released["components"].clients.get_state_snapshot()
        after = pool.get_state_snapshot()
        self.assertEqual(sorted(before), sorted(after))
        for client_id, state in before.items():
            for key, value in state["client_control"].items():
                self.assertTrue(
                    torch.equal(value, after[client_id]["client_control"][key]), client_id
                )


class AtTheCapTest(_Generated):
    def _pool_and_dataset(self, name: str, cache_bytes: int) -> tuple[Any, Any]:
        captured: dict[str, Any] = {}
        with _captured(captured):
            runner.run(_write(self.root, name, self.manifest, "local_sgd", cache_bytes), None)
        return captured["components"].clients, captured["components"].dataset

    def test_no_client_outlives_its_shard(self) -> None:
        budget = 2 * self.shard_bytes + 1
        pool, dataset = self._pool_and_dataset("cap", budget)
        built = pool.materialized_client_ids
        self.assertTrue(built)
        self.assertLessEqual(len(built), 2)
        self.assertLessEqual(dataset._shard_cache_bytes_used, budget)
        for client_id in built:
            self.assertIn(client_id, dataset._shard_cache)

    def test_with_the_cache_off_none_stays_built(self) -> None:
        pool, _ = self._pool_and_dataset("off", 0)
        self.assertEqual(pool.materialized_client_ids, [])


class AnEditedShardIsRefusedTest(_Generated):
    def test_on_the_next_use(self) -> None:
        dataset = ManifestFederatedDataset(self.manifest)
        pool = LazyClientPool(
            dataset.list_clients(),
            lambda client_id: mock.Mock(client_data=dataset.get_client_data(client_id)),
            keep_resident=dataset.touch_shard,
        )
        dataset.on_shard_evicted(pool.evict_client)
        client = pool["client_0"]
        pool.release_client("client_0")
        self.assertIs(pool["client_0"], client)
        client.client_data["train"]["x"].mul_(2.0)
        with self.assertRaises(RuntimeError) as refused:
            pool["client_0"]
        self.assertIn("edited in place", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
