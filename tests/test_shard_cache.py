"""The client-shard cache must bound memory without changing what is served.

Cross-device rounds touch every client each round; on a cluster the shards sit
on a shared filesystem, so re-reading them per round is what leaves the GPU
idle. These tests pin that caching is transparent and stays inside its budget.

Transparent has two halves, because only one of them can be enforced. A served
payload's mappings are the caller's own, so rebinding a key cannot reach the
cache; its tensors are the cached ones, so an in-place edit can, and is refused
on the next serve instead. The pooled centralized view memoises the same way and
is held to the same two.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from fedbrew.data.centralized_dataset import CentralizedFederatedDataset
from fedbrew.data.manifest_dataset import ManifestFederatedDataset, _shard_nbytes


def _write_dataset(root: Path, num_clients: int = 4, per_client: int = 8) -> Path:
    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    clients = []
    for index in range(num_clients):
        client_id = f"client_{index}"
        payload = {
            "train": {
                "x": torch.full((per_client, 1, 4, 4), index, dtype=torch.uint8),
                "y": torch.full((per_client,), index, dtype=torch.long),
            },
            "test": {
                "x": torch.full((2, 1, 4, 4), index, dtype=torch.uint8),
                "y": torch.full((2,), index, dtype=torch.long),
            },
        }
        torch.save(payload, shards / f"{client_id}.pt")
        clients.append(
            {
                "client_id": client_id,
                "shard": f"shards/{client_id}.pt",
                "num_examples": per_client + 2,
                "num_train_examples": per_client,
                "num_test_examples": 2,
            }
        )

    (root / "clients.jsonl").write_text(
        "\n".join(json.dumps(client) for client in clients), encoding="utf-8"
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"clients_file": "clients.jsonl", "shards_dir": "shards"}),
        encoding="utf-8",
    )
    return manifest


class ShardCacheTests(unittest.TestCase):
    """Caching must be invisible to callers apart from repeat-read speed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest = _write_dataset(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_cached_and_uncached_reads_agree(self) -> None:
        cached = ManifestFederatedDataset(self.manifest)
        uncached = ManifestFederatedDataset(self.manifest, shard_cache_bytes=0)
        for client_id in uncached.list_clients():
            left = cached.get_client_data(client_id)
            right = uncached.get_client_data(client_id)
            self.assertTrue(torch.equal(left["train"]["x"], right["train"]["x"]))
            self.assertTrue(torch.equal(left["train"]["y"], right["train"]["y"]))
            self.assertEqual(left["num_examples"], right["num_examples"])

    def test_repeat_access_serves_the_same_tensors(self) -> None:
        dataset = ManifestFederatedDataset(self.manifest)
        first = dataset.get_client_data("client_0")
        second = dataset.get_client_data("client_0")
        self.assertIs(first["train"]["x"], second["train"]["x"])
        # ... in structures that are not the same, so the caller owns the keys.
        self.assertIsNot(first["train"], second["train"])

    def test_disabled_cache_rereads_from_disk(self) -> None:
        dataset = ManifestFederatedDataset(self.manifest, shard_cache_bytes=0)
        first = dataset.get_client_data("client_0")
        second = dataset.get_client_data("client_0")
        self.assertIsNot(first["train"]["x"], second["train"]["x"])
        self.assertTrue(torch.equal(first["train"]["x"], second["train"]["x"]))

    def test_cache_evicts_to_stay_within_budget(self) -> None:
        probe = ManifestFederatedDataset(self.manifest, shard_cache_bytes=0)
        one_shard = _shard_nbytes(probe.get_client_data("client_0"))
        # Budget for roughly two shards, then touch all four.
        dataset = ManifestFederatedDataset(self.manifest, shard_cache_bytes=one_shard * 2)
        for client_id in dataset.list_clients():
            dataset.get_client_data(client_id)
        self.assertLessEqual(dataset._shard_cache_bytes_used, dataset.shard_cache_bytes)
        self.assertLess(len(dataset._shard_cache), len(dataset.list_clients()))

    def test_oversized_shard_is_served_without_caching(self) -> None:
        dataset = ManifestFederatedDataset(self.manifest, shard_cache_bytes=1)
        data = dataset.get_client_data("client_0")
        self.assertEqual(len(dataset._shard_cache), 0)
        self.assertEqual(int(data["train"]["y"][0]), 0)

    def test_rebinding_a_served_key_does_not_reach_the_cache(self) -> None:
        dataset = ManifestFederatedDataset(self.manifest)
        served = dataset.get_client_data("client_0")
        served["train"]["x"] = torch.full_like(served["train"]["x"], 9)
        served["train"] = {}
        served["num_train_examples"] = -1

        again = dataset.get_client_data("client_0")
        self.assertEqual(int(again["train"]["x"][0, 0, 0, 0]), 0)
        self.assertEqual(again["num_train_examples"], 8)

    def test_editing_a_served_shard_in_place_is_refused(self) -> None:
        dataset = ManifestFederatedDataset(self.manifest)
        served = dataset.get_client_data("client_0")
        # What no consumer does today, and what the cache would carry into every
        # later round if one started: an in-place rescale of the served shard.
        served["train"]["x"].add_(1)

        with self.assertRaises(RuntimeError) as raised:
            dataset.get_client_data("client_0")
        message = str(raised.exception)
        self.assertIn("client_0", message)
        self.assertIn("train.x", message)

    def test_an_uncached_shard_is_nobody_else_s_to_poison(self) -> None:
        one_shard = _shard_nbytes(
            ManifestFederatedDataset(self.manifest, shard_cache_bytes=0).get_client_data("client_0")
        )
        for label, budget in (("disabled", 0), ("shard over budget", one_shard - 1)):
            with self.subTest(label):
                dataset = ManifestFederatedDataset(self.manifest, shard_cache_bytes=budget)
                dataset.get_client_data("client_0")["train"]["x"].add_(1)
                self.assertEqual(len(dataset._shard_cache), 0)
                served = dataset.get_client_data("client_0")
                self.assertEqual(int(served["train"]["x"][0, 0, 0, 0]), 0)

    def test_torch_still_counts_in_place_edits_where_the_check_reads_them(self) -> None:
        # CachedPayload detects an in-place edit from Tensor._version, which is
        # private. A torch that stopped bumping it would leave the check passing
        # on a poisoned shard, so the assumption is pinned rather than assumed.
        tensor = torch.zeros(4)
        before = tensor._version
        view = tensor[:2]
        view.add_(1)
        self.assertGreater(tensor._version, before)

    def test_least_recently_used_client_is_evicted_first(self) -> None:
        probe = ManifestFederatedDataset(self.manifest, shard_cache_bytes=0)
        one_shard = _shard_nbytes(probe.get_client_data("client_0"))
        dataset = ManifestFederatedDataset(self.manifest, shard_cache_bytes=one_shard * 2)
        dataset.get_client_data("client_0")
        dataset.get_client_data("client_1")
        dataset.get_client_data("client_0")  # refresh client_0
        dataset.get_client_data("client_2")  # should evict client_1
        self.assertIn("client_0", dataset._shard_cache)
        self.assertNotIn("client_1", dataset._shard_cache)


class PooledPayloadTests(unittest.TestCase):
    """The centralized view memoises one payload and serves it every round."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest = _write_dataset(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _pooled(self) -> CentralizedFederatedDataset:
        return CentralizedFederatedDataset(
            ManifestFederatedDataset(self.manifest, shard_cache_bytes=0)
        )

    def test_rebinding_a_served_key_does_not_reach_the_pool(self) -> None:
        dataset = self._pooled()
        served = dataset.get_client_data("centralized")
        served["train"] = {}
        self.assertEqual(dataset.get_client_data("centralized")["num_train_examples"], 32)

    def test_editing_the_served_pool_in_place_is_refused(self) -> None:
        dataset = self._pooled()
        dataset.get_client_data("centralized")["train"]["x"].add_(1)

        with self.assertRaises(RuntimeError) as raised:
            dataset.get_client_data("centralized")
        self.assertIn("centralized", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
