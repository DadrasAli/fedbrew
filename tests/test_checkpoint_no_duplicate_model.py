"""A checkpoint must not store the global model twice.

_build_checkpoint_payload writes server_payload["model_state"] at the top
level and then calls server.save_state() for the optimizer-specific extras.
Every strategy's save_state() begins with super().save_state(), which
independently re-clones the same self._model_state under its own key, so both
copies -- numerically identical, separate storages -- were pickled into the
same file. On a real femnist_resnet18 FedLALR checkpoint that was 11.24 MB of
a 22.5 MB payload. checkpointing.save_last defaults to true and every shipped
training config sets it, so latest.pt was rewritten with the duplicate every
round of every run, for every strategy in fedbrew/servers/.

Nothing read both: _restore_server_state takes one or the other, and
eval_medmcqa_choice reads the top-level key. The risk in removing it is that
FedAvgServer.load_state skips the model restore when model_state is absent
instead of complaining, so a resume that lost both copies would silently start
over from the initial weights and report success. These tests pin the saving,
the round trip, the old on-disk format, and that failure mode.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch

from fedbrew.core import runner
from fedbrew.core.loop import _build_checkpoint_payload, _restore_server_state
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.fedopt import FedOptServer
from fedbrew.servers.scaffold import ScaffoldServer

MODEL_STATE = {"w": torch.arange(8, dtype=torch.float32)}


def _server(kind: str) -> FedAvgServer:
    if kind == "scaffold":
        server: FedAvgServer = ScaffoldServer(participation_rate=1.0, seed=0)
        server._server_control = {"w": torch.ones(8)}
    elif kind == "fedopt":
        server = FedOptServer(
            participation_rate=1.0,
            seed=0,
            server_optimizer="fedadam",
            server_learning_rate=0.01,
            beta1=0.9,
            beta2=0.99,
            tau=1e-3,
        )
        server._m = {"w": torch.full((8,), 0.5)}
        server._v = {"w": torch.full((8,), 0.25)}
    else:
        server = FedAvgServer(participation_rate=1.0, seed=0)
    server._model_state = {key: value.clone() for key, value in MODEL_STATE.items()}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


def _payload(server: FedAvgServer) -> dict[str, Any]:
    built = _build_checkpoint_payload(
        server,
        {},
        {
            "model_state": {key: value.clone() for key, value in MODEL_STATE.items()},
            "model_state_scope": "full",
            "model_state_metadata": {"model_state_scope": "full"},
        },
        {"fit_loss": 1.0},
        3,
    )
    assert built is not None
    return built


@pytest.mark.fast
class PayloadTest(unittest.TestCase):
    def test_the_model_is_stored_once(self) -> None:
        for kind in ("fedavg", "fedopt", "scaffold"):
            with self.subTest(kind=kind):
                payload = _payload(_server(kind))
                self.assertIn("model_state", payload)
                self.assertNotIn("model_state", payload["server_state"])
                self.assertNotIn("model_state_scope", payload["server_state"])
                self.assertNotIn("model_state_metadata", payload["server_state"])

    def test_the_top_level_copy_is_the_model(self) -> None:
        payload = _payload(_server("fedavg"))
        self.assertTrue(torch.equal(payload["model_state"]["w"], MODEL_STATE["w"]))
        self.assertEqual(payload["model_state_scope"], "full")
        self.assertEqual(payload["model_state_metadata"], {"model_state_scope": "full"})

    def test_strategy_state_is_untouched(self) -> None:
        """Only the three duplicated keys go; the extras are why save_state
        is called at all."""

        scaffold = _payload(_server("scaffold"))["server_state"]
        self.assertTrue(torch.equal(scaffold["server_control"]["w"], torch.ones(8)))
        self.assertIn("round_metrics", scaffold)
        self.assertIn("aggregation_weighting", scaffold)

        fedopt = _payload(_server("fedopt"))["server_state"]
        self.assertTrue(torch.equal(fedopt["m"]["w"], torch.full((8,), 0.5)))
        self.assertTrue(torch.equal(fedopt["v"]["w"], torch.full((8,), 0.25)))
        self.assertIn("server_optimizer", fedopt)

    def test_save_state_itself_still_returns_a_complete_snapshot(self) -> None:
        """Three other callers use it; the strip belongs to the writer alone."""

        state = _server("scaffold").save_state()
        self.assertIn("model_state", state)
        self.assertIn("model_state_scope", state)
        self.assertIn("model_state_metadata", state)


@pytest.mark.fast
class RestoreTest(unittest.TestCase):
    def test_a_checkpoint_written_today_round_trips(self) -> None:
        for kind in ("fedavg", "fedopt", "scaffold"):
            with self.subTest(kind=kind):
                payload = _payload(_server(kind))
                restored = _server(kind)
                restored._model_state = {"w": torch.zeros(8)}
                _restore_server_state(restored, payload)
                self.assertTrue(torch.equal(restored._model_state["w"], MODEL_STATE["w"]))
                self.assertEqual(restored._model_state_scope, "full")

    def test_a_checkpoint_in_the_old_format_still_restores(self) -> None:
        """A checkpoint written before this change carries the nested copy."""

        payload = _payload(_server("scaffold"))
        payload["server_state"]["model_state"] = {"w": torch.full((8,), 9.0)}
        payload["server_state"]["model_state_scope"] = "full"
        payload["server_state"]["model_state_metadata"] = {"model_state_scope": "full"}

        restored = _server("scaffold")
        restored._model_state = {"w": torch.zeros(8)}
        _restore_server_state(restored, payload)
        # The nested copy wins, so an old checkpoint restores exactly as it
        # always did -- the two were byte-identical in every real one.
        self.assertTrue(torch.equal(restored._model_state["w"], torch.full((8,), 9.0)))

    def test_a_checkpoint_with_no_model_is_refused(self) -> None:
        """load_state skips the restore instead of raising, so without this
        the run would resume from the initial weights and report success."""

        payload = _payload(_server("fedavg"))
        payload.pop("model_state")
        payload.pop("model_state_scope", None)
        payload.pop("model_state_metadata", None)
        with self.assertRaises(ValueError) as caught:
            _restore_server_state(_server("fedavg"), payload)
        self.assertIn("model_state", str(caught.exception))

    def test_a_non_mapping_server_state_is_still_refused(self) -> None:
        with self.assertRaises(ValueError):
            _restore_server_state(_server("fedavg"), {"server_state": [1, 2]})


def _write_config(directory: Path, hidden: int = 512) -> Path:
    config_path = directory / "ckpt.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            experiment:
              seed: 42
              output_dir: {directory / "run"}
            server:
              strategy: fedavg
              participation_rate: 1
              metrics: [fit_loss]
            client:
              update_rule: local_sgd
              batch_size: 4
              learning_rate: 0.05
              learning_rate_schedule: constant
              min_learning_rate: 0.0
              momentum: 0.0
              weight_decay: 0.0
              nesterov: false
              metrics: [fit_loss]
            data:
              num_clients: 2
              samples_per_client: 8
              input_dim: 64
              num_classes: 10
            model:
              name: mlp
              input_dim: 64
              hidden_dim: {hidden}
              num_classes: 10
            runtime:
              deterministic: true
              device: cpu
              use_amp: false
            evaluation:
              train:
                every: 1
                clients: all
            defaults:
              global_rounds: 1
              local_iterations: 1
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


class OnDiskTest(unittest.TestCase):
    """The file a real run writes, not a hand-built payload."""

    def test_the_written_checkpoint_holds_one_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.run(_write_config(root), runner.parse_args(["--quiet"]))
            written = sorted((root / "run" / "checkpoints").glob("*.pt"))
            self.assertTrue(written)
            checkpoint = torch.load(written[0], map_location="cpu", weights_only=False)
            size = written[0].stat().st_size

        self.assertIn("model_state", checkpoint)
        self.assertNotIn("model_state", checkpoint["server_state"])
        model_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in checkpoint["model_state"].values()
        )
        # One copy plus the small extras, not two. A second copy would put the
        # file above twice the model.
        self.assertLess(size, 2 * model_bytes)


if __name__ == "__main__":
    unittest.main()
