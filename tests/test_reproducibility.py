"""One seed has to reproduce one run, through the whole stack.

Nothing asserted this. tests/test_runtime_setup.py only checks that
deterministic mode sets CUBLAS_WORKSPACE_CONFIG; seed_everything was never
asserted to seed anything, no test ran the loop twice at one seed, and the one
resume test that existed compared *server state* -- the half that was already
being preserved -- so it passed while the run diverged.

These run the real runner on a real config, so they cover the whole path:
config -> factory -> task -> clients -> loop -> artifacts. The model carries
dropout, because without a stochastic layer a run is reproducible for the
trivial reason that nothing is random after initialisation, which is why the
shipped smoke config passing proved nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.core.runner import run

#: Enough rounds for a divergence to show, few enough to stay a unit test.
ROUNDS = 4

#: A stochastic layer in the training path. Everything below is deterministic
#: without it, so a reproducibility test on a dropout-free model asserts
#: nothing about the RNG.
DROPOUT = 0.3


def _config(
    output_dir: Path,
    *,
    rounds: int = ROUNDS,
    val_clients: str = "all",
    checkpoint: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "experiment": {"seed": 42, "output_dir": str(output_dir)},
        "server": {
            "strategy": "fedavg",
            "participation_rate": 1,
            "metrics": ["fit_loss", "fit_accuracy"],
        },
        "client": {
            "update_rule": "local_sgd",
            "batch_size": 4,
            "learning_rate": 0.05,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "learning_rate_schedule": "constant",
            "min_learning_rate": 0.0,
            "metrics": ["fit_loss", "fit_accuracy"],
        },
        "data": {
            "num_clients": 4,
            "samples_per_client": 16,
            "input_dim": 4,
            "num_classes": 2,
        },
        "model": {
            "name": "mlp",
            "input_dim": 4,
            "hidden_dim": 8,
            "num_classes": 2,
            "dropout": DROPOUT,
        },
        "runtime": {
            "deterministic": True,
            "deterministic_warn_only": False,
            "device": "cpu",
            "use_amp": False,
        },
        # Every block every round. `every` pins round 1 and the *final* round,
        # so a 2-round run would evaluate central_test at round 2 while a
        # 4-round run would not, and the resume comparison below would be
        # measuring that rather than the RNG.
        "evaluation": {
            "train": {"every": 1, "clients": "all"},
            "val": {"every": 1, "clients": val_clients},
            "test": {"every": 1, "clients": "all"},
            "central_test": {"every": 1},
        },
        "defaults": {"global_rounds": rounds, "local_iterations": 1},
    }
    if checkpoint:
        config["runtime"]["checkpointing"] = {
            "enabled": True,
            "save_last": True,
            "save_every_round": True,
            "keep_last": 0,
        }
    return config


def _write(root: Path, name: str, **kwargs: Any) -> Path:
    output_dir = root / name
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(_config(output_dir, **kwargs)), encoding="utf-8")
    return path


def _run(path: Path, *, resume_latest: bool = False) -> Path:
    args = argparse.Namespace(resume_latest=resume_latest) if resume_latest else None
    state = run(path, args=args)
    assert state.metrics_history, "the run produced no rounds"
    return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])


def _rows(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "round_metrics.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


#: Columns that legitimately differ between two identical runs.
TIMING = {
    "duration_sec",
    "fit_sec",
    "aggregate_sec",
    "client_eval_sec",
    "global_eval_sec",
    "checkpoint_sec",
}


def _digest(output_dir: Path, columns: set[str] | None = None) -> str:
    digest = hashlib.sha256()
    for row in _rows(output_dir):
        for key in sorted(row):
            if key in TIMING or (columns is not None and key not in columns):
                continue
            digest.update(f"{key}={row[key]};".encode())
        digest.update(b"|")
    return digest.hexdigest()


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class SameSeedTests(_TempRoot):
    def test_two_runs_at_one_seed_are_identical(self) -> None:
        first = _run(_write(self.root, "a"))
        second = _run(_write(self.root, "b"))
        self.assertEqual(_digest(first), _digest(second))
        self.assertEqual(len(_rows(first)), ROUNDS)

    @pytest.mark.fast
    def test_the_fixture_actually_exercises_the_rng(self) -> None:
        # Guard on the guard: if the model were deterministic in the forward
        # pass, every test in this module would pass for the wrong reason.
        import torch

        from fedbrew.models.torch_mlp import build_torch_mlp

        model = build_torch_mlp(
            {"input_dim": 4, "hidden_dim": 8, "num_classes": 2, "dropout": DROPOUT}
        )
        model.train()
        features = torch.ones(64, 4)
        torch.manual_seed(0)
        before = torch.get_rng_state().clone()
        first = model(features)
        self.assertFalse(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.equal(first, model(features)))


class ResumeTests(_TempRoot):
    def test_a_resumed_run_matches_the_uninterrupted_one(self) -> None:
        full = _run(_write(self.root, "full", rounds=ROUNDS, checkpoint=True))
        partial = _write(self.root, "part", rounds=ROUNDS // 2, checkpoint=True)
        _run(partial)
        resumed_config = _write(self.root, "part", rounds=ROUNDS, checkpoint=True)
        resumed = _run(resumed_config, resume_latest=True)
        self.assertEqual(len(_rows(resumed)), ROUNDS)
        self.assertEqual(_digest(full), _digest(resumed))


class EvaluationScopeTests(_TempRoot):
    """Evaluation must not move the training trajectory.

    The fit metrics are produced before any evaluation runs in the same round,
    so changing how many clients are evaluated must leave them untouched. It
    did not: the personalized arm trained through the global RNG, and the
    causal-LM task rebuilt its model per evaluated client.
    """

    FIT_COLUMNS = {"round_id", "num_clients", "num_examples", "fit_loss", "fit_accuracy"}

    def test_evaluating_more_clients_does_not_change_training(self) -> None:
        few = _run(_write(self.root, "few", val_clients="sample:1"))
        many = _run(_write(self.root, "many", val_clients="sample:3"))
        self.assertEqual(_digest(few, self.FIT_COLUMNS), _digest(many, self.FIT_COLUMNS))

    def test_the_two_runs_really_did_evaluate_different_client_counts(self) -> None:
        # Otherwise the comparison above is between two identical runs.
        few = _run(_write(self.root, "few2", val_clients="sample:1"))
        many = _run(_write(self.root, "many2", val_clients="sample:3"))
        self.assertNotEqual(_digest(few), _digest(many))


if __name__ == "__main__":
    unittest.main()
