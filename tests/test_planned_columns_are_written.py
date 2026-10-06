"""The plan header's column list is what round_metrics.csv actually carries.

`_planned_metric_names` says of itself: "A column listed here that the run does
not write, or written and not listed, would make the header worse than no
header." It once promised six columns no FedAvg-family run wrote.

The mechanism was one filter applied twice with different lists: a client
exempted its own extras from `client.metrics`, and the server ran the whole
aggregated dict through `server.metrics`, which un-exempted them, so fourteen
shipped arms asked for `optimizer_steps` and got no column. The two lists are
one now, `reporting.fit_metrics`, applied once by the server after every
client and server metric is added (docs/04 section 9), so a column is kept
exactly when the list names it or the list is empty.

The check here is the round trip: run, then diff the header against the CSV in
both directions. Nothing in it is hand-listed, so it holds for columns nobody
has thought of yet. It runs every registered rule, `RULES` has to name each
one, and the shipped arms that ask for the upload volume are checked to plan
it.
"""

from __future__ import annotations

import contextlib
import csv
import io
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.core import runner
from fedbrew.core.config import load_config
from fedbrew.core.logging import _planned_metric_names
from fedbrew.core.metrics import CLIENT_UNFILTERED_FIT_METRICS
from fedbrew.core.registry import client_updates, register_builtin_components

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "configs" / "dev" / "synthetic.yaml"

#: Columns round_metrics.csv always carries and no metrics list governs: the
#: round's identity and the timing block.
BOOKKEEPING = frozenset(
    {
        "round_id",
        "num_clients",
        "num_examples",
        "duration_sec",
        "fit_sec",
        "aggregate_sec",
        "client_eval_sec",
        "global_eval_sec",
        "checkpoint_sec",
    }
)


def _write(root: Path, name: str, fit_metrics: list[str]) -> Path:
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(root / name)
    raw["client"]["update_rule"] = "fedavg"
    raw["client"]["update_mode"] = "sequential_epoch"
    raw["client"]["frozen_gradient_weighting"] = "examples"
    raw.setdefault("reporting", {})["fit_metrics"] = list(fit_metrics)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


#: The per-round upload volume every rule reports.
VOLUME = ["communicated_parameters", "communicated_bytes"]

_SGD_KEYS = {
    "batch_size": 32,
    "learning_rate": 0.01,
    "learning_rate_schedule": "constant",
    "min_learning_rate": 0.0,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
}
_ENGINE_KEYS = {"update_mode": "sequential_epoch", "frozen_gradient_weighting": "examples"}

#: A minimal client block for every registered update rule, with the strategy it
#: pairs with. Each carries only keys the rule reads: a key a rule ignores is
#: refused at load, so a shared superset would not load for most of them.
RULES: dict[str, dict[str, Any]] = {
    "local_sgd": {"strategy": "fedavg", "client": dict(_SGD_KEYS)},
    "fedavg": {"strategy": "fedavg", "client": {**_SGD_KEYS, **_ENGINE_KEYS}},
    "centralized": {"strategy": "centralized", "client": {**_SGD_KEYS, **_ENGINE_KEYS}},
    # Refused under model_scope: global, where it would train and evaluate
    # exactly like plain fedavg.
    "fedavg_ft": {
        "strategy": "fedavg",
        "client": {**_SGD_KEYS, **_ENGINE_KEYS, "finetune_epochs": 1},
        "evaluation": {"model_scope": "both"},
    },
    "local_adamw": {
        "strategy": "fedavg",
        "client": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "learning_rate_schedule": "constant",
            "min_learning_rate": 0.0,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
        },
    },
    "fedprox": {
        "strategy": "fedavg",
        "client": {"batch_size": 32, "learning_rate": 0.01, "proximal_mu": 0.01},
    },
    "scaffold": {"strategy": "scaffold", "client": {"batch_size": 32, "learning_rate": 0.01}},
    "delta_sgd": {
        "strategy": "fedavg",
        "client": {"batch_size": 32, "eta_0": 0.05, **_ENGINE_KEYS},
    },
    "fedlalr": {"strategy": "fedlalr", "client": {"batch_size": 32, "learning_rate": 0.003}},
    "fedlada": {
        "strategy": "fedlada",
        "server": {"server_learning_rate": 1.0},
        "client": {
            "batch_size": 32,
            "learning_rate": 0.003,
            "beta1": 0.9,
            "beta2": 0.99,
            "epsilon": 1.0e-8,
            "lada_alpha": 0.1,
        },
    },
    "fafed": {
        "strategy": "fafed",
        "client": {
            "batch_size": 32,
            "learning_rate": 0.003,
            "beta2": 0.99,
            "fafed_alpha": 0.1,
            "fafed_rho": 0.01,
        },
    },
}


def _write_rule(root: Path, name: str, rule: str, *, fit_metrics: list[str]) -> Path:
    """One round of `rule` on the synthetic base, with the list as given."""

    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(root / name)
    raw["server"] = {
        "strategy": RULES[rule]["strategy"],
        "participation_rate": 1,
        **RULES[rule].get("server", {}),
    }
    raw["client"] = {"update_rule": rule, **RULES[rule]["client"]}
    raw.setdefault("reporting", {})["fit_metrics"] = list(fit_metrics)
    if "evaluation" in RULES[rule]:
        raw.setdefault("evaluation", {}).update(RULES[rule]["evaluation"])
    raw["schedule"]["rounds"] = 1
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _run_and_read_header(path: Path) -> set[str]:
    runner.run(path, runner.parse_args(["--quiet"]))
    config = load_config(str(path))
    csv_path = Path(config.experiment.output_dir) / "round_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return set(next(csv.reader(handle)))


class TheHeaderMatchesTheFileTest(unittest.TestCase):
    EXTRAS = ["optimizer_steps", "client_learning_rate"]

    def _check(self, path: Path) -> None:
        planned = set(_planned_metric_names(load_config(str(path))))
        written = _run_and_read_header(path) - BOOKKEEPING
        self.assertEqual(
            planned - written, set(), "the header promises columns the run does not write"
        )
        self.assertEqual(
            written - planned, set(), "the run writes columns the header does not list"
        )

    def test_when_the_list_does_not_name_the_extras(self) -> None:
        """The shipped FedAvg-family shape: the extras are not kept."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "dropped", ["fit_loss", "fit_accuracy"])
            self._check(path)
            written = _run_and_read_header(path)
            for name in self.EXTRAS:
                self.assertNotIn(name, written)

    def test_when_the_list_names_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "kept", ["fit_loss", "fit_accuracy", *self.EXTRAS])
            self._check(path)
            written = _run_and_read_header(path)
            for name in self.EXTRAS:
                self.assertIn(name, written)

    def test_when_the_list_is_empty_and_keeps_everything(self) -> None:
        """An empty list keeps everything, so every extra the rule emits lands."""

        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory), "unfiltered", [])
            self._check(path)
            written = _run_and_read_header(path)
            for name in CLIENT_UNFILTERED_FIT_METRICS["fedavg"]:
                self.assertIn(name, written)


class EveryRuleRoundTripsTest(unittest.TestCase):
    """Header and file agree for every registered rule, and one list decides for all."""

    KEPT = ["fit_loss", "fit_accuracy"]

    def _round_trip(self, path: Path) -> set[str]:
        config = load_config(str(path))
        planned = set(_planned_metric_names(config))
        written = _run_and_read_header(path) - BOOKKEEPING
        self.assertEqual(
            planned - written, set(), "the header promises columns the run does not write"
        )
        self.assertEqual(
            written - planned, set(), "the run writes columns the header does not list"
        )
        return written

    @pytest.mark.fast
    def test_every_registered_rule_has_an_entry(self) -> None:
        """A rule added tomorrow and left out of RULES fails here, not in the field."""

        register_builtin_components()
        self.assertEqual(sorted(client_updates.list()), sorted(RULES))

    def test_a_list_that_does_not_name_the_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for rule in sorted(RULES):
                with self.subTest(rule=rule):
                    path = _write_rule(
                        Path(directory), f"{rule}-without", rule, fit_metrics=self.KEPT
                    )
                    self.assertTrue(set(VOLUME).isdisjoint(self._round_trip(path)))

    def test_a_list_that_names_the_volume(self) -> None:
        """Enough for every rule: nothing filters it before the server's list.

        Under the two lists, the rules that filtered their extras on the client
        needed the names in both, and the FedAvg family in the server's only.
        """

        with tempfile.TemporaryDirectory() as directory:
            for rule in sorted(RULES):
                with self.subTest(rule=rule):
                    path = _write_rule(
                        Path(directory), f"{rule}-with", rule, fit_metrics=self.KEPT + VOLUME
                    )
                    self.assertTrue(set(VOLUME) <= self._round_trip(path))


class ADiagnosticBuiltFromAClientMetricTest(unittest.TestCase):
    """FedLALR's spread across clients is built from each client's coordinate mean.

    Under the two lists a client list without the mean dropped the spread; the
    clients have no list now, so the spread is kept or dropped by the one list
    like any other column, whether or not it keeps the mean itself.
    """

    SPREAD = {f"effective_learning_rate_across_clients_{s}" for s in ("mean", "std", "min", "max")}
    SOURCE = "effective_learning_rate_coordinate_mean"

    def test_the_spread_is_written_when_the_list_names_it_and_not_its_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rule(
                Path(directory),
                "fedlalr-spread",
                "fedlalr",
                fit_metrics=["fit_loss", *sorted(self.SPREAD)],
            )
            planned = set(_planned_metric_names(load_config(str(path))))
            written = _run_and_read_header(path) - BOOKKEEPING
            self.assertEqual(planned, written)
            self.assertTrue(self.SPREAD <= written)
            self.assertNotIn(self.SOURCE, written)

    def test_the_spread_is_dropped_when_the_list_does_not_name_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rule(
                Path(directory), "fedlalr-nospread", "fedlalr", fit_metrics=["fit_loss"]
            )
            planned = set(_planned_metric_names(load_config(str(path))))
            written = _run_and_read_header(path) - BOOKKEEPING
            self.assertEqual(planned, written)
            self.assertTrue(self.SPREAD.isdisjoint(written))


class TheShippedArmsPlanTheVolumeTheyAskForTest(unittest.TestCase):
    """Preflight tells a reader to compare arms on communicated_bytes.

    So an arm whose list names the volume has to plan it. Scoped to every
    config in the tree, so a new arm is covered without an edit here.
    """

    def test_every_shipped_config_that_names_the_volume_plans_it(self) -> None:
        from tests.shipped_resolved_configs import REPO, no_generated_data, shipped_run_configs

        checked = 0
        for path in shipped_run_configs():
            with no_generated_data():
                config = load_config(REPO / path)
            if not set(VOLUME) & set(config.reporting.fit_metrics):
                continue
            checked += 1
            with self.subTest(config=str(path)):
                self.assertTrue(set(VOLUME) <= set(_planned_metric_names(config)))
        # fedprox, scaffold, delta_sgd and fedlalr on FEMNIST, scaffold on MNIST.
        self.assertEqual(checked, 5)


class PrintEveryWritesEveryRoundTest(unittest.TestCase):
    """--print-every N changes what the terminal says, not what the run writes.

    The same round trip as above, from the other side: the reporter prints
    rounds 1, N, 2N, ... and the final round, and round_metrics.csv still holds
    every round, with each split evaluated on its own schedule rather than N's.
    tests/test_run_reporting.py pins which rounds print; this needs a run.
    """

    def test_every_round_is_written_and_only_the_scheduled_ones_print(self) -> None:
        raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        raw["schedule"]["rounds"] = 7
        raw["runtime"]["checkpointing"]["enabled"] = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw["experiment"]["output_dir"] = str(root / "run")
            path = root / "print-every.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                runner.run(path, runner.parse_args(["--no-rich", "--print-every", "3"]))
            with (root / "run" / "round_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        printed = [int(n) for n in re.findall(r"\bround (\d+)/7\b", stdout.getvalue())]
        self.assertEqual(printed, [1, 3, 6, 7])
        self.assertEqual([int(row["round_id"]) for row in rows], list(range(1, 8)))
        # The test split at the default every: 10 over seven rounds: 1 and 7.
        evaluated = [int(row["round_id"]) for row in rows if row["test_loss_avg"]]
        self.assertEqual(evaluated, [1, 7])


if __name__ == "__main__":
    unittest.main()
