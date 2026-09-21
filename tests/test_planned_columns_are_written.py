"""The plan header's column list is what round_metrics.csv actually carries.

`_planned_metric_names` says of itself: "A column listed here that the run does
not write, or written and not listed, would make the header worse than no
header." It was promising six columns no FedAvg-family run writes.

The mechanism is one filter applied twice with different lists. A client
exempts its own extras -- `optimizer_steps`, `communicated_bytes`,
`client_learning_rate` and two more -- from `client.metrics`, adding them after
`filter_metrics` so a metrics list cannot remove them. The server then runs the
whole aggregated dict through `filter_metrics` again, against `server.metrics`,
which un-exempts them. `configs/femnist/fedavg.yaml` lists
`optimizer_steps` and `client_learning_rate` under `client.metrics` and neither
under `server.metrics`; fourteen shipped configs are in that shape, which is
every FedAvg-family arm in the tree.

The existing guard on the header, `test_the_verbose_column_list_is_what_the_run
_will_write`, checks that particular names *are* in the planned list. It cannot
catch a name that is in the list and not in the file, which is this defect. So
the check here is the round trip: run, then diff the header against the CSV in
both directions. Nothing in it is hand-listed, so it holds for columns nobody
has thought of yet.

The round trip used to run one rule, `fedavg`, and the header's "not written"
row consulted only the extras that rule family exempts. The four rules that
filter everything -- fedprox, scaffold, delta_sgd, fedlalr -- were covered by
neither, and five shipped configs of theirs listed `communicated_bytes` under
`client.metrics` while `server.metrics` dropped it: no column, and no row saying
so, under a preflight notice telling the reader to compare arms on exactly that
column. Reading a run's CSV found it. So the round trip now runs every
registered rule, `RULES` has to name each one, and the shipped arms that ask
for the volume are checked to write it.
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
from fedbrew.core.logging import (
    _client_metrics_the_server_filter_removes,
    _planned_metric_names,
)
from fedbrew.core.metrics import (
    CLIENT_UNFILTERED_FIT_METRICS,
    client_fit_extras,
    dropped_client_fit_extras,
    surviving_client_fit_extras,
)
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


def _write(root: Path, name: str, **overrides: Any) -> Path:
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(root / name)
    raw["client"]["update_rule"] = "fedavg"
    raw["client"]["update_mode"] = "sequential_epoch"
    raw["client"]["frozen_gradient_weighting"] = "examples"
    for section, values in overrides.items():
        raw[section].update(values)
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
}


def _write_rule(
    root: Path, name: str, rule: str, *, server_metrics: list[str], client_metrics: list[str]
) -> Path:
    """One round of `rule` on the synthetic base, with the two lists as given."""

    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(root / name)
    raw["server"] = {
        "strategy": RULES[rule]["strategy"],
        "participation_rate": 1,
        "metrics": list(server_metrics),
    }
    raw["client"] = {"update_rule": rule, **RULES[rule]["client"], "metrics": list(client_metrics)}
    if "evaluation" in RULES[rule]:
        raw.setdefault("evaluation", {}).update(RULES[rule]["evaluation"])
    raw["defaults"]["global_rounds"] = 1
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

    def test_when_server_metrics_does_not_list_the_client_extras(self) -> None:
        """The shipped shape: asked for under client.metrics, dropped anyway."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write(
                root,
                "dropped",
                server={"metrics": ["fit_loss", "fit_accuracy"]},
                client={"metrics": ["fit_loss", "fit_accuracy", *self.EXTRAS]},
            )
            self._check(path)
            written = _run_and_read_header(path)
            for name in self.EXTRAS:
                self.assertNotIn(name, written)

    def test_when_server_metrics_does_list_them(self) -> None:
        """The same config with the extras added where they are read."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write(
                root,
                "kept",
                server={"metrics": ["fit_loss", "fit_accuracy", *self.EXTRAS]},
                client={"metrics": ["fit_loss", "fit_accuracy", *self.EXTRAS]},
            )
            self._check(path)
            written = _run_and_read_header(path)
            for name in self.EXTRAS:
                self.assertIn(name, written)

    def test_when_server_metrics_is_empty_and_keeps_everything(self) -> None:
        """An empty list keeps everything, so every extra the rule emits lands."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write(
                root,
                "unfiltered",
                server={"metrics": []},
                client={"metrics": ["fit_loss", "fit_accuracy"]},
            )
            self._check(path)
            written = _run_and_read_header(path)
            for name in client_fit_extras("fedavg"):
                self.assertIn(name, written)


@pytest.mark.fast
class TheSurvivingSetIsTheComplementTest(unittest.TestCase):
    def test_surviving_and_dropped_partition_the_extras(self) -> None:
        for rule in ("fedavg", "local_sgd", "scaffold"):
            for server_metrics in ([], ["fit_loss"], ["fit_loss", "optimizer_steps"]):
                with self.subTest(rule=rule, server_metrics=server_metrics):
                    surviving = set(surviving_client_fit_extras(rule, server_metrics))
                    dropped = set(dropped_client_fit_extras(rule, server_metrics))
                    self.assertEqual(surviving | dropped, set(client_fit_extras(rule)))
                    self.assertEqual(surviving & dropped, set())

    def test_an_empty_server_list_drops_nothing(self) -> None:
        self.assertEqual(dropped_client_fit_extras("fedavg", []), [])

    def test_a_rule_with_no_exempt_extras_still_loses_what_it_asks_for(self) -> None:
        """No exempt extras is not nothing to drop.

        This test used to read "the five rules that filter everything at the end
        have no row", and pinned it -- four rules, and the row was wrong to be
        silent: their client keeps what client.metrics names, and the server
        then drops what server.metrics does not.
        """

        self.assertEqual(client_fit_extras("scaffold"), frozenset())
        self.assertEqual(dropped_client_fit_extras("scaffold", ["fit_loss"]), [])

        config = load_config(str(REPO_ROOT / "configs" / "femnist" / "scaffold.yaml"))
        config.server.metrics = [name for name in config.server.metrics if name not in VOLUME]
        self.assertEqual(_client_metrics_the_server_filter_removes(config), sorted(VOLUME))


@pytest.mark.fast
class TheHeaderSaysWhatItWillNotWriteTest(unittest.TestCase):
    """Naming the gap is the other half: the config is not wrong, just unmet."""

    def test_the_line_names_only_what_was_asked_for_and_dropped(self) -> None:
        config = load_config(str(REPO_ROOT / "configs" / "femnist" / "fedavg.yaml"))
        self.assertEqual(
            _client_metrics_the_server_filter_removes(config),
            ["client_learning_rate", "optimizer_steps"],
        )

    def test_an_extra_the_config_never_asked_for_is_not_named(self) -> None:
        """`communicated_bytes` is dropped too, and no config asks for it.

        Reporting every dropped extra would put five names in front of every
        reader, four of which nobody wanted. The line exists to close the gap
        between what a config asks for and what it gets, so it says nothing
        when the config asked for nothing.
        """

        config = load_config(str(REPO_ROOT / "configs" / "femnist" / "fedavg.yaml"))
        dropped = dropped_client_fit_extras(config.client.update_rule, config.server.metrics)
        self.assertIn("communicated_bytes", dropped)
        self.assertNotIn("communicated_bytes", _client_metrics_the_server_filter_removes(config))


class EveryRuleRoundTripsTest(unittest.TestCase):
    """Header, row and file agree for every registered rule, in all three shapes."""

    KEPT = ["fit_loss", "fit_accuracy"]

    def _round_trip(self, path: Path) -> tuple[set[str], list[str]]:
        config = load_config(str(path))
        planned = set(_planned_metric_names(config))
        written = _run_and_read_header(path) - BOOKKEEPING
        self.assertEqual(
            planned - written, set(), "the header promises columns the run does not write"
        )
        self.assertEqual(
            written - planned, set(), "the run writes columns the header does not list"
        )
        return written, _client_metrics_the_server_filter_removes(config)

    @pytest.mark.fast
    def test_every_registered_rule_has_an_entry(self) -> None:
        """A rule added tomorrow and left out of RULES fails here, not in the field."""

        register_builtin_components()
        self.assertEqual(sorted(client_updates.list()), sorted(RULES))

    def test_asked_for_by_the_client_and_not_kept_by_the_server(self) -> None:
        """The shipped defect's shape: no column, and the row has to say so."""

        with tempfile.TemporaryDirectory() as directory:
            for rule in sorted(RULES):
                with self.subTest(rule=rule):
                    path = _write_rule(
                        Path(directory),
                        f"{rule}-asked",
                        rule,
                        server_metrics=self.KEPT,
                        client_metrics=self.KEPT + VOLUME,
                    )
                    written, not_written = self._round_trip(path)
                    self.assertTrue(set(VOLUME).isdisjoint(written))
                    self.assertEqual(not_written, sorted(VOLUME))

    def test_named_by_both_lists(self) -> None:
        """The fix: every rule writes them, and the row is empty."""

        with tempfile.TemporaryDirectory() as directory:
            for rule in sorted(RULES):
                with self.subTest(rule=rule):
                    path = _write_rule(
                        Path(directory),
                        f"{rule}-both",
                        rule,
                        server_metrics=self.KEPT + VOLUME,
                        client_metrics=self.KEPT + VOLUME,
                    )
                    written, not_written = self._round_trip(path)
                    self.assertTrue(set(VOLUME) <= written)
                    self.assertEqual(not_written, [])

    def test_named_by_the_server_only(self) -> None:
        """Enough for a rule that exempts its extras, not for the four that filter them.

        Which is why the five arms name the volume in both lists: moving the
        names from client.metrics to server.metrics would lose them again.
        """

        with tempfile.TemporaryDirectory() as directory:
            for rule in sorted(RULES):
                with self.subTest(rule=rule):
                    path = _write_rule(
                        Path(directory),
                        f"{rule}-server",
                        rule,
                        server_metrics=self.KEPT + VOLUME,
                        client_metrics=self.KEPT,
                    )
                    written, not_written = self._round_trip(path)
                    if rule in CLIENT_UNFILTERED_FIT_METRICS:
                        self.assertTrue(set(VOLUME) <= written)
                    else:
                        self.assertTrue(set(VOLUME).isdisjoint(written))
                    self.assertEqual(not_written, [])


class ADiagnosticBuiltFromAClientMetricTest(unittest.TestCase):
    """FedLALR's spread across clients exists only if its clients report the mean.

    The header used to promise the four columns unconditionally; a config whose
    client.metrics left out effective_learning_rate_coordinate_mean got none of
    them. The round trips above run that shape; this runs the other.
    """

    SPREAD = {f"effective_learning_rate_across_clients_{s}" for s in ("mean", "std", "min", "max")}

    def test_the_spread_is_promised_and_written_when_the_source_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rule(
                Path(directory),
                "fedlalr-spread",
                "fedlalr",
                server_metrics=["fit_loss"],
                client_metrics=["fit_loss", "effective_learning_rate_coordinate_mean"],
            )
            planned = set(_planned_metric_names(load_config(str(path))))
            written = _run_and_read_header(path) - BOOKKEEPING
            self.assertEqual(planned, written)
            self.assertTrue(self.SPREAD <= written)

    def test_the_spread_is_not_promised_when_the_source_is_filtered_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rule(
                Path(directory),
                "fedlalr-nospread",
                "fedlalr",
                server_metrics=["fit_loss"],
                client_metrics=["fit_loss"],
            )
            planned = set(_planned_metric_names(load_config(str(path))))
            written = _run_and_read_header(path) - BOOKKEEPING
            self.assertEqual(planned, written)
            self.assertTrue(self.SPREAD.isdisjoint(written))


class TheShippedArmsWriteTheVolumeTheyAskForTest(unittest.TestCase):
    """Preflight tells a reader to compare arms on communicated_bytes.

    So an arm that asks for the volume under client.metrics has to write it.
    Scoped to every config in the tree, so a new arm is covered without an edit
    here.
    """

    def test_no_shipped_config_asks_for_the_volume_and_drops_it(self) -> None:
        checked = 0
        for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
            try:
                config = load_config(str(path))
            except Exception:  # noqa: BLE001 - config-loading coverage is elsewhere.
                continue
            if not set(VOLUME) & set(config.client.metrics or ()):
                continue
            checked += 1
            with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                dropped = _client_metrics_the_server_filter_removes(config)
                self.assertEqual([name for name in dropped if name in VOLUME], [])
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
        raw["defaults"]["global_rounds"] = 7
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
