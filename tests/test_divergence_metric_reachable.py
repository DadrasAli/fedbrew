"""A divergence.metric that reporting.fit_metrics drops is refused at load.

The defect has the same shape as checkpointing.best_metric naming a column no
run produces, which _validate_checkpoint_metric_is_emitted already refuses: a
name passes every syntactic check and still never appears.

There used to be two filters between the task and the round record,
``client.metrics`` and ``server.metrics``, each with its own exemptions, and
the check had to follow both. They are one list now, applied once by the
server after every client and server metric is added (docs/04 section 9),
so the check is one question: does a non-empty ``reporting.fit_metrics`` name
the watched metric?

It is refused in validate_config, on the run path via load_config, and not in
validation.py. validate_full_config runs only under --validate-only, so an
issue raised there does not stop an ordinary run -- the config that silences
the monitor would still train. That is the distinction commit "Refuse fedprox
and scaffold options their plain SGD step cannot honour" established, and a
class below holds the check to it.

What it costs is larger here. Every divergence detector reads the one metric
name, so a filtered-out name silences all of them -- non_finite included --
for the whole run, and nothing fails. The loop does notice, and prints
`divergence.metric=... was never present in any round's metrics` -- after the
final round, which on a 500-round FEMNIST arm is several GPU-hours after the
point it would have been worth knowing.

Only filtered names are in scope. The list does not reach the evaluation
columns or the central-test metrics, so a divergence.metric naming one of those
is safe whatever it says. Those are the cases below that assert the config
*loads*: a check that fires on a correct config is worse than no check,
because the next person turns it off.
"""

from __future__ import annotations

import copy
import glob
import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

from fedbrew.core.config import load_config, validate_config
from fedbrew.core.metrics import CLIENT_UNFILTERED_FIT_METRICS, SERVER_DIAGNOSTIC_METRICS

BASE = "configs/dev/smoke.yaml"


def _refusal(config) -> str | None:
    """The message validate_config raises for this config, or None."""

    try:
        validate_config(config)
    except ValueError as error:
        return str(error)
    return None


def _fedlalr_config(*, fit_metrics: list[str] | None = None, divergence_metric: str = "fit_loss"):
    """The smoke base as a valid FedLALR pair: both halves, no options it refuses.

    The guards below that assert a FedLALR config *loads* used to set only
    server.strategy, so the pairing refusal fired first and a load assertion
    that read "no divergence.metric refusal" passed whatever the check did.
    FINDINGS.csv POST-F13. Built from YAML so the pair is judged the way a run
    judges it.
    """

    raw = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
    raw["server"]["strategy"] = "fedlalr"
    raw["client"] = {"update_rule": "fedlalr", "batch_size": 4, "learning_rate": 0.01}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fedlalr.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config = load_config(path)
    if fit_metrics is not None:
        config.reporting.fit_metrics = fit_metrics
    config.divergence.metric = divergence_metric
    config.divergence.non_finite = True
    return config


def _refuses(config) -> bool:
    message = _refusal(config)
    return message is not None and "divergence.metric" in message


def _config(*, fit_metrics: list[str] | None = None, divergence_metric: str = "fit_loss"):
    config = copy.deepcopy(load_config(BASE))
    if fit_metrics is not None:
        config.reporting.fit_metrics = fit_metrics
    config.divergence.metric = divergence_metric
    config.divergence.non_finite = True
    return config


@pytest.mark.fast
class ItFiresOnTheRealDefectTest(unittest.TestCase):
    def test_a_non_empty_list_that_omits_the_watched_metric(self) -> None:
        self.assertTrue(_refuses(_config(fit_metrics=["fit_accuracy"])))

    def test_the_message_names_the_metric_and_both_ways_out(self) -> None:
        message = _refusal(_config(fit_metrics=["fit_accuracy"]))
        assert message is not None
        self.assertIn("fit_loss", message)
        self.assertIn("reporting.fit_metrics", message)
        self.assertIn("empty", message)

    def test_a_shipped_config_with_fit_loss_removed(self) -> None:
        """A real arm's list, minus the watched name, as a refusal."""

        config = copy.deepcopy(load_config("configs/femnist/fedavg.yaml"))
        self.assertIn("fit_loss", config.reporting.fit_metrics)
        self.assertEqual(config.divergence.metric, "fit_loss")
        config.reporting.fit_metrics = [
            name for name in config.reporting.fit_metrics if name != "fit_loss"
        ]
        self.assertTrue(_refuses(config))


@pytest.mark.fast
class NothingIsExemptFromTheListTest(unittest.TestCase):
    """Every fit-side column goes through the one list: the rule's and the strategy's too.

    Under the two lists, the extras five rules added after the client filter
    and the strategy diagnostics added after the server filter were exempt
    from one list or both, and the check had to know which. Now a non-empty
    list either names the watched column or drops it.
    """

    def test_a_rule_extra_the_list_omits_is_refused(self) -> None:
        config = _config(fit_metrics=["fit_accuracy"], divergence_metric="communicated_bytes")
        self.assertTrue(_refuses(config))

    def test_a_rule_extra_the_list_names_loads(self) -> None:
        config = _config(
            fit_metrics=["fit_accuracy", "communicated_bytes"],
            divergence_metric="communicated_bytes",
        )
        self.assertFalse(_refuses(config))

    def test_a_strategy_diagnostic_the_list_omits_is_refused(self) -> None:
        """momentum_norm blowing up is a divergence signal, and a plausible watch."""

        message = _refusal(
            _fedlalr_config(fit_metrics=["fit_accuracy"], divergence_metric="momentum_norm")
        )
        assert message is not None
        self.assertIn("reporting.fit_metrics", message)

    def test_a_strategy_diagnostic_the_list_names_loads(self) -> None:
        """On a complete FedLALR pair, asserting no refusal of any kind. POST-F13."""

        config = _fedlalr_config(
            fit_metrics=["fit_accuracy", "momentum_norm"], divergence_metric="momentum_norm"
        )
        self.assertIsNone(_refusal(config))

    def test_every_rule_is_judged_the_same(self) -> None:
        """No per-rule exemption table decides the verdict any more.

        The check is called directly rather than through validate_config:
        swapping update_rule alone makes a paired config half-paired, and that
        refusal fires first and would answer a different question.
        """

        from test_client_communication_cost import BUILDERS

        from fedbrew.core.config import _validate_divergence_metric_is_reachable

        for rule in sorted(BUILDERS):
            with self.subTest(rule=rule):
                config = _config(
                    fit_metrics=["fit_accuracy"], divergence_metric="communicated_bytes"
                )
                config.client.update_rule = rule
                with self.assertRaises(ValueError):
                    _validate_divergence_metric_is_reachable(config)


@pytest.mark.fast
class ItStaysQuietWhenTheMetricSurvivesTest(unittest.TestCase):
    """The ways the list cannot hide the watched metric."""

    def test_an_empty_list_keeps_everything(self) -> None:
        self.assertFalse(_refuses(_config(fit_metrics=[])))

    def test_a_list_that_names_it(self) -> None:
        self.assertFalse(_refuses(_config(fit_metrics=["fit_loss", "fit_accuracy"])))

    def test_an_evaluation_column_is_never_filtered(self) -> None:
        """The point of docs/08-metrics.md section 4.3, as a preflight case."""

        config = _config(
            fit_metrics=["fit_accuracy"],
            divergence_metric="val_accuracy_sample_weighted_avg",
        )
        self.assertFalse(_refuses(config))

    def test_a_central_test_metric_is_never_filtered(self) -> None:
        config = _config(fit_metrics=["fit_accuracy"], divergence_metric="central_test_loss")
        self.assertFalse(_refuses(config))

    def test_a_task_supplied_central_test_metric_is_never_filtered_either(self) -> None:
        """`central_test_` is checked by prefix, not against the two names
        the framework itself produces: `_evaluate_central_test_set` passes
        through any finite numeric key a task's `evaluate_global` reports, and
        none of it passes through filter_metrics regardless of its name."""

        config = _config(
            fit_metrics=["fit_accuracy"],
            divergence_metric="central_test_optimality_gap",
        )
        self.assertFalse(_refuses(config))

    def test_a_spread_column_needs_no_source_in_the_list(self) -> None:
        """FedLALR's spread is built from each client's coordinate mean.

        Under the two lists, a client list without the mean left the server
        nothing to spread and was refused (FINDINGS.csv POST-F12). The clients
        have no list now and always report it, so naming the spread is enough.
        """

        for name in (f"effective_learning_rate_across_clients_{s}" for s in ("mean", "max")):
            with self.subTest(metric=name):
                config = _fedlalr_config(fit_metrics=["fit_loss", name], divergence_metric=name)
                self.assertIsNone(_refusal(config))

    def test_divergence_switched_off_entirely(self) -> None:
        config = _config(fit_metrics=["fit_accuracy"])
        config.divergence.non_finite = False
        config.divergence.blowup_factor = None
        config.divergence.blowup_absolute = None
        config.divergence.patience = None
        self.assertFalse(config.divergence.active)
        self.assertFalse(_refuses(config))


class NoShippedConfigTripsItTest(unittest.TestCase):
    """A new preflight error has to be checked against the tree it ships with.

    Every shipped config that sets a non-empty reporting.fit_metrics already
    lists fit_loss. If one stops doing so this fails here rather than in a run.
    """

    def test_every_config_loads(self) -> None:
        """load_config now runs the check, so loading is the whole assertion."""

        offenders = []
        checked = 0
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                load_config(path)
            except ValueError as error:
                if "divergence.metric" in str(error):
                    offenders.append(f"{path}  {error}")
                continue
            except Exception:  # noqa: BLE001 - generator configs are not run configs
                continue
            checked += 1
        self.assertGreater(checked, 20, "the config scan found almost nothing")
        self.assertEqual(offenders, [], f"shipped configs trip the new check: {offenders}")


@pytest.mark.fast
class ItIsOnTheRunPathTest(unittest.TestCase):
    """The check has to be somewhere an ordinary run reaches.

    validate_full_config is called only from the --validate-only path in
    runner.py, so a check living there reports and does not gate: the config
    that silences the divergence monitor would still train. validate_config is
    called by load_config, which every run goes through.

    This was got wrong first time and moved. The test is here so it cannot
    drift back.
    """

    def test_load_config_itself_refuses(self) -> None:
        import tempfile

        import yaml

        raw = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
        raw["reporting"]["fit_metrics"] = ["fit_accuracy"]
        raw.setdefault("divergence", {})["metric"] = "fit_loss"
        raw["divergence"]["non_finite"] = True
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(raw, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError) as caught:
                load_config(path)
            self.assertIn("divergence.metric", str(caught.exception))
        finally:
            Path(path).unlink()

    def test_it_is_not_left_behind_in_the_preflight_module(self) -> None:
        """Two copies would drift, and the preflight one would not gate."""

        source = (
            Path(__file__).resolve().parent.parent / "fedbrew" / "core" / "validation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("metric_filtered_out", source)


@pytest.mark.fast
class TheDiagnosticsTableMatchesTheServersTest(unittest.TestCase):
    """SERVER_DIAGNOSTIC_METRICS is a claim about behaviour, so check behaviour.

    tests/test_metric_filter_scope.py already aggregates a round on each of the
    two servers. This asserts the plan header's table and that test's table
    are the same set, so a diagnostic added to a server reaches the header too.
    """

    def test_it_equals_what_the_servers_emit(self) -> None:
        from test_metric_filter_scope import DIAGNOSTICS

        observed = {
            module.removesuffix(".py").removesuffix("_server"): set(names)
            for module, names in DIAGNOSTICS.items()
        }
        table = {name: set(names) for name, names in SERVER_DIAGNOSTIC_METRICS.items()}
        self.assertEqual(table, observed)


class TheClientTableMatchesTheClientsTest(unittest.TestCase):
    """CLIENT_UNFILTERED_FIT_METRICS is a claim about behaviour too.

    A run gives its clients no list, but a client constructed with one still
    filters: five rules filter the task metrics and then add their extras, so
    those extras survive any list; four assemble everything and filter the
    lot, so nothing survives a list that does not name it. That split is not written
    down anywhere in the clients -- it is a consequence of where each one calls
    filter_metrics -- so the table is derived here rather than trusted.
    """

    def test_it_equals_what_survives_a_list_naming_nothing(self) -> None:
        from test_client_communication_cost import BUILDERS, _request

        observed: dict[str, set[str]] = {}
        for rule, build in sorted(BUILDERS.items()):
            client = build()
            client.metrics = ["no_such_metric"]
            survived = set(client.fit(_request()).metrics)
            if survived:
                observed[rule] = survived

        table = {name: set(names) for name, names in CLIENT_UNFILTERED_FIT_METRICS.items()}
        self.assertEqual(
            table,
            observed,
            "CLIENT_UNFILTERED_FIT_METRICS disagrees with what a client's fit "
            "returns under a metrics list that names nothing it computes",
        )


if __name__ == "__main__":
    unittest.main()
