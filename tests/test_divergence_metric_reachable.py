"""A divergence.metric either metrics filter drops is refused at load.

The defect has the same shape as checkpointing.best_metric naming a column no
run produces, which _validate_checkpoint_metric_is_emitted already refuses: a
name passes every syntactic check and still never appears.

There are two filters between the task and the round record, and this check
covered one. The client-side filter is the earlier and stricter: every rule
runs its fit metrics through client.metrics before the FitResult exists, so a
list omitting fit_loss means no client ever reports it, the server aggregate
has nothing to filter, and the default divergence.metric watches a name no
round can contain. Nothing checked it, and the chapter said two checks covered
every way a name goes missing.

It is refused in validate_config, on the run path via load_config, and not in
validation.py. validate_full_config runs only under --validate-only, so an
issue raised there does not stop an ordinary run -- the config that silences
the monitor would still train. That is the distinction commit "Refuse fedprox
and scaffold options their plain SGD step cannot honour" established, and the
last class below holds the check to it.

What it costs is larger here. Every divergence detector reads the one metric
name, so a filtered-out name silences all of them -- non_finite included --
for the whole run, and nothing fails. The loop does notice, and prints
`divergence.metric=... was never present in any round's metrics` -- after the
final round, which on a 500-round FEMNIST arm is several GPU-hours after the
point it would have been worth knowing.

Only filtered names are in scope, and the two lists filter different sets.
Neither reaches the evaluation columns, the central-test metrics or the
strategy's own diagnostics, so a divergence.metric naming one of those is safe
whatever either list says. client.metrics additionally cannot remove the
extras five of the nine rules add after filtering, which server.metrics governs
for every rule. Those are the cases below that assert the config *loads*: a
check that fires on a correct config is worse than no check, because the next
person turns it off.
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


def _fedlalr_config(
    *,
    server_metrics: list[str] | None = None,
    client_metrics: list[str] | None = None,
    divergence_metric: str = "fit_loss",
):
    """The smoke base as a valid FedLALR pair: both halves, no options it refuses.

    The guards below that assert a FedLALR config *loads* used to set only
    server.strategy, so the pairing refusal fired first and a load assertion
    that read "no divergence.metric refusal" passed whatever the check did.
    FINDINGS.csv POST-F13. Built from YAML so the pair is judged the way a run
    judges it.
    """

    raw = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
    raw["server"]["strategy"] = "fedlalr"
    raw["client"] = {
        "update_rule": "fedlalr",
        "batch_size": 4,
        "learning_rate": 0.01,
        "metrics": ["fit_loss"],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fedlalr.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config = load_config(path)
    if server_metrics is not None:
        config.server.metrics = server_metrics
    if client_metrics is not None:
        config.client.metrics = client_metrics
    config.divergence.metric = divergence_metric
    config.divergence.non_finite = True
    return config


def _refuses(config) -> bool:
    message = _refusal(config)
    return message is not None and "divergence.metric" in message


def _config(
    *,
    server_metrics: list[str] | None = None,
    client_metrics: list[str] | None = None,
    divergence_metric: str = "fit_loss",
):
    config = copy.deepcopy(load_config(BASE))
    if server_metrics is not None:
        config.server.metrics = server_metrics
    if client_metrics is not None:
        config.client.metrics = client_metrics
    config.divergence.metric = divergence_metric
    config.divergence.non_finite = True
    return config


@pytest.mark.fast
class ItFiresOnTheRealDefectTest(unittest.TestCase):
    def test_a_non_empty_list_that_omits_the_watched_metric(self) -> None:
        self.assertTrue(_refuses(_config(server_metrics=["fit_accuracy"])))

    def test_the_message_names_the_metric_and_both_ways_out(self) -> None:
        message = _refusal(_config(server_metrics=["fit_accuracy"]))
        assert message is not None
        self.assertIn("fit_loss", message)
        self.assertIn("server.metrics", message)
        self.assertIn("empty", message)


@pytest.mark.fast
class TheClientFilterIsTheOtherWayTest(unittest.TestCase):
    """The earlier of the two filters, and the one nothing checked.

    A client.metrics that omits the watched name drops it before the FitResult
    exists, so server.metrics -- however permissive -- has nothing to keep.
    """

    def test_a_client_list_that_omits_the_watched_metric(self) -> None:
        self.assertTrue(_refuses(_config(client_metrics=["fit_accuracy"])))

    def test_it_fires_even_when_server_metrics_names_it(self) -> None:
        """server.metrics keeping a name the client never sent is no defence."""

        config = _config(client_metrics=["fit_accuracy"], server_metrics=["fit_loss"])
        self.assertTrue(_refuses(config))

    def test_the_message_names_the_client_list(self) -> None:
        message = _refusal(_config(client_metrics=["fit_accuracy"]))
        assert message is not None
        self.assertIn("fit_loss", message)
        self.assertIn("client.metrics", message)
        self.assertIn("empty", message)

    def test_a_shipped_config_with_fit_loss_removed(self) -> None:
        """A real arm's client list, minus the watched name, as a refusal."""

        config = copy.deepcopy(load_config("configs/femnist/fedavg.yaml"))
        self.assertIn("fit_loss", config.client.metrics)
        self.assertEqual(config.divergence.metric, "fit_loss")
        config.client.metrics = [name for name in config.client.metrics if name != "fit_loss"]
        self.assertTrue(_refuses(config))

    def test_an_empty_client_list_keeps_everything(self) -> None:
        self.assertFalse(_refuses(_config(client_metrics=[])))

    def test_a_client_list_that_names_it(self) -> None:
        self.assertFalse(_refuses(_config(client_metrics=["fit_loss", "fit_accuracy"])))

    def test_an_evaluation_column_is_not_a_client_fit_metric(self) -> None:
        config = _config(
            client_metrics=["fit_accuracy"],
            divergence_metric="val_accuracy_sample_weighted_avg",
        )
        self.assertFalse(_refuses(config))

    def test_an_extra_this_rule_adds_after_filtering_is_exempt(self) -> None:
        """local_sgd adds communicated_bytes after the filter, so it survives.

        server.metrics is emptied because that filter *does* govern the name:
        the two exemptions are different sets, and this isolates the client's.
        """

        config = _config(
            client_metrics=["fit_accuracy"],
            server_metrics=[],
            divergence_metric="communicated_bytes",
        )
        self.assertEqual(config.client.update_rule, "local_sgd")
        self.assertFalse(_refuses(config))

    def test_a_rule_that_filters_its_extras_is_not_exempt(self) -> None:
        """delta_sgd filters everything, so the same name is refused there."""

        config = _config(
            client_metrics=["fit_accuracy"],
            server_metrics=[],
            divergence_metric="communicated_bytes",
        )
        config.client.update_rule = "delta_sgd"
        self.assertTrue(_refuses(config))

    def test_the_server_filter_still_governs_a_client_exempt_name(self) -> None:
        """Exempt from one filter is not exempt from the other."""

        config = _config(
            client_metrics=["fit_accuracy"],
            server_metrics=["fit_loss"],
            divergence_metric="communicated_bytes",
        )
        message = _refusal(config)
        assert message is not None
        self.assertIn("server.metrics", message)

    def test_another_rule_s_extras_are_not_borrowed(self) -> None:
        """client_learning_rate is a FedAvgClient extra; local_sgd has none."""

        config = _config(
            client_metrics=["fit_accuracy"],
            server_metrics=[],
            divergence_metric="client_learning_rate",
        )
        self.assertTrue(_refuses(config))
        # A complete fedavg client, so nothing else is refused first and the
        # assertion below is about the exemption rather than about the order
        # validate_config runs its checks in. POST-F13.
        config.client.update_rule = "fedavg"
        config.client.extra["update_mode"] = "sequential_epoch"
        config.client.extra["frozen_gradient_weighting"] = "examples"
        self.assertIsNone(_refusal(config))


@pytest.mark.fast
class ItStaysQuietWhenTheMetricSurvivesTest(unittest.TestCase):
    """Four ways a metrics list cannot hide the watched metric."""

    def test_an_empty_list_keeps_everything(self) -> None:
        self.assertFalse(_refuses(_config(server_metrics=[])))

    def test_a_list_that_names_it(self) -> None:
        self.assertFalse(_refuses(_config(server_metrics=["fit_loss", "fit_accuracy"])))

    def test_an_evaluation_column_is_never_filtered(self) -> None:
        """The point of docs/08-metrics.md section 4.3, as a preflight case."""

        config = _config(
            server_metrics=["fit_accuracy"],
            divergence_metric="val_accuracy_sample_weighted_avg",
        )
        self.assertFalse(_refuses(config))

    def test_a_central_test_metric_is_never_filtered(self) -> None:
        config = _config(server_metrics=["fit_accuracy"], divergence_metric="central_test_loss")
        self.assertFalse(_refuses(config))

    def test_a_task_supplied_central_test_metric_is_never_filtered_either(self) -> None:
        """`central_test_` is checked by prefix, not against the two names
        the framework itself produces: `_evaluate_central_test_set` passes
        through any finite numeric key a task's `evaluate_global` reports, and
        none of it passes through filter_metrics regardless of its name."""

        config = _config(
            server_metrics=["fit_accuracy"],
            divergence_metric="central_test_optimality_gap",
        )
        self.assertFalse(_refuses(config))

    def test_the_strategy_s_own_diagnostics_are_never_filtered(self) -> None:
        """Since the diagnostics moved after the filter, watching one is legal.

        momentum_norm blowing up is a divergence signal, so this is a config
        someone would plausibly write. Reporting it would be a false error.
        On a complete FedLALR pair, and asserting no refusal of any kind: the
        version of this test that paired fedlalr with local_sgd passed whatever
        the check did. POST-F13.
        """

        config = _fedlalr_config(server_metrics=["fit_accuracy"], divergence_metric="momentum_norm")
        self.assertIsNone(_refusal(config))

    def test_another_strategy_s_diagnostics_are_not_borrowed(self) -> None:
        """The exemption is per strategy, not a union over all of them."""

        config = _config(server_metrics=["fit_accuracy"], divergence_metric="momentum_norm")
        config.server.strategy = "fedavg"
        self.assertTrue(_refuses(config))

    def test_divergence_switched_off_entirely(self) -> None:
        config = _config(server_metrics=["fit_accuracy"])
        config.divergence.non_finite = False
        config.divergence.blowup_factor = None
        config.divergence.blowup_absolute = None
        config.divergence.patience = None
        self.assertFalse(config.divergence.active)
        self.assertFalse(_refuses(config))


@pytest.mark.fast
class ASpreadNeedsTheMetricItIsBuiltFromTest(unittest.TestCase):
    """FedLALR's across-clients spread is built from each client's coordinate mean.

    With a non-empty client.metrics that drops the mean, no round carries the
    spread, so asking for it -- to watch, or in either list -- is refused at
    load, naming the metric to keep. The reachability check used to exempt
    every strategy diagnostic whole, so the monitor loaded and stayed silent
    for the run. FINDINGS.csv POST-F12.
    """

    SOURCE = "effective_learning_rate_coordinate_mean"
    SPREAD = [f"effective_learning_rate_across_clients_{s}" for s in ("mean", "std", "min", "max")]

    def test_watching_a_spread_column_without_its_source_is_refused(self) -> None:
        for name in self.SPREAD:
            with self.subTest(metric=name):
                message = _refusal(
                    _fedlalr_config(
                        server_metrics=[], client_metrics=["fit_loss"], divergence_metric=name
                    )
                )
                self.assertIsNotNone(message)
                self.assertIn(f"divergence.metric names {name!r}", message)
                self.assertIn(f"Add {self.SOURCE!r} to client.metrics", message)

    def test_asking_for_one_in_a_metrics_list_without_its_source_is_refused(self) -> None:
        for key in ("server_metrics", "client_metrics"):
            with self.subTest(list=key):
                lists = {"server_metrics": [], "client_metrics": ["fit_loss"]}
                lists[key] = [*lists[key], self.SPREAD[1]]
                message = _refusal(_fedlalr_config(**lists))
                self.assertIsNotNone(message)
                self.assertIn(f"names {self.SPREAD[1]!r}", message)
                self.assertIn(self.SOURCE, message)

    def test_with_the_source_kept_every_spread_column_loads(self) -> None:
        for name in self.SPREAD:
            with self.subTest(metric=name):
                config = _fedlalr_config(
                    server_metrics=["fit_loss"],
                    client_metrics=["fit_loss", self.SOURCE],
                    divergence_metric=name,
                )
                self.assertIsNone(_refusal(config))

    def test_an_empty_client_list_keeps_the_source(self) -> None:
        config = _fedlalr_config(
            server_metrics=[], client_metrics=[], divergence_metric=self.SPREAD[2]
        )
        self.assertIsNone(_refusal(config))


class NoShippedConfigTripsItTest(unittest.TestCase):
    """A new preflight error has to be checked against the tree it ships with.

    Every shipped config that sets a non-empty server.metrics already lists
    fit_loss. If one stops doing so this fails here rather than in a run.
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
        raw["server"]["metrics"] = ["fit_accuracy"]
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
    two servers. This asserts the preflight table and that test's table are
    the same set, so a diagnostic added to a server reaches preflight too
    instead of producing a wrong verdict here.
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

    Five rules filter the task metrics and then add their extras, so those
    extras survive any list; four assemble everything and filter the lot, so
    nothing survives a list that does not name it. That split is not written
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

    @pytest.mark.fast
    def test_the_table_decides_the_refusal_for_every_rule(self) -> None:
        """From table to verdict, per rule, on a name every client computes.

        The check is called directly rather than through validate_config:
        swapping update_rule alone makes a paired config half-paired, and that
        refusal fires first and would answer a different question.
        """

        from test_client_communication_cost import BUILDERS

        from fedbrew.core.config import _validate_divergence_metric_is_reachable

        for rule in sorted(BUILDERS):
            with self.subTest(rule=rule):
                config = _config(
                    client_metrics=["fit_accuracy"],
                    server_metrics=[],
                    divergence_metric="communicated_bytes",
                )
                config.client.update_rule = rule
                exempt = "communicated_bytes" in CLIENT_UNFILTERED_FIT_METRICS.get(rule, ())
                try:
                    _validate_divergence_metric_is_reachable(config)
                    refused = False
                except ValueError:
                    refused = True
                self.assertEqual(
                    refused,
                    not exempt,
                    f"{rule} is {'exempt' if exempt else 'not exempt'} in the "
                    "table and the check disagrees",
                )


if __name__ == "__main__":
    unittest.main()
