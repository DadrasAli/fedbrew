"""What client.metrics and server.metrics govern, and what they do not.

Two decisions are pinned here, because they are the same decision: a metrics
list selects among *measured* metrics and reaches nothing else.

Two server strategies add numbers of their own to the round record: SCAFFOLD
its control-variate norms, FedLALR its optimizer-state norms and learning-rate
dispersion. Both must add them *after* filter_metrics has been applied to the
aggregated client metrics.

They did not agree. SCAFFOLD filtered first and so always emitted its two;
FedLALR filtered last and so dropped its own unless server.metrics happened to
list them. Same config key, opposite meaning,
depending on which strategy was selected -- a trap whichever side is right.

The side chosen is SCAFFOLD's, because it is the one the client already uses:
torch_sgd_client.py applies client.metrics to the task metrics and then adds
the algorithm extras, so client.metrics cannot remove them either. Under the
other order a metrics list can silently drop a column that
checkpointing.best_metric or the divergence guard names, and the run then
fails mid-flight on a missing key it never asked to lose.

Each case below filters down to one metric, which is what makes the assertion
mean something: fit_accuracy must be gone (the filter really ran) while every
diagnostic must be present (it ran before they were added).

The second decision is the evaluation path, which neither filter touches at
all. The alternative -- extending server.metrics over it -- was rejected
because eight shipped configs set checkpointing.best_metric to an evaluation
column, and the filter has no idea that key exists: a list that omitted it
would break checkpoint selection at the round it first tried to save. The
evaluation column set is controlled instead by client_statistics, per
statistic rather than per name. docs/08-metrics.md section 4.3 carries the
reasoning; the last class here holds it to the code.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.servers.fedlalr import FedLALRServer
from fedbrew.servers.scaffold import ScaffoldServer

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVERS = REPO_ROOT / "fedbrew" / "servers"

#: The client metrics every result below reports. Only the first is requested,
#: so the second is the probe that proves the filter is not a no-op.
KEPT = "fit_loss"
DROPPED = "fit_accuracy"

#: server module -> the diagnostics it adds after the filter.
DIAGNOSTICS = {
    "scaffold.py": ("server_control_norm", "mean_client_control_delta_norm"),
    "fedlalr.py": (
        "momentum_norm",
        "second_moment_norm",
        "effective_learning_rate_across_clients_mean",
        "effective_learning_rate_across_clients_std",
        "effective_learning_rate_across_clients_min",
        "effective_learning_rate_across_clients_max",
    ),
}


def _client_metrics(**extra: float) -> dict[str, float]:
    return {KEPT: 1.0, DROPPED: 0.5, **extra}


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"w": torch.full((2,), value)}


def _seed_model_state(server: object) -> None:
    """Give a server a model without building a task for it.

    initialize() would need a real model builder; these tests are about one
    dict of floats, so the state is set directly. The scope and metadata go
    with it because _federated_payload refuses to broadcast without them.
    """

    server._model_state = _state(0.0)
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}


class DiagnosticsSurviveTheFilterTest(unittest.TestCase):
    """One aggregate call per strategy, with server.metrics down to one name."""

    def _assert_shape(self, round_info: RoundInfo, module: str) -> None:
        metrics = round_info.metrics
        self.assertIn(KEPT, metrics, "the requested client metric was lost")
        self.assertNotIn(
            DROPPED,
            metrics,
            f"{module} did not apply server.metrics at all; this test cannot "
            "tell filter-then-add from no filter unless it does",
        )
        for name in DIAGNOSTICS[module]:
            with self.subTest(metric=name):
                self.assertIn(
                    name,
                    metrics,
                    f"{module} lost its own {name} to server.metrics. "
                    "Diagnostics are added after filter_metrics, not before.",
                )

    def test_scaffold(self) -> None:
        server = ScaffoldServer(participation_rate=1.0, seed=0, metrics=[KEPT])
        _seed_model_state(server)
        server._server_control = _state(0.0)
        # The roster size, which configure_round would set and this test skips.
        # SCAFFOLD refuses to aggregate without it rather than reading it off
        # the results, so a metric-filtering test has to say what N is even
        # though nothing here depends on the value.
        server._num_clients = 1
        round_info = RoundInfo(round_id=1)
        server.aggregate(
            round_info,
            [
                FitResult(
                    round_id=1,
                    client_id="a",
                    num_examples=10,
                    payload={"model_state": _state(1.0), "control_delta": _state(0.5)},
                    metrics=_client_metrics(),
                )
            ],
        )
        self._assert_shape(round_info, "scaffold.py")

    def test_fedlalr(self) -> None:
        server = FedLALRServer(
            epsilon=1e-8,
            participation_rate=1.0,
            seed=0,
            metrics=[KEPT],
            aggregation_weighting="uniform",
        )
        _seed_model_state(server)
        server._ensure_optimizer_state()
        round_info = RoundInfo(round_id=1)
        server.aggregate(
            round_info,
            [
                FitResult(
                    round_id=1,
                    client_id=client_id,
                    num_examples=10,
                    payload={
                        "model_state": _state(rate),
                        "model_state_scope": "full",
                        "model_state_metadata": {"model_state_scope": "full"},
                        "momentum_state": _state(0.0),
                        "second_moment_state": _state(1.0),
                    },
                    metrics=_client_metrics(effective_learning_rate_coordinate_mean=rate),
                )
                for client_id, rate in (("a", 2.0), ("b", 4.0))
            ],
        )
        self._assert_shape(round_info, "fedlalr.py")


class NoThirdServerTest(unittest.TestCase):
    """A new strategy that adds diagnostics has to make the same choice.

    Two tests above cover two modules. This one covers the third nobody
    has written yet: it fails when a server starts adding metrics of its own,
    so the order question is asked once, at the point of writing, rather than
    discovered from a missing column in a finished run.
    """

    @staticmethod
    def _adds_diagnostics(path: Path) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            target = node.func.value
            if (
                node.func.attr == "update"
                and isinstance(target, ast.Name)
                and target.id == "metrics"
            ):
                return True
        return False

    def test_only_the_documented_two_add_their_own(self) -> None:
        found = sorted(
            path.name
            for path in SERVERS.glob("*.py")
            if path.name != "__init__.py" and self._adds_diagnostics(path)
        )
        self.assertEqual(
            found,
            sorted(DIAGNOSTICS),
            "a server strategy gained or lost server-side diagnostics. Add it "
            "to DIAGNOSTICS with a case above, filter before adding, and list "
            "it in docs/08-metrics.md section 7.",
        )


class EvaluationPathIsOutsideBothFiltersTest(unittest.TestCase):
    """Neither list can shorten the evaluation columns, by construction."""

    def test_the_loop_never_filters(self) -> None:
        """The plainest form of the claim: the loop does not import it.

        _aggregate_client_split_metrics writes into round_info.metrics
        directly. If a future change routes the evaluation aggregate through
        filter_metrics, this fails and the chapter has to be rewritten with
        it.
        """

        source = (REPO_ROOT / "fedbrew" / "core" / "loop.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "filter_metrics",
            source,
            "the loop now filters; docs/08-metrics.md section 4.3 says it does "
            "not, and eight shipped configs point checkpointing.best_metric at "
            "a column this would be able to drop",
        )

    def test_the_evaluation_column_set_depends_only_on_client_statistics(self) -> None:
        """No metrics list is in scope, so none can be consulted."""

        from fedbrew.core.loop import _aggregate_client_split_metrics

        parameters = set(inspect.signature(_aggregate_client_split_metrics).parameters)
        self.assertEqual(parameters, {"results", "split", "statistics_config"})

    def test_the_request_overrides_client_metrics(self) -> None:
        """The evaluation request names its two metrics outright."""

        source = (REPO_ROOT / "fedbrew" / "core" / "loop.py").read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r'"metrics": \["loss", "accuracy"\]',
            "the evaluation request no longer fixes its metric list",
        )

    def test_the_shipped_configs_still_make_the_reason_true(self) -> None:
        """The argument rests on best_metric naming an evaluation column.

        If that stopped being true the trade-off would be worth revisiting, so
        the chapter's reasoning is checked and not just its conclusion.
        """

        offenders = []
        found = 0
        for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped.startswith("best_metric:"):
                    continue
                found += 1
                value = stripped.split(":", 1)[1].strip()
                # personal_ is the model_scope prefix, so personal_val_* is an
                # evaluation column too -- and one only the evaluation path
                # can produce, which makes the point more strongly.
                bare = value.removeprefix("personal_")
                if not bare.startswith(("train_", "val_", "test_", "central_test_")):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}  {value}")
        self.assertGreaterEqual(found, 8, "no shipped config sets best_metric")
        self.assertEqual(
            offenders,
            [],
            "these point best_metric at something other than an evaluation "
            f"column: {offenders}. docs/08-metrics.md section 4.3 argues from "
            "the fact that they do not.",
        )


class ChapterStatesTheDecisionTest(unittest.TestCase):
    """A deliberate choice has to read as one, or the next reader 'fixes' it."""

    def test_section_4_3_calls_it_deliberate_and_says_why(self) -> None:
        text = (REPO_ROOT / "docs" / "08-metrics.md").read_text(encoding="utf-8")
        section = text.split("### 4.3", 1)[1].split("## 5.", 1)[0]
        for phrase in (
            "design choice, not an oversight",
            "client_statistics",
            "checkpointing.best_metric",
            "evaluation.metrics",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, section)


if __name__ == "__main__":
    unittest.main()
