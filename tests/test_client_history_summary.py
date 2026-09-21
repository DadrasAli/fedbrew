"""run.json's scale block must not re-read the whole run every round.

on_round_flush rewrites run.json each round so a killed job's artifacts stay
readable -- a real requirement. But the six aggregates in its scale block were
each recomputed by scanning every per-client record accumulated since round 1,
and those lists are never truncated: one append per client per round for the
rest of the run. Per-round cost was O(records so far) and cumulative cost was
quadratic in round count, and it scaled with the square of the client count
as well. It fired regardless of per_client_csv, which
only gates whether the CSVs are written.

The histories now carry their own running totals. These tests pin that the
totals are right, that the fast path really avoids the scan, that the summary
cannot drift from the list, and that a plain list still works.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest

from fedbrew.core import runner
from fedbrew.core.artifacts import (
    _client_update_metric_phase_counts,
    _total_client_examples_processed,
    _total_client_fits,
    _total_client_test_examples,
    _total_client_train_examples,
    _unique_clients,
)
from fedbrew.core.state import (
    ClientEvaluationHistory,
    ClientEvaluationRecord,
    ClientMetricRecord,
    ClientUpdateHistory,
)


def _evaluation(round_id: int, client_id: str) -> ClientEvaluationRecord:
    return ClientEvaluationRecord(
        round_id=round_id,
        client_id=client_id,
        participated=True,
        train_num_examples=10,
        test_num_examples=3,
        global_model_train_loss=1.0,
        global_model_train_accuracy=0.5,
        global_model_test_loss=1.0,
        global_model_test_accuracy=0.5,
    )


def _update(round_id: int, client_id: str, phase: str = "fit") -> ClientMetricRecord:
    return ClientMetricRecord(
        round_id=round_id,
        client_id=client_id,
        phase=phase,
        num_examples=7,
        metrics={"fit_loss": 1.0},
    )


def _populate() -> tuple[ClientEvaluationHistory, ClientUpdateHistory]:
    evaluations = ClientEvaluationHistory()
    updates = ClientUpdateHistory()
    for round_id in (1, 2, 3):
        evaluations.extend(_evaluation(round_id, f"c{index}") for index in range(4))
        updates.extend(_update(round_id, f"c{index}") for index in range(3))
        updates.append(_update(round_id, "c0", phase="eval"))
    return evaluations, updates


@pytest.mark.fast
class SummaryValueTest(unittest.TestCase):
    """Every total the scale block needs, against the scan it replaces."""

    def test_the_totals_match_a_full_scan(self) -> None:
        evaluations, updates = _populate()
        plain_evaluations = list(evaluations)
        plain_updates = list(updates)

        self.assertEqual(
            _unique_clients(evaluations, updates),
            _unique_clients(plain_evaluations, plain_updates),
        )
        self.assertEqual(
            _client_update_metric_phase_counts(updates),
            _client_update_metric_phase_counts(plain_updates),
        )
        self.assertEqual(_total_client_fits([], updates), _total_client_fits([], plain_updates))
        self.assertEqual(
            _total_client_train_examples(evaluations),
            _total_client_train_examples(plain_evaluations),
        )
        self.assertEqual(
            _total_client_test_examples(evaluations),
            _total_client_test_examples(plain_evaluations),
        )
        self.assertEqual(
            _total_client_examples_processed([], evaluations, updates),
            _total_client_examples_processed([], plain_evaluations, plain_updates),
        )

    def test_the_values_are_the_ones_expected(self) -> None:
        evaluations, updates = _populate()
        self.assertEqual(_unique_clients(evaluations, updates), {"c0", "c1", "c2", "c3"})
        self.assertEqual(_client_update_metric_phase_counts(updates), {"fit": 9, "eval": 3})
        self.assertEqual(_total_client_fits([], updates), 9)
        self.assertEqual(_total_client_train_examples(evaluations), 120)
        self.assertEqual(_total_client_test_examples(evaluations), 36)
        # 12 evaluations x 13 examples + 12 updates x 7 examples
        self.assertEqual(_total_client_examples_processed([], evaluations, updates), 156 + 84)

    def test_an_empty_history_reports_the_fit_phase(self) -> None:
        """The scan seeded {"fit": 0}; the fast path has to as well, or a run
        with no fits would drop the key instead of reporting zero."""

        self.assertEqual(_client_update_metric_phase_counts(ClientUpdateHistory()), {"fit": 0})


@pytest.mark.fast
class NoFullScanTest(unittest.TestCase):
    """The point of the change: the helpers stop reading the history."""

    class _RefusingUpdates(ClientUpdateHistory):
        __slots__ = ()

        def __iter__(self):  # type: ignore[override]
            raise AssertionError("the summary path must not scan the history")

    class _RefusingEvaluations(ClientEvaluationHistory):
        __slots__ = ()

        def __iter__(self):  # type: ignore[override]
            raise AssertionError("the summary path must not scan the history")

    def test_no_helper_iterates_a_summarised_history(self) -> None:
        evaluations = self._RefusingEvaluations()
        updates = self._RefusingUpdates()
        for round_id in (1, 2):
            for index in range(3):
                evaluations.append(_evaluation(round_id, f"c{index}"))
                updates.append(_update(round_id, f"c{index}"))

        # Each of these scanned every record in the run before the change.
        self.assertEqual(_unique_clients(evaluations, updates), {"c0", "c1", "c2"})
        self.assertEqual(_client_update_metric_phase_counts(updates), {"fit": 6})
        self.assertEqual(_total_client_fits([], updates), 6)
        self.assertEqual(_total_client_train_examples(evaluations), 60)
        self.assertEqual(_total_client_test_examples(evaluations), 18)
        self.assertEqual(_total_client_examples_processed([], evaluations, updates), 78 + 42)

    def test_a_plain_list_still_gets_the_full_scan(self) -> None:
        """Standalone callers and tests pass ordinary lists."""

        evaluations, updates = _populate()
        self.assertEqual(_total_client_fits([], list(updates)), 9)
        self.assertEqual(_total_client_train_examples(list(evaluations)), 120)


@pytest.mark.fast
class AppendOnlyTest(unittest.TestCase):
    """A summary that can drift is worse than the scan it replaces."""

    def test_every_mutation_that_would_desynchronise_is_refused(self) -> None:
        _, updates = _populate()
        for name, call in (
            ("pop", lambda: updates.pop()),
            ("remove", lambda: updates.remove(updates[0])),
            ("clear", lambda: updates.clear()),
            ("insert", lambda: updates.insert(0, _update(9, "z"))),
            ("sort", lambda: updates.sort(key=lambda record: record.round_id)),
            ("reverse", lambda: updates.reverse()),
            ("setitem", lambda: updates.__setitem__(0, _update(9, "z"))),
            ("delitem", lambda: updates.__delitem__(0)),
        ):
            with self.subTest(mutation=name):
                with self.assertRaises(TypeError):
                    call()

    def test_extend_accumulates_every_record(self) -> None:
        updates = ClientUpdateHistory()
        updates.extend(_update(1, f"c{index}") for index in range(5))
        self.assertEqual(len(updates), 5)
        self.assertEqual(updates.summary.phase_counts, {"fit": 5})
        self.assertEqual(updates.summary.num_examples, 35)

    def test_reading_the_history_is_unchanged(self) -> None:
        """The CSV writers and the round table index and iterate these."""

        _, updates = _populate()
        self.assertEqual(len(updates), 12)
        self.assertEqual(updates[0].client_id, "c0")
        self.assertEqual(len(list(updates)), 12)
        self.assertEqual(len(updates[1:3]), 2)


def _write_config(directory: Path) -> Path:
    config_path = directory / "history.yaml"
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
              num_clients: 3
              samples_per_client: 8
              input_dim: 4
              num_classes: 2
            model:
              name: mlp
              input_dim: 4
              hidden_dim: 4
              num_classes: 2
            runtime:
              deterministic: true
              device: cpu
              use_amp: false
            evaluation:
              train:
                every: 1
                clients: all
              test:
                every: 1
                clients: all
            defaults:
              global_rounds: 3
              local_iterations: 1
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


class RealRunTest(unittest.TestCase):
    """The scale block a real run writes must equal a full rescan of it."""

    def test_run_json_scale_matches_a_scan_of_the_same_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = runner.run(_write_config(root), runner.parse_args(["--quiet"]))
            written = json.loads((root / "run" / "run.json").read_text(encoding="utf-8"))
            evaluations = list(state.client_metrics_history)
            updates = list(state.client_update_metrics_history)
            # The loop has to be filling the summarised histories, or the run
            # is still paying the per-round scan and this comparison is
            # comparing a scan against a scan.
            self.assertIsInstance(state.client_metrics_history, ClientEvaluationHistory)
            self.assertIsInstance(state.client_update_metrics_history, ClientUpdateHistory)
            self.assertEqual(
                state.client_update_metrics_history.summary.phase_counts,
                _client_update_metric_phase_counts(updates),
            )
            self.assertEqual(
                state.client_metrics_history.summary.train_examples,
                _total_client_train_examples(evaluations),
            )

        scale = written["scale"]
        self.assertEqual(scale["unique_clients"], len(_unique_clients(evaluations, updates)))
        self.assertEqual(
            scale["client_update_metric_phase_counts"],
            _client_update_metric_phase_counts(updates),
        )
        self.assertEqual(scale["total_client_fits"], _total_client_fits([], updates))
        self.assertEqual(
            scale["total_client_train_examples_evaluated"],
            _total_client_train_examples(evaluations),
        )
        self.assertEqual(
            scale["total_client_test_examples_evaluated"],
            _total_client_test_examples(evaluations),
        )
        self.assertEqual(
            scale["total_client_examples_processed"],
            _total_client_examples_processed([], evaluations, updates),
        )
        # Not a trivially-empty comparison.
        self.assertGreater(scale["total_client_fits"], 0)
        self.assertGreater(scale["unique_clients"], 0)


if __name__ == "__main__":
    unittest.main()
