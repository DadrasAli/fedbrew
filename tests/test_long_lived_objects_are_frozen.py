"""What a run's first round built is set aside from the collector until the run ends.

``run_fl_loop`` freezes every tracked object once its first round is done
(``_LongLivedObjects``, chapter 11 §9), so later full collections do not
traverse the clients, shards and models that live for the whole run, and
unfreezes them when the run ends -- however it ends. Pinned here: nothing is
frozen during the first round, everything is from the second on, nothing is
left frozen afterwards, and a process that froze objects itself, or turned
the collector off, is left as it was. What the interpreter froze at its own
start (CPython 3.12 freezes a few hundred tuples; earlier releases none) does
not count as the process's own.
"""

from __future__ import annotations

import gc
import unittest
from unittest import mock

from fedbrew.core import loop
from fedbrew.core.protocol import FitRequest, FitResult
from tests.test_resume_metrics_continuity import _EVALUATION, _Client, _Dataset, _Server

ROUNDS = 3


class _FailingClient(_Client):
    def fit(self, request: FitRequest) -> FitResult:
        if request.round_id == 2:
            raise RuntimeError("round 2 fails")
        return super().fit(request)


def _run(client: type[_Client] = _Client) -> list[tuple[int, int]]:
    """Run the fixture; return (round, freeze count) at each client's fit."""

    seen: list[tuple[int, int]] = []

    def progress(round_id: int, done: int, total: int, phase: str) -> None:
        if phase == "fit" and done:
            seen.append((round_id, gc.get_freeze_count()))

    loop.run_fl_loop(
        server=_Server(),
        client={"client_0": client(), "client_1": client()},
        dataset=_Dataset(),
        global_rounds=ROUNDS,
        evaluation=_EVALUATION,
        on_client_progress=progress,
    )
    return seen


class LongLivedObjectsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.start = gc.get_freeze_count()
        self.assertLessEqual(
            self.start, loop._FROZEN_AT_IMPORT, "nothing but the interpreter has frozen objects"
        )
        self.addCleanup(gc.unfreeze)

    def test_frozen_from_the_second_round_and_released_at_the_end(self) -> None:
        seen = _run()
        self.assertEqual([round_id for round_id, _ in seen], [1, 1, 2, 2, 3, 3])
        self.assertTrue(all(count == self.start for round_id, count in seen if round_id == 1))
        self.assertTrue(all(count > self.start for round_id, count in seen if round_id > 1))
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_a_run_that_raises_releases_them(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "round 2 fails"):
            _run(_FailingClient)
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_what_the_process_froze_itself_is_left_to_it(self) -> None:
        gc.freeze()
        with (
            mock.patch.object(gc, "freeze", wraps=gc.freeze) as freeze,
            mock.patch.object(gc, "unfreeze", wraps=gc.unfreeze) as unfreeze,
        ):
            _run()
        # The run neither froze more nor released what the process froze.
        # The count of frozen objects cannot say so: an object frozen here
        # that another thread lets go of while the run goes on (an earlier
        # test's idle executor, tqdm's monitor) leaves the count, so it fell
        # mid-run once in a serial run of the whole suite.
        freeze.assert_not_called()
        unfreeze.assert_not_called()
        self.assertGreater(gc.get_freeze_count(), self.start)

    def test_nothing_is_frozen_with_the_collector_off(self) -> None:
        gc.disable()
        self.addCleanup(gc.enable)
        seen = _run()
        self.assertTrue(all(count == self.start for _, count in seen))


if __name__ == "__main__":
    unittest.main()
