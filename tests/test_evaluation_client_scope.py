"""Which clients the post-aggregation evaluation runs over.

The distinction these tests pin down is why both sampling modes exist: a fixed
draw holds the client set constant so a round-over-round change in a metric can
only come from the model, while a per-round redraw tracks the whole client
population at the cost of sampling noise in every comparison.

Every call here passes ``"val"`` as the split. The draw is per split -- two
splits at the same ``N`` must not draw the same clients -- and
`tests/test_evaluation_draw_is_per_split.py` is where that property lives.
This module is about the four modes, so it holds the split fixed and varies
everything else.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.core.config import parse_evaluation_client_scope
from fedbrew.core.loop import _evaluation_client_selector
from fedbrew.core.protocol import ClientInfo

pytestmark = pytest.mark.fast

_ROSTER = [ClientInfo(client_id=f"c{index:04d}", num_examples=10) for index in range(200)]
_SELECTED = ["c0007", "c0003"]


def _ids(selector, round_id: int) -> list[str]:
    return [info.client_id for info in selector(round_id, _SELECTED)]


class ClientScopeParsingTests(unittest.TestCase):
    def test_the_four_supported_forms_parse(self) -> None:
        self.assertEqual(parse_evaluation_client_scope("all"), ("all", None))
        self.assertEqual(parse_evaluation_client_scope("participating"), ("participating", None))
        self.assertEqual(parse_evaluation_client_scope("sample:100"), ("sample", 100))
        self.assertEqual(parse_evaluation_client_scope("resample:250"), ("resample", 250))

    def test_a_typo_fails_at_config_load_rather_than_silently(self) -> None:
        for value in ("sample", "sample:0", "sample:x", "resample:-3", "both", 5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_evaluation_client_scope(value)


class ClientScopeSelectionTests(unittest.TestCase):
    def test_all_returns_the_whole_roster_every_round(self) -> None:
        selector = _evaluation_client_selector(_ROSTER, "all", 42, "val")

        self.assertEqual(len(_ids(selector, 1)), len(_ROSTER))
        self.assertEqual(_ids(selector, 1), _ids(selector, 9))

    def test_participating_follows_the_round_and_keeps_roster_order(self) -> None:
        selector = _evaluation_client_selector(_ROSTER, "participating", 42, "val")

        self.assertEqual([info.client_id for info in selector(1, _SELECTED)], _SELECTED)

    def test_sample_draws_once_and_reuses_it_every_round(self) -> None:
        selector = _evaluation_client_selector(_ROSTER, "sample:20", 42, "val")

        first = _ids(selector, 1)
        self.assertEqual(len(first), 20)
        self.assertEqual(first, _ids(selector, 2))
        self.assertEqual(first, _ids(selector, 500))

    def test_resample_draws_a_different_set_each_round(self) -> None:
        selector = _evaluation_client_selector(_ROSTER, "resample:20", 42, "val")

        rounds = [tuple(_ids(selector, round_id)) for round_id in (1, 2, 3)]
        self.assertEqual(len(set(rounds)), 3)
        self.assertTrue(all(len(clients) == 20 for clients in rounds))

    def test_both_modes_are_reproducible_from_the_seed_alone(self) -> None:
        # Nothing about the draw is persisted, so a resumed run has to be able
        # to rebuild the identical client set from the config.
        for scope in ("sample:20", "resample:20"):
            with self.subTest(scope=scope):
                one = _evaluation_client_selector(_ROSTER, scope, 42, "val")
                two = _evaluation_client_selector(_ROSTER, scope, 42, "val")
                self.assertEqual(_ids(one, 4), _ids(two, 4))

    def test_a_different_seed_draws_a_different_fixed_set(self) -> None:
        seeded = _evaluation_client_selector(_ROSTER, "sample:20", 42, "val")
        other = _evaluation_client_selector(_ROSTER, "sample:20", 7, "val")

        self.assertNotEqual(_ids(seeded, 1), _ids(other, 1))

    def test_the_draw_ignores_the_order_the_dataset_lists_clients_in(self) -> None:
        shuffled = list(reversed(_ROSTER))
        forward = _evaluation_client_selector(_ROSTER, "sample:20", 42, "val")
        backward = _evaluation_client_selector(shuffled, "sample:20", 42, "val")

        self.assertEqual(sorted(_ids(forward, 1)), sorted(_ids(backward, 1)))

    def test_results_come_back_in_roster_order(self) -> None:
        selector = _evaluation_client_selector(_ROSTER, "sample:20", 42, "val")

        chosen = _ids(selector, 1)
        self.assertEqual(chosen, sorted(chosen))

    def test_asking_for_more_clients_than_exist_evaluates_all_of_them(self) -> None:
        selector = _evaluation_client_selector(_ROSTER, "sample:5000", 42, "val")

        self.assertEqual(len(_ids(selector, 1)), len(_ROSTER))


if __name__ == "__main__":
    unittest.main()
