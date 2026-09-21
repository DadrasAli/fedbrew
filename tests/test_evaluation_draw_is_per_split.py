"""`val: sample:N` and `test: sample:N` must not draw the identical clients.

The draw key was ``f"evaluation_clients:{seed}:{sample_size}:{round_id}"`` --
seed, size, round, and nothing else. Two splits configured at the same ``N``
therefore got the same key and the same clients, so the clients whose
validation data selects `best.pt` were exactly the clients whose test data is
reported. The test number then inherits whatever makes that particular
subsample easy or hard, on top of the ordinary selection bias.

Two shipped configs are in that shape, and `TheShippedConfigsTest` reads them
rather than restating them:

    config                                val        test       overlap before
    configs/openimage/fedavg.yaml         sample:2000 sample:2000  2000/2000
    configs/reference_evaluation.yaml     sample:40   sample:40      40/40

`configs/reference_evaluation.yaml` even says the property it did not have, on
the `test` block: "the reported number must not come from the clients the model
was just fitted on".

The fix puts the split in the key. Measured after it, the overlap is what
independent draws give: 1/40 against a hypergeometric mean of 1.6, and
280/2000 against 290.5.

`IndependenceTest` asserts that as a distribution rather than as one number.
"the two sets differ" would pass for a key that perturbed the draw slightly
and left the two splits 95% shared, which is the failure worth catching and
the one a single-draw assertion cannot see.

A run that uses neither `sample:N` nor `resample:N` makes no draw for the split
to change, so the fix moves none of its numbers. `NoRunOnDiskIsAffectedTest`
checks the runs it finds under `outputs/` rather than trusting that, because it
is the whole argument for making this change at all.
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

import pytest
import yaml

from fedbrew.core.loop import _evaluation_client_selector, _sampled_client_infos
from fedbrew.core.protocol import ClientInfo

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")

#: The two shipped configs that sample two splits at one size. Written out so
#: that editing one of them moves this list rather than quietly emptying the
#: premise of every test below.
CONFIGS_THAT_SAMPLE_TWO_SPLITS = {
    "configs/openimage/fedavg.yaml": {"val": "sample:2000", "test": "sample:2000"},
    "configs/reference_evaluation.yaml": {"val": "sample:40", "test": "sample:40"},
}


def _roster(count: int) -> list[ClientInfo]:
    return [ClientInfo(client_id=f"c{index:05d}", num_examples=10) for index in range(count)]


def _drawn(
    clients: list[ClientInfo], scope: str, seed: int, split: str, round_id: int = 1
) -> set[str]:
    selector = _evaluation_client_selector(clients, scope, seed, split)
    return {info.client_id for info in selector(round_id, [])}


def _draw_under_the_old_key(
    clients: list[ClientInfo], size: int, seed: int, round_id: int | None
) -> set[str]:
    """The pre-fix draw, reconstructed: the key carried no split.

    `_sampled_client_infos` is otherwise unchanged, so this differs from it in
    exactly the one term the fix added.
    """

    key = f"evaluation_clients:{seed}:{size}:{round_id}"
    return set(random.Random(key).sample(sorted(info.client_id for info in clients), size))


class TheSplitIsInTheDrawTest(unittest.TestCase):
    def test_two_splits_at_the_same_size_draw_different_clients(self) -> None:
        for population, size in ((1000, 40), (13771, 2000)):
            with self.subTest(population=population, size=size):
                clients = _roster(population)
                val = _drawn(clients, f"sample:{size}", 42, "val")
                test = _drawn(clients, f"sample:{size}", 42, "test")
                self.assertNotEqual(val, test)

    def test_all_three_splits_differ_from_each_other(self) -> None:
        clients = _roster(1000)
        drawn = {split: _drawn(clients, "sample:40", 42, split) for split in SPLITS}
        for left in SPLITS:
            for right in SPLITS:
                if left < right:
                    with self.subTest(pair=(left, right)):
                        self.assertNotEqual(drawn[left], drawn[right])

    def test_resample_is_per_split_too(self) -> None:
        """The two modes share `_sampled_client_infos`; both had the defect."""

        clients = _roster(1000)
        for round_id in (1, 7, 30):
            with self.subTest(round_id=round_id):
                self.assertNotEqual(
                    _drawn(clients, "resample:40", 42, "val", round_id),
                    _drawn(clients, "resample:40", 42, "test", round_id),
                )


class IndependenceTest(unittest.TestCase):
    """Different is not enough: the draws have to be independent.

    A key that mixed the split in weakly would give two sets that differ by a
    handful of clients and pass every assertion above. The overlap between two
    independent draws of size n from a population of N is hypergeometric with
    mean n^2/N, so averaging it over many seeds is a check that fails for a
    correlated key and passes for an independent one.
    """

    def test_the_mean_overlap_is_the_hypergeometric_mean(self) -> None:
        population, size, trials = 1000, 100, 200
        clients = _roster(population)
        overlaps = [
            len(
                _drawn(clients, f"sample:{size}", seed, "val")
                & _drawn(clients, f"sample:{size}", seed, "test")
            )
            for seed in range(trials)
        ]
        expected = size * size / population
        observed = sum(overlaps) / trials
        # The standard error of the mean over `trials` draws; three of them is
        # a bound that a correlated key (mean at or near `size`) cannot meet
        # and an independent one clears with room.
        variance = expected * (1 - size / population) * ((population - size) / (population - 1))
        tolerance = 3 * (variance / trials) ** 0.5
        self.assertLess(
            abs(observed - expected),
            tolerance,
            f"mean overlap {observed:.2f} against hypergeometric mean {expected:.2f}",
        )

    def test_no_seed_produces_the_identical_set(self) -> None:
        clients = _roster(1000)
        for seed in range(50):
            with self.subTest(seed=seed):
                self.assertNotEqual(
                    _drawn(clients, "sample:40", seed, "val"),
                    _drawn(clients, "sample:40", seed, "test"),
                )


class TheOtherSplitsCoincideOnlyWhenTheyMustTest(unittest.TestCase):
    """The two cases where sharing a set is right, kept from the fix's reach.

    Every other property of the draw -- determinism, fixed-across-rounds,
    resample-varies-across-rounds, roster-order independence, roster-order
    output, the two unsampled modes -- is `tests/test_evaluation_client_scope.py`,
    which covers the four modes and holds the split fixed. Duplicating it here
    would give two modules that fail together and say the same thing.
    """

    def setUp(self) -> None:
        self.clients = _roster(1000)

    def test_a_size_at_or_above_the_population_returns_everyone(self) -> None:
        """No draw happens, so every split gets the whole roster. Correct."""

        for size in (1000, 5000):
            with self.subTest(size=size):
                self.assertEqual(
                    _drawn(self.clients, f"sample:{size}", 42, "val"),
                    _drawn(self.clients, f"sample:{size}", 42, "test"),
                )

    def test_the_unsampled_modes_do_not_vary_by_split(self) -> None:
        """`all` and `participating` name their clients; there is nothing to draw."""

        for scope in ("all", "participating"):
            with self.subTest(scope=scope):
                selected = ["c00000", "c00001"]
                drawn = {
                    split: [
                        info.client_id
                        for info in _evaluation_client_selector(self.clients, scope, 42, split)(
                            1, selected
                        )
                    ]
                    for split in SPLITS
                }
                self.assertEqual(len({tuple(value) for value in drawn.values()}), 1)

    def test_the_split_reaches_the_sampler_and_is_the_only_change(self) -> None:
        """Same arguments but the split, on the function the selector calls."""

        drawn = {
            split: {
                info.client_id for info in _sampled_client_infos(self.clients, 40, 42, None, split)
            }
            for split in SPLITS
        }
        self.assertEqual(len({frozenset(value) for value in drawn.values()}), 3)


class TheShippedConfigsTest(unittest.TestCase):
    """The premise: without a config in this shape the fix guards nothing."""

    def test_the_two_configs_still_sample_two_splits_at_one_size(self) -> None:
        for relative, expected in CONFIGS_THAT_SAMPLE_TWO_SPLITS.items():
            with self.subTest(config=relative):
                config = yaml.safe_load((REPO_ROOT / relative).read_text(encoding="utf-8"))
                evaluation = config["evaluation"]
                self.assertEqual(
                    {split: evaluation[split]["clients"] for split in expected}, expected
                )

    def test_they_would_have_drawn_identically_before_the_fix(self) -> None:
        """The old key rebuilt, run on the shapes those two configs describe.

        Not a restatement of the fix: the old key is reconstructed here and
        drawn from directly, so the table in this module's docstring is a
        measurement the test repeats rather than a claim about history. The
        contrast beside it is the shipped function.
        """

        for relative, expected in CONFIGS_THAT_SAMPLE_TWO_SPLITS.items():
            size = int(next(iter(expected.values())).split(":")[1])
            population = size * 5
            with self.subTest(config=relative):
                clients = _roster(population)
                before = {
                    split: _draw_under_the_old_key(clients, size, 42, None)
                    for split in ("val", "test")
                }
                self.assertEqual(
                    before["val"], before["test"], "the old key was not split-blind after all"
                )
                self.assertEqual(len(before["val"] & before["test"]), size)

                after = {
                    split: _drawn(clients, f"sample:{size}", 42, split) for split in ("val", "test")
                }
                self.assertNotEqual(after["val"], after["test"])
                self.assertLess(len(after["val"] & after["test"]), size)


class NoRunOnDiskIsAffectedTest(unittest.TestCase):
    """The argument for changing a draw at all, checked rather than trusted.

    It reads the runs actually on disk, so it depends on local state by design:
    any run under the gitignored ``outputs/`` whose config samples a split --
    ``sample:N`` or ``resample:N`` on any evaluation block -- fails it locally,
    and it cannot tell a run written before the fix from one written after. A
    fresh clone and CI have no such run and pass. When this is a gate's only
    failure, the assertion names the runs: check that they are local runs with
    a sampled scope before reading it as a regression. Deleting them to make it
    pass would defeat the guard.
    """

    def test_no_recorded_run_used_a_sampled_client_scope(self) -> None:
        sampled = {}
        for path in sorted((REPO_ROOT / "outputs").rglob("run.json")):
            evaluation = (
                json.loads(path.read_text(encoding="utf-8")).get("config", {}).get("evaluation", {})
            )
            hits = {
                split: block.get("clients")
                for split, block in (evaluation or {}).items()
                if isinstance(block, dict)
                and str(block.get("clients", "")).startswith(("sample:", "resample:"))
            }
            if hits:
                sampled[str(path.relative_to(REPO_ROOT))] = hits
        self.assertEqual(sampled, {}, "a recorded run's client draw moved with this fix")


if __name__ == "__main__":
    unittest.main()
