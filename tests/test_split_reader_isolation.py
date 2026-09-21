"""No training or validation path may resolve to the test slice.

The FEMNIST split leak was fixed on the writing side: FEMNIST now cuts three
disjoint slices per writer and only the third reaches global_test.pt
(tests/test_femnist_support.py, tests/test_femnist_writer_split.py). Writing
them apart is half of it. The other half is that nothing on the reading side
puts them back together -- a resolver that falls back from a missing "eval" to
"test", or a train path that takes the whole client mapping when "train" is
absent, would restore exactly the leak the split was carved to remove, with the
data on disk still perfectly correct.

There are three resolvers and they have different contracts, which is why they
each need their own check:

  _get_train_data        what fit() trains on            -> train, only ever
  _get_eval_data         the split-less evaluate() path  -> val, else refuse
  _get_evaluation_split  the loop's train/val/test eval  -> exactly that split

`_get_eval_data` used to read "-> eval, else train", and this module recorded
that as its contract. It was the reading-side leak the paragraph above says
nothing does: a client too small to hold out a val slice scored its own
training data, and the caller got those numbers under its evaluation metric
names with nothing in the count or the CSV to say which slice they came from.
Two documents disagreeing in writing, which is worse than one being silent. It
now delegates to `_get_evaluation_split(.., "val")` -- one resolver for the val
slice, not two that agree most of the time -- and refuses when there is none.
FINDINGS.csv P07-F08.

"val" is the config and metric name; "eval" is the on-disk shard key
(torch_sgd_client.py:643-645). That rename is the seam where a mapping mistake
would be easiest to make and hardest to see, since both names are legitimate
elsewhere in the codebase.

Each slice is tagged with a distinct value, so a resolver returning the wrong
one is identified by name rather than merely detected.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.clients.torch_sgd_client import (
    _get_eval_data,
    _get_evaluation_split,
    _get_train_data,
)

pytestmark = pytest.mark.fast

TRAIN, EVAL, TEST = 1.0, 2.0, 3.0
TAGS = {TRAIN: "train", EVAL: "eval", TEST: "test"}


def _split(tag: float, size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "x": torch.full((size, 2), tag),
        "y": torch.zeros(size, dtype=torch.long),
    }


def _client_data(*, with_test: bool = True) -> dict[str, object]:
    data: dict[str, object] = {"train": _split(TRAIN), "eval": _split(EVAL)}
    if with_test:
        data["test"] = _split(TEST)
    return data


def _which(resolved: object) -> str:
    """Name the slice a resolver returned, from the tag baked into it."""

    assert isinstance(resolved, dict), resolved
    tags = {float(value) for value in resolved["x"].flatten().tolist()}
    assert len(tags) == 1, f"resolver returned a mixture of slices: {tags}"
    return TAGS[tags.pop()]


class TrainPathTests(unittest.TestCase):
    def test_the_train_path_returns_the_train_slice(self) -> None:
        self.assertEqual(_which(_get_train_data(_client_data())), "train")

    def test_the_train_path_ignores_a_test_slice_entirely(self) -> None:
        with_test = _get_train_data(_client_data(with_test=True))
        without = _get_train_data(_client_data(with_test=False))
        # Adding a test slice must not change what training sees, at all.
        self.assertTrue(torch.equal(with_test["x"], without["x"]))

    def test_a_client_with_only_a_test_slice_trains_on_nothing_resolvable(self) -> None:
        """The fallback must not be "whatever is left".

        _get_train_data falls back to the whole mapping when there is no
        "train" key, which is how the unsplit single-tensor shard format is
        supported. That fallback must return the mapping itself -- something a
        dataloader will refuse -- rather than reaching into "test".
        """

        resolved = _get_train_data({"test": _split(TEST)})
        self.assertNotIn("x", resolved)
        self.assertEqual(set(resolved), {"test"})


class EvalPathTests(unittest.TestCase):
    def test_the_split_less_eval_path_returns_the_eval_slice(self) -> None:
        self.assertEqual(_which(_get_eval_data(_client_data())), "eval")

    def test_it_refuses_rather_than_falling_back_to_train(self) -> None:
        """The finding, as the one case that produced it.

        A client with train and test rows but no val slice. This returned the
        train slice and the caller reported it as an evaluation metric.
        """

        data = {"train": _split(TRAIN), "test": _split(TEST)}
        with self.assertRaises(ValueError) as caught:
            _get_eval_data(data, "c0")
        self.assertIn("no non-empty val split", str(caught.exception))

    def test_an_empty_eval_slice_is_refused_too(self) -> None:
        """Present but empty is the same as absent, and was the same fallback."""

        data = {
            "train": _split(TRAIN),
            "eval": _split(EVAL, size=0),
            "test": _split(TEST),
        }
        with self.assertRaises(ValueError):
            _get_eval_data(data, "c0")

    def test_it_never_reaches_the_test_slice(self) -> None:
        """The property this module exists for, restated for the new contract.

        The old fallback never reached test either -- it reached train. The
        refusal must not have quietly widened the search on the way past.
        """

        for data in (
            {"train": _split(TRAIN), "test": _split(TEST)},
            {"test": _split(TEST)},
            {"train": _split(TRAIN), "eval": _split(EVAL, size=0), "test": _split(TEST)},
        ):
            with self.subTest(keys=sorted(data)):
                with self.assertRaises(ValueError):
                    _get_eval_data(data, "c0")

    def test_it_reads_the_same_slice_the_loop_would(self) -> None:
        """One resolver, checked against the other rather than asserted equal."""

        data = _client_data()
        self.assertIs(_get_eval_data(data), _get_evaluation_split(data, "val"))

    def test_the_message_names_the_way_out(self) -> None:
        """A client without a val slice can still be evaluated on the rest."""

        with self.assertRaises(ValueError) as caught:
            _get_eval_data({"train": _split(TRAIN), "test": _split(TEST)}, "c0")
        message = str(caught.exception)
        self.assertIn("c0", message)
        self.assertIn('payload["splits"]', message)


class EvaluationSplitTests(unittest.TestCase):
    """The loop's resolver: each name must reach its own slice and no other."""

    def test_each_split_name_resolves_to_its_own_slice(self) -> None:
        data = _client_data()
        for requested, expected in (("train", "train"), ("val", "eval"), ("test", "test")):
            with self.subTest(split=requested):
                self.assertEqual(_which(_get_evaluation_split(data, requested)), expected)

    def test_val_does_not_fall_through_to_test_when_eval_is_missing(self) -> None:
        """The leak this whole file exists to prevent.

        A client with no eval slice must report no validation metric, not
        quietly report its test metric under the val_* name -- which is
        precisely the shape of that FEMNIST leak, one layer up.
        """

        data = {"train": _split(TRAIN), "test": _split(TEST)}
        self.assertIsNone(_get_evaluation_split(data, "val"))

    def test_an_empty_eval_slice_does_not_fall_through_to_test(self) -> None:
        data = {
            "train": _split(TRAIN),
            "eval": _split(EVAL, size=0),
            "test": _split(TEST),
        }
        self.assertIsNone(_get_evaluation_split(data, "val"))

    def test_test_does_not_fall_through_to_eval_when_test_is_missing(self) -> None:
        """The same leak in the other direction.

        Pre-fix FEMNIST data has no per-client test slice at all. Resolving
        "test" to the eval slice would reproduce the identical central_test_*
        and val_* columns the fix removed, on data that no longer has that
        defect written into it.
        """

        data = {"train": _split(TRAIN), "eval": _split(EVAL)}
        self.assertIsNone(_get_evaluation_split(data, "test"))

    def test_train_does_not_fall_through_to_any_holdout(self) -> None:
        data = {"eval": _split(EVAL), "test": _split(TEST)}
        self.assertIsNone(_get_evaluation_split(data, "train"))

    def test_the_three_splits_are_never_the_same_object(self) -> None:
        # Aliasing would make every disjointness check downstream vacuous.
        data = _client_data()
        resolved = {name: _get_evaluation_split(data, name) for name in ("train", "val", "test")}
        identities = {id(value) for value in resolved.values()}
        self.assertEqual(len(identities), 3)


if __name__ == "__main__":
    unittest.main()
