"""Overlapping windows and a per-client eval split cannot both be asked for.

`build_next_token_examples` starts a window every `causal_lm.stride` tokens, so
`stride < sequence_length` makes consecutive windows share tokens. Those windows
are then dealt to clients by index and cut into train and eval by a shuffled
index, and neither step knows two windows overlap -- so a client's eval window
can be tokens it also trained on, and its loss is optimistic by an amount no
metric reports. The shipped dev config has `stride == sequence_length`.

The measurement this guard replaces, on 4,000 distinct tokens with
`sequence_length` 16, four clients and `eval_ratio` 0.2 -- how many of the four
clients had at least one eval window overlapping their own train windows, and
the worst single eval window:

| stride | windows | clients leaking | worst eval window |
| --- | --- | --- | --- |
| 16 | 249 | 0 | 0 of 16 tokens |
| 12 | 332 | 4 | 8 of 16 |
| 8 | 498 | 4 | 16 of 16 |
| 4 | 996 | 4 | 16 of 16 |
| 1 | 3,984 | 4 | 16 of 16 |

The last three rows are the point: against the *union* of a client's train
windows the leak is not bounded by `sequence_length - stride`, which is one
pair's overlap. At stride 8 the two train windows either side of an eval window
cover all of it, so the eval split is entirely seen data.

`_require_disjoint_client_windows` refuses that combination. The overlap itself
stays available at `eval_ratio: 0`, where there is no client split to leak
across -- the source test split is cut at the record level before tokenization.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.data.hf_causal_lm_text import (
    _require_disjoint_client_windows,
    _split_client_examples,
    build_next_token_examples,
)
from fedbrew.data.partitioners.iid import partition_iid

pytestmark = pytest.mark.fast

SEQUENCE_LENGTH = 16


def _worst_client_overlap(stride: int, eval_ratio: float = 0.2) -> tuple[int, int]:
    """(clients with a leaking eval window, worst overlap in tokens).

    Every token id is distinct, so a token appearing in two windows is a real
    overlap rather than a repeated word.
    """

    inputs, targets = build_next_token_examples(
        torch.arange(4000, dtype=torch.long),
        sequence_length=SEQUENCE_LENGTH,
        stride=stride,
    )
    partitions = partition_iid(list(range(len(targets))), 4, seed=42)
    leaking = 0
    worst = 0
    for position, client_id in enumerate(sorted(partitions)):
        train_indices, eval_indices = _split_client_examples(
            partitions[client_id],
            eval_ratio=eval_ratio,
            seed=42 + position + 1009,
        )
        trained = {int(token) for index in train_indices for token in inputs[index]}
        shared = [len(trained & {int(token) for token in inputs[index]}) for index in eval_indices]
        if any(shared):
            leaking += 1
            worst = max(worst, max(shared))
    return leaking, worst


class TheLeakIsRealTests(unittest.TestCase):
    """What the refusal is for, measured rather than asserted."""

    def test_a_full_stride_shares_nothing(self) -> None:
        self.assertEqual(_worst_client_overlap(stride=SEQUENCE_LENGTH), (0, 0))

    def test_a_half_stride_puts_a_whole_eval_window_in_the_train_split(self) -> None:
        leaking, worst = _worst_client_overlap(stride=SEQUENCE_LENGTH // 2)
        self.assertEqual(leaking, 4)
        self.assertEqual(worst, SEQUENCE_LENGTH)


class TheRefusalTests(unittest.TestCase):
    def test_an_overlapping_stride_with_a_client_eval_split_is_refused(self) -> None:
        for stride in (1, 4, 8, 12, SEQUENCE_LENGTH - 1):
            with self.subTest(stride=stride):
                with self.assertRaises(ValueError) as caught:
                    _require_disjoint_client_windows(
                        sequence_length=SEQUENCE_LENGTH,
                        stride=stride,
                        eval_ratio=0.2,
                    )
                message = str(caught.exception)
                self.assertIn(str(stride), message)
                self.assertIn(str(SEQUENCE_LENGTH), message)
                # Both ways out, named.
                self.assertIn("causal_lm.stride", message)
                self.assertIn("eval_ratio", message)

    def test_a_stride_at_or_above_the_sequence_length_is_allowed(self) -> None:
        for stride in (SEQUENCE_LENGTH, SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH * 4):
            with self.subTest(stride=stride):
                _require_disjoint_client_windows(
                    sequence_length=SEQUENCE_LENGTH,
                    stride=stride,
                    eval_ratio=0.2,
                )

    def test_overlap_is_allowed_when_no_client_eval_split_is_cut(self) -> None:
        """There is nothing for a window to leak across: the source test split
        is cut at the record level, before tokenization."""

        for stride in (1, 8, SEQUENCE_LENGTH - 1):
            with self.subTest(stride=stride):
                _require_disjoint_client_windows(
                    sequence_length=SEQUENCE_LENGTH,
                    stride=stride,
                    eval_ratio=0.0,
                )


if __name__ == "__main__":
    unittest.main()
