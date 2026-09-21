"""generic_sft must decide client eligibility on packed windows, not rows.

It counted rows: a client with a non-empty train list and a non-empty
client_eval list was eligible. pack_sft_examples drops any window that is not
full, so a client whose eval rows total fewer than sequence_length + 1 tokens
packs to zero eval windows -- and train_fraction, applied after selection, can
do the same to its train side. The record was written anyway, with
num_eval_examples: 0 and nothing on stderr. oasst1_sft.py:235 already decides
this after packing.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.data.generic_sft import FieldMapping, PreparedRow, _build_shards
from fedbrew.data.oasst1_sft import IGNORE_INDEX, TokenizedSFTExample

pytestmark = pytest.mark.fast

#: Short enough that a client's rows can plausibly fall under one window.
SEQUENCE_LENGTH = 8
EOS = 0


def _example(num_tokens: int) -> TokenizedSFTExample:
    token_ids = tuple(range(1, num_tokens + 1))
    return TokenizedSFTExample(
        input_ids=torch.tensor(token_ids[:-1] or [1], dtype=torch.long),
        labels=torch.tensor(token_ids[1:] or [1], dtype=torch.long),
        token_ids=token_ids,
        active_token_mask=tuple(True for _ in token_ids),
        prompt_length=0,
        target_token_count=num_tokens,
    )


def _rows(clients: dict[str, tuple[int, int]], test_tokens: int = 200):
    """One PreparedRow per split per client, plus the global test split."""

    prepared = [PreparedRow(client_key="", split="global_test", example=_example(test_tokens))]
    for client, (train_tokens, eval_tokens) in clients.items():
        prepared.append(
            PreparedRow(client_key=client, split="train", example=_example(train_tokens))
        )
        prepared.append(
            PreparedRow(client_key=client, split="client_eval", example=_example(eval_tokens))
        )
    return prepared


def _many_train_rows(client: str, count: int, tokens_each: int):
    """A client whose train side is many small rows, so train_fraction bites."""

    return [
        PreparedRow(client_key=client, split="train", example=_example(tokens_each))
        for _ in range(count)
    ]


def _mapping() -> FieldMapping:
    return FieldMapping(
        prompt_template="{q}",
        response_template="{a}",
        client_field="c",
        group_field="c",
        required_fields=("q", "a", "c"),
        choice=None,
        anonymize_client_ids=False,
        system_prompt=None,
    )


def _build(prepared, num_clients: int, **caps):
    return _build_shards(
        prepared,
        seed=0,
        mapping=_mapping(),
        num_clients=num_clients,
        sequence_length=SEQUENCE_LENGTH,
        eos_token_id=EOS,
        ignore_index=IGNORE_INDEX,
        caps=caps,
    )


class EligibilityAfterPackingTests(unittest.TestCase):
    def test_a_client_that_packs_to_no_eval_windows_is_not_selected(self) -> None:
        # "thin" has eval rows, but not sequence_length + 1 tokens of them, so
        # every eval window is incomplete and dropped.
        prepared = _rows({"big": (400, 400), "thin": (400, 3), "other": (400, 400)})
        selected, windows, _ = _build(prepared, num_clients=2)
        self.assertNotIn("thin", selected)
        self.assertEqual(sorted(selected), ["big", "other"])
        for client in selected:
            with self.subTest(client=client):
                self.assertGreater(windows[client]["eval"][1].shape[0], 0)
                self.assertGreater(windows[client]["train"][1].shape[0], 0)

    def test_train_fraction_cannot_leave_a_selected_client_with_no_windows(
        self,
    ) -> None:
        # "cut" ranks first on train tokens, so it is selected either way, and
        # train_fraction -- applied after that ranking -- then reduces its 200
        # one-token rows to four, under the nine tokens one window needs.
        prepared = _rows({"a": (150, 400), "b": (150, 400)})
        prepared += _many_train_rows("cut", count=200, tokens_each=1)
        prepared.append(PreparedRow(client_key="cut", split="client_eval", example=_example(400)))

        without_fraction, _, _ = _build(prepared, num_clients=2)
        self.assertIn("cut", without_fraction)

        selected, windows, _ = _build(prepared, num_clients=2, train_fraction=0.02)
        self.assertNotIn("cut", selected)
        for client in selected:
            with self.subTest(client=client):
                self.assertGreater(windows[client]["train"][1].shape[0], 0)

    def test_too_few_eligible_clients_is_an_error_that_says_why(self) -> None:
        prepared = _rows({"big": (400, 400), "thin": (400, 3)})
        with self.assertRaises(ValueError) as caught:
            _build(prepared, num_clients=2)
        message = str(caught.exception)
        self.assertIn("only 1 clients", message)
        # Name the client, the split, and the threshold it fell under.
        self.assertIn("thin", message)
        self.assertIn("eval", message)
        self.assertIn(str(SEQUENCE_LENGTH + 1), message)

    def test_every_selected_client_has_windows_in_both_splits(self) -> None:
        prepared = _rows({f"c{i}": (400, 400) for i in range(5)})
        selected, windows, _ = _build(prepared, num_clients=3)
        self.assertEqual(len(selected), 3)
        for client in selected:
            with self.subTest(client=client):
                for split in ("train", "eval"):
                    self.assertGreater(windows[client][split][1].shape[0], 0)


class SelectionOrderTests(unittest.TestCase):
    def test_clients_are_still_ranked_by_train_tokens(self) -> None:
        prepared = _rows({"small": (200, 400), "large": (900, 400), "mid": (500, 400)})
        selected, _, _ = _build(prepared, num_clients=2)
        self.assertEqual(selected, ["large", "mid"])

    def test_the_ranking_is_taken_before_train_fraction(self) -> None:
        # train_fraction scales every client by the same factor, so which
        # clients are picked must not depend on it. A config that drops no
        # client selects exactly what it did before this change.
        prepared = _rows({"small": (200, 400), "large": (900, 400), "mid": (500, 400)})
        full, _, _ = _build(prepared, num_clients=2)
        scaled, _, _ = _build(
            _rows({"small": (200, 400), "large": (900, 400), "mid": (500, 400)}),
            num_clients=2,
            train_fraction=0.5,
        )
        self.assertEqual(full, scaled)


if __name__ == "__main__":
    unittest.main()
