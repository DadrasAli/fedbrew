"""A checkpoint's client states are written stacked, and either format resumes the same run.

Format 1 held ``client_states`` as one dict per client, and ``torch.save``
pickled every value of every one of them; format 2 holds one column per key
over the clients -- a setting once, the clients' numbers or tensors as one
tensor each (``stack_client_states``, ``fedbrew/core/checkpointing.py``).
``load_checkpoint`` reads both and gives format 1's layout. What is pinned
here:

- stacking and unstacking gives back every state exactly: the same keys in
  the same order, the same types, floats bit for bit (NaN, -0.0, inf),
  integers past int64, tensors of the same dtype, shape and bits, each in
  storage of its own, and a client whose state has other keys kept whole;
- where the clients' structure matches, a key is one column: a setting
  stored once, the clients' numbers and control variates one tensor each;
- a format-1 file is read as it is, a format this code does not know is
  refused, and so is a format-2 file whose client states are not stacked;
- a run resumed from a format-2 ``latest.pt`` and the same run resumed from
  that checkpoint rewritten in format 1 are bit-identical to each other and
  to the uninterrupted run, for FedAvg and for SCAFFOLD, whose client states
  carry a control variate per parameter.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from fedbrew.core.checkpointing import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    stack_client_states,
    unstack_client_states,
)
from fedbrew.core.refusal import RunRefused
from tests.test_checkpoint_size_is_constant_in_rounds import _run, _write
from tests.test_reproducibility import _digest


def same(left: Any, right: Any) -> bool:
    """Equal values of the same types, key order included; floats and tensors bit for bit."""

    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(_bits(left), _bits(right))
        )
    if isinstance(left, float):
        return math.copysign(1.0, left) == math.copysign(1.0, right) and (
            left == right or (math.isnan(left) and math.isnan(right))
        )
    if isinstance(left, dict):
        return list(left) == list(right) and all(same(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    if hasattr(left, "tobytes") and hasattr(left, "dtype"):
        # An array of the RNG state: numpy's, compared by its bytes.
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and (left.tobytes() == right.tobytes())
        )
    return bool(left == right)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().contiguous().reshape(-1)
    return flat.view(torch.uint8) if flat.numel() else flat


def _states(count: int) -> dict[str, dict[str, Any]]:
    """Client states of every kind a column can hold, and one that fits none."""

    states: dict[str, dict[str, Any]] = {}
    for index in range(count):
        client_id = f"client_{index}"
        states[client_id] = {
            "client_id": client_id,
            "num_examples": 10 + index % 7,
            "learning_rate": 0.05,
            "measured": [float("nan"), -0.0, math.inf, -math.inf, 1 / 3][index % 5],
            "maybe": None if index % 2 else 1.5,
            "nesterov": bool(index % 3),
            "base_seed": 42,
            "huge": 2**70 + index,
            "metrics": ["fit_loss", "fit_accuracy"],
            "labels": [f"l{index}"],
            "update_mode": "sequential_epoch",
            "max_local_steps": None,
            "client_control": {
                "fc.weight": torch.randn(3, 2, dtype=torch.float32),
                "fc.bias": torch.randn(2, dtype=torch.float64),
            },
            "empty": {},
            "ragged": torch.zeros(index % 2 + 1),
        }
    states["client_odd"] = {"client_id": "client_odd", "extra": 1}
    return states


class RoundTripTest(unittest.TestCase):
    def test_every_state_comes_back_exactly(self) -> None:
        states = _states(12)
        back = unstack_client_states(stack_client_states(states))
        self.assertTrue(same(back, states))
        self.assertEqual(list(back), list(states))

    def test_each_restored_tensor_has_storage_of_its_own(self) -> None:
        back = unstack_client_states(stack_client_states(_states(4)))
        pointers = {
            state["client_control"]["fc.weight"].untyped_storage().data_ptr()
            for state in back.values()
            if "client_control" in state
        }
        self.assertEqual(len(pointers), 4)
        back["client_0"]["metrics"].append("changed")
        self.assertEqual(back["client_1"]["metrics"], ["fit_loss", "fit_accuracy"])

    def test_matching_structure_is_one_column_per_key(self) -> None:
        stacked = stack_client_states(_states(12))
        kinds = {key: next(iter(column)) for key, column in stacked["columns"].items()}
        self.assertEqual(
            kinds,
            {
                "client_id": "client_id",
                "num_examples": "int",
                "learning_rate": "float",
                "measured": "float",
                "maybe": "values",
                "nesterov": "bool",
                "base_seed": "shared",
                "huge": "values",
                "metrics": "shared",
                "labels": "values",
                "update_mode": "shared",
                "max_local_steps": "shared",
                "client_control": "mapping",
                "empty": "mapping",
                "ragged": "values",
            },
        )
        control = stacked["columns"]["client_control"]["mapping"]
        self.assertEqual(control["fc.weight"]["tensor"].shape, (12, 3, 2))
        self.assertEqual(control["fc.bias"]["tensor"].dtype, torch.float64)
        self.assertEqual(list(stacked["separate"]), ["client_odd"])

    def test_no_clients_and_one(self) -> None:
        for states in ({}, {"only": {"client_id": "only", "x": 1.0}}, {"bare": {}}):
            with self.subTest(states=list(states)):
                self.assertTrue(same(unstack_client_states(stack_client_states(states)), states))


class FormatsAreReadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "latest.pt"

    def _load(self, payload: dict[str, Any]) -> dict[str, Any]:
        torch.save(payload, self.path)
        return load_checkpoint(self.path)

    def test_format_one_is_read_as_it_is(self) -> None:
        states = _states(3)
        loaded = self._load({"round_id": 2, "client_states": states})
        self.assertTrue(same(loaded, {"round_id": 2, "client_states": states}))

    def test_format_two_is_read_as_format_one(self) -> None:
        states = _states(3)
        loaded = self._load(
            {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "round_id": 2,
                "client_states": stack_client_states(states),
            }
        )
        self.assertTrue(same(loaded, {"round_id": 2, "client_states": states}))

    def test_an_unknown_format_is_refused(self) -> None:
        for version, words in (
            (CHECKPOINT_FORMAT + 1, "reads formats 1 to"),
            (0, "positive integer"),
            ("2", "positive integer"),
        ):
            with self.subTest(version=version), self.assertRaisesRegex(RunRefused, words):
                self._load({"checkpoint_format": version, "round_id": 1})

    def test_format_two_client_states_not_stacked_are_refused(self) -> None:
        with self.assertRaisesRegex(RunRefused, "stacked layout"):
            self._load({"checkpoint_format": 2, "round_id": 1, "client_states": {"client_0": {}}})

    def test_client_states_that_are_no_mapping_are_left_for_the_resume_to_refuse(self) -> None:
        loaded = self._load({"checkpoint_format": 2, "round_id": 1, "client_states": [1]})
        self.assertEqual(loaded["client_states"], [1])


class BothFormatsResumeTheSameRunTest(unittest.TestCase):
    """Three rounds, then three more from latest.pt in each format, against six in one go."""

    def test_each_family(self) -> None:
        for family in ("fedavg", "scaffold"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                whole = _run(_write(root, "whole", family, 6))
                partial = _run(_write(root, "partial", family, 3))
                latest = partial / "checkpoints" / "latest.pt"
                raw = torch.load(latest, weights_only=False)
                self.assertEqual(raw["checkpoint_format"], CHECKPOINT_FORMAT)
                self.assertEqual(set(raw["client_states"]), {"clients", "columns", "separate"})

                shutil.copytree(partial, root / "old")
                # Format 1: what load_checkpoint reads, saved as it is.
                torch.save(load_checkpoint(latest), root / "old" / "checkpoints" / "latest.pt")
                resumed = {
                    name: _run(_write(root, name, family, 6), resume_latest=True)
                    for name in ("partial", "old")
                }

                finals = [
                    load_checkpoint(path / "checkpoints" / "latest.pt")
                    for path in (whole, *resumed.values())
                ]
                self.assertTrue(same(finals[1], finals[2]), "the two resumes")
                self.assertTrue(same(finals[1], finals[0]), "a resume and the whole run")
                self.assertEqual(finals[0]["round_id"], 6)
                self.assertEqual(
                    {_digest(path) for path in (whole, *resumed.values())}, {_digest(whole)}
                )


if __name__ == "__main__":
    unittest.main()
