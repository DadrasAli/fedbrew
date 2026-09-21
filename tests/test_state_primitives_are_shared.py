"""The two state primitives have one definition, and every caller uses it.

`servers/fedopt.py` carried its own `_validate_matching_keys` and
`_as_cpu_tensor`, copied from `core/torch_utils.py` and differing only in that
the copy took exactly two states where the original takes any number. Nothing
was wrong with either. What was wrong is that there were two: `as_cpu_tensor`
is the single door every elementwise state operation reads a value through, so
it is where a dtype policy would go -- and a dtype policy added to the original
would have left FedOpt's `_square_model_state`, `_sign_model_state` and
`_multiply_model_states` reading through the copy, unchanged and silent.

That is not hypothetical here. `P02-F03` in `FINDINGS.csv` is a dtype defect
whose fix belongs in exactly that function, and it is still open.

So the checks are the two halves of "one definition": the name resolves to the
same object in both modules, and no third module has quietly grown its own.
The behavioural pair underneath them exists because identity alone would still
pass if `fedopt` imported the shared function and then wrapped it in a local
one that swallowed the error.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.core import torch_utils
from fedbrew.servers import fedopt

pytestmark = pytest.mark.fast

PACKAGE = Path(__file__).resolve().parent.parent / "fedbrew"

#: The primitives, and the module that is allowed to define them.
SHARED = ("as_cpu_tensor", "validate_matching_keys")
HOME = PACKAGE / "core" / "torch_utils.py"


def _definitions(name: str) -> list[Path]:
    """Every module defining a top-level function called `name`, or `_name`."""

    found: list[Path] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (name, f"_{name}"):
                found.append(path)
    return found


class OneDefinitionTest(unittest.TestCase):
    def test_only_torch_utils_defines_them(self) -> None:
        for name in SHARED:
            with self.subTest(primitive=name):
                self.assertEqual(
                    _definitions(name),
                    [HOME],
                    f"{name} is defined outside core/torch_utils.py. A second "
                    "copy is a second place a policy change has to reach, and "
                    "the one that is not updated fails silently.",
                )

    def test_fedopt_uses_the_shared_object_and_not_a_lookalike(self) -> None:
        self.assertIs(fedopt.as_cpu_tensor, torch_utils.as_cpu_tensor)
        self.assertIs(fedopt.validate_matching_keys, torch_utils.validate_matching_keys)


class SharedBehaviourReachesFedOptTest(unittest.TestCase):
    """Identity is the mechanism; these are what it buys, at the call sites."""

    def test_a_non_tensor_is_refused_with_the_shared_message(self) -> None:
        state = {"w": torch.ones(2)}
        for operation in (
            lambda: fedopt._square_model_state({"w": [1.0, 2.0]}),
            lambda: fedopt._sign_model_state({"w": [1.0, 2.0]}),
            lambda: fedopt._multiply_model_states({"w": [1.0, 2.0]}, state),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(TypeError, "state value for w is not a tensor"):
                    operation()

    def test_mismatched_keys_are_refused_with_the_shared_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "all states must have the same keys"):
            fedopt._multiply_model_states({"w": torch.ones(2)}, {"b": torch.ones(2)})

    def test_the_shared_arity_is_the_one_fedopt_now_has(self) -> None:
        """The copy took two states; the original takes any number.

        Pinned because it is the only behavioural difference the copy had, so
        it is the one thing this change could have altered by accident.
        """

        torch_utils.validate_matching_keys({"w": torch.ones(2)})
        torch_utils.validate_matching_keys(
            {"w": torch.ones(2)}, {"w": torch.ones(2)}, {"w": torch.ones(2)}
        )
        with self.assertRaisesRegex(ValueError, "states must not be empty"):
            torch_utils.validate_matching_keys()


class FedOptStillComputesTheSameThingTest(unittest.TestCase):
    """The three helpers that read through the primitive, on known input."""

    def test_square_sign_and_multiply(self) -> None:
        state = {"w": torch.tensor([-2.0, 3.0])}
        self.assertTrue(
            torch.equal(fedopt._square_model_state(state)["w"], torch.tensor([4.0, 9.0]))
        )
        self.assertTrue(
            torch.equal(fedopt._sign_model_state(state)["w"], torch.tensor([-1.0, 1.0]))
        )
        other = {"w": torch.tensor([5.0, 7.0])}
        self.assertTrue(
            torch.equal(
                fedopt._multiply_model_states(state, other)["w"], torch.tensor([-10.0, 21.0])
            )
        )


if __name__ == "__main__":
    unittest.main()
