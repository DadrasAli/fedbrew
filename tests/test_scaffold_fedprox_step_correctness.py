"""What SCAFFOLD's gradient correction and FedProx's proximal term compute.

Nothing before this measured it. test_scaffold_fedprox_communication_cost.py's
fixture zeroes the control variate specifically to avoid exercising the
correction at all, and no other test drives either algorithm's local step and
checks a number. It was written as the "before" for a refactor -- from a
hand-rolled forward/loss/backward/step that reached into
``task._move_batch``/``task._criterion`` to a correcting optimizer wrapped
around ``task.train_step``. That refactor is done: both clients step through
``_ScaffoldCorrectingOptimizer`` and ``_FedProxCorrectingOptimizer``, and when
it landed the trajectory pinned below matched it bit for bit.

Two independent kinds of evidence, not one:

``...ClosedFormPredictionTest`` derives what the correction should produce
from the algorithm itself -- SCAFFOLD's correction is exactly
``-lr * (server_control - client_control)`` added to a single step, no matter
what the loss gradient is; FedProx's proximal term is exactly zero at the
reference point and exactly ``-lr * mu * (w1 - w0)`` one step later, where
``w1`` is measured independently of the client under test. Neither prediction
is copied from the implementation being checked, so this catches a wrong
formula, not just a changed one. It holds before and after any correct
refactor, by construction.

``TrajectoryIsPinnedTest`` holds one run's floating-point output. It is a
regression pin, not a byte-for-byte proof: it fails when what either client
computes changes, and it tolerates the few float32 steps by which the same
computation lands differently on another CPU -- see ``PIN_TOLERANCE``. A pin
alone would not catch a formula that was wrong from the start; the closed-form
tests above are what rule that out.
"""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from fedbrew.clients.torch_fedprox_client import TorchFedProxClient
from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.core.protocol import FitRequest
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

_LEARNING_RATE = 0.1
_PROXIMAL_MU = 0.5

#: One float32 step for a magnitude in [0.5, 1), 2**-24 = 5.96e-08. Every value
#: pinned below has magnitude under 1, so none has a coarser step than this.
FLOAT32_STEP_BELOW_ONE = torch.finfo(torch.float32).eps / 2

#: How far a pinned value may sit from its pin, in float32 steps.
#:
#: The pins were first compared at atol 1e-9, which for 8 of the 10 pinned
#: weights and biases is less than one float32 step: bit equality under another
#: name. The same computation does not land on the same bits on every CPU,
#: because torch picks its CPU kernels by instruction set. On 2026-09-13 a CI
#: job missed the SCAFFOLD pin by 1.4e-9 while the `tests` job matched it
#: exactly. Measured that day on torch 2.13.0+cpu and 2.14.0+cu130, over every
#: pinned value, the loss metrics included:
#:
#:   torch release, thread count (1, 4, 64), MKL instruction overrides   no move
#:   torch's non-vectorized kernels (ATEN_CPU_CAPABILITY=default)       5.96e-08, one step
#:   margin for instruction sets this machine lacks (AVX-512)           4x
#:                                                                     -> 4 steps = 2.38e-07
#:
#: The smallest real change measured moves a pinned value 23x further than
#: that: proximal_mu from 0.5 to 0.505 moves every FedProx pinned value by at
#: least 5.5e-06, and the learning rate from 0.1 to 0.101 moves every SCAFFOLD
#: pinned value by at least 1.0e-04. So the tolerance absorbs the hardware and
#: still fails a 1% change to either.
PIN_STEPS = 4
PIN_TOLERANCE = PIN_STEPS * FLOAT32_STEP_BELOW_ONE


class _TinyClassificationTask(TorchClassificationTask):
    """The real TorchClassificationTask -- real ``_criterion``, real
    ``train_step`` -- over a fixed tiny linear model, so the trajectory is
    small enough to print and check by hand instead of a FEMNIST-sized one."""

    def build_model(self, config: dict | None = None) -> nn.Module:
        torch.manual_seed(0)
        return nn.Linear(3, 2)


def _initial_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _data() -> dict[str, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    return {
        "train": {
            "X": torch.randn(6, 3, generator=generator),
            "y": torch.randint(0, 2, (6,), generator=generator),
        }
    }


def _moved(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return sum(float((a[name] - b[name]).abs().sum()) for name in a)


def _scaffold_result(server_control: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Run one SCAFFOLD local iteration, one pass (one batch of all 6
    examples: one step), from the same initial state, return (final state,
    metrics)."""

    client = TorchScaffoldClient(
        client_id="c0",
        task=_TinyClassificationTask(),
        model_config={},
        client_data=_data(),
        local_iterations=1,
        batch_size=6,
        learning_rate=_LEARNING_RATE,
        train_shuffle=False,
    )
    result = client.fit(
        FitRequest(
            round_id=1,
            client_id="c0",
            payload={
                "model_state": copy.deepcopy(_initial_state()),
                "server_control": server_control,
            },
        )
    )
    return result.payload["model_state"], result.metrics


def _fedprox_result(proximal_mu: float) -> tuple[dict, dict]:
    """Run one FedProx local iteration, one pass (two batches of 3: two
    steps), from the same initial state, return (final state, metrics)."""

    client = TorchFedProxClient(
        client_id="c0",
        task=_TinyClassificationTask(),
        model_config={},
        client_data=_data(),
        local_iterations=1,
        batch_size=3,
        learning_rate=_LEARNING_RATE,
        train_shuffle=False,
        proximal_mu=proximal_mu,
    )
    result = client.fit(
        FitRequest(
            round_id=1,
            client_id="c0",
            payload={"model_state": copy.deepcopy(_initial_state())},
        )
    )
    return result.payload["model_state"], result.metrics


class ScaffoldClosedFormPredictionTest(unittest.TestCase):
    """One local step, so the loss gradient at the start is identical
    whether or not a correction is applied -- nothing has moved yet -- and
    the correction's whole effect on the update is exactly
    ``-lr * (server_control - client_control)``, independent of the loss."""

    def test_the_correction_matches_a_closed_form_prediction(self) -> None:
        initial_state = _initial_state()
        zero_control = {name: torch.zeros_like(value) for name, value in initial_state.items()}
        server_control = {
            name: torch.full_like(value, 0.05) for name, value in initial_state.items()
        }

        plain, _ = _scaffold_result(zero_control)
        corrected, _ = _scaffold_result(server_control)

        predicted = {name: plain[name] - _LEARNING_RATE * server_control[name] for name in plain}
        print(
            "SCAFFOLD single-step correction: actual vs closed-form "
            "lr * server_control, max elementwise error = "
            f"{max(float((predicted[n] - corrected[n]).abs().max()) for n in plain):.2e}"
        )
        for name in plain:
            with self.subTest(parameter=name):
                torch.testing.assert_close(corrected[name], predicted[name], atol=1e-6, rtol=0)

        # A correction that no-ops would also pass an all-zero prediction --
        # confirm the effect is actually there before trusting the match above.
        self.assertGreater(_moved(corrected, plain), 1e-4)


class FedProxClosedFormPredictionTest(unittest.TestCase):
    """Two local steps. The proximal gradient ``mu * (w - w0)`` is exactly
    zero at the reference point ``w0``, so step 1 is identical with or
    without it; step 2's correction is then exactly
    ``-lr * mu * (w1 - w0)``, where ``w1`` is measured independently of
    TorchFedProxClient, through the bare task."""

    def test_the_correction_matches_a_closed_form_prediction(self) -> None:
        initial_state = _initial_state()

        base, _ = _fedprox_result(0.0)
        prox, _ = _fedprox_result(_PROXIMAL_MU)

        # w1, measured independently: one step of the bare task's own
        # train_step on the first batch, nothing from either client class.
        model = _TinyClassificationTask().build_model()
        data = _data()["train"]
        optimizer = torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE)
        _TinyClassificationTask().train_step(model, (data["X"][:3], data["y"][:3]), optimizer)
        w1 = {name: value.detach().clone() for name, value in model.state_dict().items()}

        predicted = {
            name: base[name] - _LEARNING_RATE * _PROXIMAL_MU * (w1[name] - initial_state[name])
            for name in base
        }
        print(
            "FedProx step-2 correction: actual vs closed-form "
            "-lr * mu * (w1 - w0), max elementwise error = "
            f"{max(float((predicted[n] - prox[n]).abs().max()) for n in base):.2e}"
        )
        for name in base:
            with self.subTest(parameter=name):
                torch.testing.assert_close(prox[name], predicted[name], atol=1e-6, rtol=0)

        self.assertGreater(_moved(prox, base), 1e-4)


class TrajectoryIsPinnedTest(unittest.TestCase):
    """One run's floating-point output, from the two scenarios above.

    Compared within PIN_TOLERANCE, not bit for bit. The refactor this was
    first recorded for has landed and matched it exactly; what it is now is a
    regression pin, which fails when either client's local step computes
    something else.
    """

    def test_scaffold_single_step_weight_and_bias(self) -> None:
        server_control = {
            name: torch.full_like(value, 0.05) for name, value in _initial_state().items()
        }
        state, metrics = _scaffold_result(server_control)
        print("SCAFFOLD pinned weight[0,:3] =", state["weight"].flatten()[:3].tolist())
        print("SCAFFOLD pinned bias         =", state["bias"].tolist())
        torch.testing.assert_close(
            state["weight"].flatten()[:3],
            torch.tensor([0.006106026004999876, 0.25745514035224915, -0.5030878782272339]),
            atol=PIN_TOLERANCE,
            rtol=0,
        )
        torch.testing.assert_close(
            state["bias"],
            torch.tensor([-0.000890903000254184, 0.43722668290138245]),
            atol=PIN_TOLERANCE,
            rtol=0,
        )
        self.assertAlmostEqual(metrics["fit_loss"], 0.6598123908042908, delta=PIN_TOLERANCE)

    def test_fedprox_two_step_weight_and_bias(self) -> None:
        state, metrics = _fedprox_result(_PROXIMAL_MU)
        print("FedProx pinned weight[0,:3] =", state["weight"].flatten()[:3].tolist())
        print("FedProx pinned bias         =", state["bias"].tolist())
        torch.testing.assert_close(
            state["weight"].flatten()[:3],
            torch.tensor([0.025114120915532112, 0.21753637492656708, -0.5193787813186646]),
            atol=PIN_TOLERANCE,
            rtol=0,
        )
        torch.testing.assert_close(
            state["bias"],
            torch.tensor([0.020940179005265236, 0.4253956079483032]),
            atol=PIN_TOLERANCE,
            rtol=0,
        )
        self.assertAlmostEqual(
            metrics["fit_proximal_loss"], 0.006182524375617504, delta=PIN_TOLERANCE
        )
        self.assertAlmostEqual(metrics["fit_total_loss"], 0.6130761979147792, delta=PIN_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
