"""A batched step reads each client's own hyperparameters, and rounds as torch's optimizers do.

Clients stepped together share their program's shape -- which terms the step
has -- not its values: the learning rate, momentum, weight decay, FedProx's mu
and the clipping norm are each client's own tensors (``ProgramValues``). Each
scalar form of torch's optimizers has a tensor form that rounds the same, so a
client stepped beside clients with other values is the client stepped alone,
bit for bit, and a client stepped alone is ``torch.optim``'s step.
"""

from __future__ import annotations

import unittest

import torch

from fedbrew.clients.batch_orders import LocalLoop
from fedbrew.clients.batched_update import (
    ClientBatchPlan,
    LocalProgram,
    OptimizerSpec,
    ProgramValues,
    apply_update,
    initial_optimizer_state,
)

ROWS = 4
LRS = [0.1, 0.037, 3e-3, 0.7]


def _programs(kind: str) -> list[LocalProgram]:
    """``ROWS`` programs of one shape, every value different."""

    programs = []
    for row, lr in enumerate(LRS):
        if kind == "sgd":
            spec = OptimizerSpec("sgd", lr=lr)
            programs.append(LocalProgram(optimizer=spec))
        elif kind == "momentum":
            spec = OptimizerSpec(
                "sgd", lr=lr, momentum=0.5 + 0.1 * row, weight_decay=1e-3 * (row + 1), nesterov=True
            )
            programs.append(LocalProgram(optimizer=spec))
        elif kind == "prox":
            spec = OptimizerSpec("sgd", lr=lr)
            programs.append(LocalProgram(optimizer=spec, proximal_mu=0.01 * (row + 1)))
        elif kind == "clip_batch":
            spec = OptimizerSpec("sgd", lr=lr)
            programs.append(LocalProgram(optimizer=spec, max_grad_norm=0.5 * (row + 1)))
        elif kind == "clip_combined":
            spec = OptimizerSpec("sgd", lr=lr)
            programs.append(
                LocalProgram(optimizer=spec, combine="full", max_grad_norm=0.5 * (row + 1))
            )
        elif kind == "adamw":
            spec = OptimizerSpec("adamw", lr=lr, weight_decay=1e-2 * (row + 1))
            programs.append(LocalProgram(optimizer=spec))
        else:
            raise AssertionError(kind)
    return programs


KINDS = ("sgd", "momentum", "prox", "clip_batch", "clip_combined", "adamw")


def _tensors(dtype: torch.dtype, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "weight": torch.randn(ROWS, 3, 5, generator=generator, dtype=dtype),
        "bias": torch.randn(ROWS, 3, generator=generator, dtype=dtype),
    }


def _row(tree: object, index: int) -> object:
    if isinstance(tree, dict):
        return {name: _row(value, index) for name, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree[index]
    return tree


def _steps(
    kind: str,
    dtype: torch.dtype,
    values_of,
    call,
) -> list[dict[str, torch.Tensor]]:
    """Two steps of ``kind`` from the same start, parameters after each."""

    programs = _programs(kind)
    params = _tensors(dtype, 0)
    reference = _tensors(dtype, 1)
    state = initial_optimizer_state(programs[0].optimizer, params)
    seen = []
    for step in (1, 2):
        grads = _tensors(dtype, 10 + step)
        params, state = call(programs, params, grads, state, step, reference, values_of)
        seen.append(params)
    return seen


def _together(programs, params, grads, state, step, reference, values_of):
    values = values_of(programs).at(step)
    return torch.func.vmap(
        lambda p, g, s, r, v: apply_update(programs[0], p, g, s, step, r, values=v)
    )(params, grads, state, reference, values)


def _alone(programs, params, grads, state, step, reference, values_of):
    rows = []
    for index, program in enumerate(programs):
        values = {name: value[0] for name, value in values_of([program]).at(step).items()}
        rows.append(
            apply_update(
                program,
                _row(params, index),
                _row(grads, index),
                _row(state, index),
                step,
                _row(reference, index),
                values=values,
            )
        )
    new_params = {name: torch.stack([row[0][name] for row in rows]) for name in params}
    new_state = {
        slot: {name: torch.stack([row[1][slot][name] for row in rows]) for name in params}
        for slot in rows[0][1]
    }
    return new_params, new_state


class ABucketSharesAShapeNotValuesTest(unittest.TestCase):
    def _plan(self, program: LocalProgram) -> ClientBatchPlan:
        return ClientBatchPlan(
            program=program,
            train_data=None,
            loop=LocalLoop(epochs=1),
            train_order=None,
            eval_order=None,
            evaluate=True,
            start={},
            client_id="c",
            seed=0,
            refuse=lambda: ValueError(),
        )

    def test_values_differ_and_the_bucket_is_one(self) -> None:
        for kind in KINDS:
            with self.subTest(kind=kind):
                buckets = {self._plan(program).bucket for program in _programs(kind)}
                self.assertEqual(len(buckets), 1)

    def test_a_term_present_or_not_is_another_bucket(self) -> None:
        plain = LocalProgram(optimizer=OptimizerSpec("sgd", lr=0.1))
        for other in (
            LocalProgram(optimizer=OptimizerSpec("sgd", lr=0.1, momentum=0.9)),
            LocalProgram(optimizer=OptimizerSpec("sgd", lr=0.1, weight_decay=1e-4)),
            LocalProgram(optimizer=OptimizerSpec("sgd", lr=0.1), proximal_mu=0.01),
            LocalProgram(optimizer=OptimizerSpec("sgd", lr=0.1), max_grad_norm=1.0),
            LocalProgram(optimizer=OptimizerSpec("adamw", lr=0.1)),
        ):
            with self.subTest(other=other):
                self.assertNotEqual(self._plan(plain).bucket, self._plan(other).bucket)


class AClientBesideOthersIsTheClientAloneTest(unittest.TestCase):
    """Stepped in one vmapped stack with other values, each row is its own step alone."""

    def test_every_shape_in_both_float_widths(self) -> None:
        for dtype in (torch.float64, torch.float32):

            def values_of(programs, dtype=dtype):
                return ProgramValues(programs, 2, dtype, "cpu")

            for kind in KINDS:
                with self.subTest(kind=kind, dtype=dtype):
                    together = _steps(kind, dtype, values_of, _together)
                    alone = _steps(kind, dtype, values_of, _alone)
                    for after_together, after_alone in zip(together, alone, strict=True):
                        for name in after_together:
                            self.assertTrue(
                                torch.equal(after_together[name], after_alone[name]), name
                            )

    def test_the_whole_stack_stepped_at_once_is_the_vmapped_step(self) -> None:
        # _run_summed applies an unclipped step to the stacked tensors directly.
        for kind in ("sgd", "momentum", "prox", "adamw"):
            with self.subTest(kind=kind):
                programs = _programs(kind)
                values = ProgramValues(programs, 2, torch.float64, "cpu")
                params = _tensors(torch.float64, 0)
                grads = _tensors(torch.float64, 3)
                reference = _tensors(torch.float64, 1)
                state = initial_optimizer_state(programs[0].optimizer, params)
                stacked, _ = apply_update(
                    programs[0], params, grads, state, 1, reference, values=values.at(1)
                )
                vmapped, _ = _together(
                    programs, params, grads, state, 1, reference, lambda _, v=values: v
                )
                for name in stacked:
                    self.assertTrue(torch.equal(stacked[name], vmapped[name]), name)


class AClientAloneIsTorchsStepTest(unittest.TestCase):
    """The tensor forms round as the scalar forms torch.optim uses."""

    def _torch_steps(self, optimizer_type, dtype, **settings):
        params = _tensors(dtype, 0)
        leaves = [params["weight"][0].clone(), params["bias"][0].clone()]
        optimizer = optimizer_type(leaves, foreach=False, **settings)
        seen = []
        for step in (1, 2):
            grads = _tensors(dtype, 10 + step)
            leaves[0].grad = grads["weight"][0].clone()
            leaves[1].grad = grads["bias"][0].clone()
            optimizer.step()
            seen.append([leaf.detach().clone() for leaf in leaves])
        return seen

    def _ours(self, program, dtype):
        values = ProgramValues([program], 2, dtype, "cpu")
        params = _row(_tensors(dtype, 0), 0)
        state = initial_optimizer_state(program.optimizer, params)
        seen = []
        for step in (1, 2):
            grads = _row(_tensors(dtype, 10 + step), 0)
            row = {name: value[0] for name, value in values.at(step).items()}
            params, state = apply_update(program, params, grads, state, step, values=row)
            seen.append([params["weight"], params["bias"]])
        return seen

    def test_sgd_with_momentum_nesterov_and_weight_decay(self) -> None:
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                settings = {"lr": 0.037, "momentum": 0.9, "weight_decay": 1e-3, "nesterov": True}
                ours = self._ours(LocalProgram(optimizer=OptimizerSpec("sgd", **settings)), dtype)
                theirs = self._torch_steps(torch.optim.SGD, dtype, **settings)
                for mine, reference in zip(ours, theirs, strict=True):
                    for a, b in zip(mine, reference, strict=True):
                        self.assertTrue(torch.equal(a, b))

    def test_adamw(self) -> None:
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                settings = {"lr": 0.037, "weight_decay": 1e-2}
                program = LocalProgram(optimizer=OptimizerSpec("adamw", **settings))
                ours = self._ours(program, dtype)
                theirs = self._torch_steps(torch.optim.AdamW, dtype, **settings)
                for mine, reference in zip(ours, theirs, strict=True):
                    for a, b in zip(mine, reference, strict=True):
                        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
