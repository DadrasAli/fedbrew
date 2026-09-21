"""`update_mode: full_gradient` is one step on the gradient of the whole split.

One iteration is one optimizer step on the exact gradient of the task's
training loss over every sample in the client's train split: the gradient
`train_step` would take on one batch holding all of them. It is computed batch
by batch, at the iteration's starting model, so `batch_size` decides how much is
in memory and nothing else.

Exact means each batch gradient is weighted by its share of the count the
task's loss is a mean over, `TaskAdapter.train_loss_denominator`. For a loss
that averages over examples that is the example count, and the mode equals
`frozen_batch_gradients` under `examples` weighting. For the causal-LM loss,
a mean over active target tokens, it is the token count, and the frozen mode's
example weighting is not the gradient of the pass: FINDINGS.csv POST-F19,
53.5% on two batches of 2 and 8 tokens, which is why that mode is refused on
such a task.

Three guards, each over three tasks -- classification (examples), the
fed-lasso example (examples, plus a parameter penalty inside every batch's
loss) and the causal LM (tokens):

- `OneBatchOverTheWholeSplitTest`: the step equals `single_batch` on one batch
  holding every sample, which is `train_step` with a real optimizer and no
  combination at all.
- `InvariantToBatchSizeAndShuffleTest`: the step is the same at every batch
  size, including ones that leave a short last batch, and in any order.
- `EqualToFrozenWhereThatIsExactTest`: the step equals `frozen_batch_gradients`
  + `examples` on the two example-mean tasks, and on the token-mean one, where
  the frozen mode would not be the gradient, that mode is refused (POST-F19).

`OwnLoopRulesTest`: fedprox, scaffold and fedlalr, which run their own loop
rather than the engine, take the mode through `full_gradient_into_grad`. Each
rule's step on the whole split equals its own `sequential_epoch` over one batch
holding every row, and with no correction to make (fedprox at mu 0, scaffold at
zero controls) it is fedavg's `full_gradient` exactly.

Mutated in place and restored before this counted, each caught: weighting batches
by example count instead of the task's denominator (the causal-LM cases fail),
the causal-LM override returning the example count (same), the division by the
summed denominator dropped (every case fails), a parameter update after every
batch instead of one per pass (every case fails), and each refusal removed
(`RefusalsTest`).
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import Tensor, nn

from fedbrew.clients.fedavg_client import FedAvgClient
from fedbrew.clients.local_update_modes import run_delta_sgd_update_mode, run_sgd_update_mode
from fedbrew.clients.torch_adamw_client import TorchAdamWClient
from fedbrew.clients.torch_fedlalr_client import TorchFedLALRClient
from fedbrew.clients.torch_fedprox_client import TorchFedProxClient
from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.config import amp_unsupported_sgd_engine_setting, load_config
from fedbrew.core.protocol import FitRequest
from fedbrew.core.refusal import RunRefused
from fedbrew.tasks.base import TaskAdapter
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask
from tests.test_empty_training_batches import _Task

REPO_ROOT = Path(__file__).resolve().parent.parent
LEARNING_RATE = 0.1


def _flat(model: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def _relative(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / b.norm())


class _Problem:
    """A task, a model at a fixed start, and a train split cut any way asked."""

    #: Relative tolerance on the update: float64 problems reach rounding, and
    #: float32 ones the summation order of a few hundred terms.
    tolerance = 1e-5
    rows = 0

    def __init__(self) -> None:
        self.task: TaskAdapter
        self.model: nn.Module

    def loader(self, batch_size: int, seed: int | None) -> Iterable[Any]:
        """The split in batches of `batch_size`, shuffled by `seed` or in order."""

        raise NotImplementedError

    def update(
        self,
        mode: str,
        batch_size: int,
        *,
        seed: int | None = None,
        iterations: int = 1,
        weighting: str = "examples",
        max_grad_norm: float | None = None,
    ) -> tuple[Tensor, int]:
        """Parameters moved by one local update, and the steps it reports."""

        model = copy.deepcopy(self.model)
        before = _flat(model)
        result = run_sgd_update_mode(
            task=self.task,
            model=model,
            train_loader=self.loader(batch_size, seed),
            local_iterations=iterations,
            learning_rate=LEARNING_RATE,
            update_mode=mode,
            frozen_gradient_weighting=weighting,
            client_id="client_0",
            max_grad_norm=max_grad_norm,
        )
        return before - _flat(model), result.optimizer_steps

    def one_batch(self, iterations: int = 1) -> Tensor:
        """`single_batch` over one batch holding every sample: the definition."""

        return self.update("single_batch", self.rows, iterations=iterations)[0]


class _Classification(_Problem):
    rows = 11

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(3)
        self.x = torch.randn(self.rows, 5, generator=generator)
        self.y = torch.randint(0, 3, (self.rows,), generator=generator)
        self.task = TorchClassificationTask(model_config={}, batch_size=4, device="cpu")
        torch.manual_seed(5)
        self.model = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 3))

    def loader(self, batch_size: int, seed: int | None) -> Iterable[Any]:
        config: dict[str, Any] = {"batch_size": batch_size, "shuffle": seed is not None}
        if seed is not None:
            config["seed"] = seed
        return self.task.build_dataloader({"x": self.x, "y": self.y}, config)


def _fed_lasso_module() -> Any:
    """examples/fed-lasso/problem.py, imported for its classes without registering."""

    name = "fed_lasso_problem_for_full_gradient"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "examples" / "fed-lasso" / "problem.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class _FedLasso(_Problem):
    """16 rows, float64, and `lam ||x||_1` inside every batch's loss."""

    tolerance = 1e-12
    rows = 16

    def __init__(self) -> None:
        problem = _fed_lasso_module()
        spec = problem.ProblemSpec()
        self.task = problem.FedLassoTask(
            model_config={"input_dim": spec.dim, "penalty_strength": spec.penalty_strength},
            dataset_metadata={"reference": problem.reference_of(spec)},
        )
        self.data = {"x": spec.design(), "y": spec.client_targets()[3]}
        self.model = problem.LassoModel(spec.dim, spec.penalty_strength)
        with torch.no_grad():
            self.model.x.copy_(torch.linspace(-0.3, 0.3, spec.dim, dtype=torch.float64))

    def loader(self, batch_size: int, seed: int | None) -> Iterable[Any]:
        config: dict[str, Any] = {"batch_size": batch_size, "shuffle": seed is not None}
        if seed is not None:
            config["seed"] = seed
        return self.task.build_dataloader(self.data, config)


_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class _CausalLM(_Problem):
    """Eight sequences whose active target tokens run from 0 to 4."""

    rows = 8

    def __init__(self) -> None:
        from fedbrew.tasks.causal_lm import TorchCausalLMTask

        torch.manual_seed(7)
        model_config = {
            "name": "tiny_gpt2",
            "vocab_size": 258,
            "sequence_length": 4,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "dropout": 0.0,
            "pad_token_id": 0,
        }
        self.task = TorchCausalLMTask(model_config=model_config, batch_size=2, device="cpu")
        self.model = self.task.build_model()
        generator = torch.Generator().manual_seed(11)
        self.inputs = torch.randint(1, 258, (self.rows, 4), generator=generator)
        targets = torch.randint(1, 258, (self.rows, 4), generator=generator)
        # Token 0 is the declared padding: sequence i keeps i % 5 active targets.
        for index in range(self.rows):
            targets[index, index % 5 :] = 0
        self.targets = targets

    def loader(self, batch_size: int, seed: int | None) -> Iterable[Any]:
        order = torch.arange(self.rows)
        if seed is not None:
            order = torch.randperm(self.rows, generator=torch.Generator().manual_seed(seed))
        return [
            (
                self.inputs[order[start : start + batch_size]],
                self.targets[order[start : start + batch_size]],
            )
            for start in range(0, self.rows, batch_size)
        ]


def _problems() -> list[tuple[str, Callable[[], _Problem]]]:
    problems: list[tuple[str, Callable[[], _Problem]]] = [
        ("classification", _Classification),
        ("fed-lasso", _FedLasso),
    ]
    if _HAS_TRANSFORMERS:
        problems.append(("causal-lm", _CausalLM))
    return problems


class OneBatchOverTheWholeSplitTest(unittest.TestCase):
    """The step is `train_step` on one batch holding every training sample."""

    def test_one_iteration_is_one_step_on_the_whole_split(self) -> None:
        for label, build in _problems():
            problem = build()
            reference = problem.one_batch()
            for batch_size in (1, 3, problem.rows - 1):
                with self.subTest(task=label, batch_size=batch_size):
                    moved, steps = problem.update("full_gradient", batch_size)
                    self.assertEqual(steps, 1)
                    self.assertLess(_relative(moved, reference), problem.tolerance)

    def test_k_iterations_are_k_steps_on_the_whole_split(self) -> None:
        for label, build in _problems():
            problem = build()
            reference = problem.one_batch(iterations=3)
            with self.subTest(task=label):
                moved, steps = problem.update("full_gradient", 3, iterations=3)
                self.assertEqual(steps, 3)
                self.assertLess(_relative(moved, reference), problem.tolerance)

    def test_a_clip_bounds_the_one_combined_gradient(self) -> None:
        """Clipped once per update, like the frozen mode: the step's norm is
        `lr * max_grad_norm` when the gradient is longer than that."""

        problem = _Classification()
        gradient_norm = float(problem.one_batch().norm()) / LEARNING_RATE
        threshold = gradient_norm / 4.0
        moved, _ = problem.update("full_gradient", 3, max_grad_norm=threshold)
        self.assertAlmostEqual(float(moved.norm()), LEARNING_RATE * threshold, places=6)
        direction = problem.one_batch()
        self.assertLess(_relative(moved / moved.norm(), direction / direction.norm()), 1e-5)


class InvariantToBatchSizeAndShuffleTest(unittest.TestCase):
    """`batch_size` and `train_shuffle` move nothing but the summation order."""

    def test_every_batch_size_and_order_gives_one_step(self) -> None:
        for label, build in _problems():
            problem = build()
            reference, _ = problem.update("full_gradient", problem.rows)
            for batch_size in (1, 2, 3, 5, problem.rows, 2 * problem.rows):
                for seed in (None, 1, 2):
                    with self.subTest(task=label, batch_size=batch_size, seed=seed):
                        moved, _ = problem.update("full_gradient", batch_size, seed=seed)
                        self.assertLess(_relative(moved, reference), problem.tolerance)


class EqualToFrozenWhereThatIsExactTest(unittest.TestCase):
    """Where `frozen_batch_gradients` + `examples` is exact, the two agree."""

    def test_they_agree_on_a_loss_that_averages_over_examples(self) -> None:
        for label, build in (("classification", _Classification), ("fed-lasso", _FedLasso)):
            problem = build()
            for batch_size in (1, 3, problem.rows):
                with self.subTest(task=label, batch_size=batch_size):
                    full, _ = problem.update("full_gradient", batch_size)
                    frozen, _ = problem.update("frozen_batch_gradients", batch_size)
                    self.assertLess(_relative(full, frozen), problem.tolerance)

    @unittest.skipUnless(_HAS_TRANSFORMERS, "transformers is an optional LLM dependency")
    def test_where_it_would_not_be_exact_the_frozen_mode_is_refused(self) -> None:
        """FINDINGS.csv POST-F19: on the token-mean loss `full_gradient` is still
        the gradient of the pass, and the frozen mode does not run."""

        problem = _CausalLM()
        full, _ = problem.update("full_gradient", 2)
        self.assertLess(_relative(full, problem.one_batch()), problem.tolerance)
        with self.assertRaises(ValueError) as caught:
            problem.update("frozen_batch_gradients", 2)
        self.assertIn("POST-F19", str(caught.exception))


def _config_with(source: str, change: Callable[[dict[str, Any]], None]) -> Any:
    document = yaml.safe_load((REPO_ROOT / source).read_text(encoding="utf-8"))
    change(document)
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "config.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return load_config(path)


def _mode(mode: str, **client: Any) -> Callable[[dict[str, Any]], None]:
    def change(document: dict[str, Any]) -> None:
        document["client"]["update_mode"] = mode
        document["client"].update(client)

    return change


@pytest.mark.fast
class RefusalsTest(unittest.TestCase):
    """Where the mode loads, and the three places it is refused."""

    def test_every_rule_that_takes_a_mode_loads_it(self) -> None:
        for source in (
            "configs/mnist/fedavg.yaml",
            "configs/mnist/centralized.yaml",
            "configs/femnist/fedavg_ft.yaml",
            *OWN_LOOP_CONFIGS.values(),
        ):
            with self.subTest(config=source):
                config = _config_with(source, _mode("full_gradient"))
                self.assertEqual(config.client.extra["update_mode"], "full_gradient")

    def test_delta_sgd_loads_it_and_refuses_drop_last_under_it(self) -> None:
        config = _config_with("configs/femnist/delta_sgd.yaml", _mode("full_gradient"))
        self.assertEqual(config.client.extra["update_mode"], "full_gradient")
        with self.assertRaises(RunRefused) as caught:
            _config_with("configs/femnist/delta_sgd.yaml", _mode("full_gradient", drop_last=True))
        self.assertIn("client.drop_last", str(caught.exception))

    def test_drop_last_is_refused_under_it(self) -> None:
        with self.assertRaises(RunRefused) as caught:
            _config_with("configs/mnist/fedavg.yaml", _mode("full_gradient", drop_last=True))
        self.assertIn("client.drop_last", str(caught.exception))
        _config_with("configs/mnist/fedavg.yaml", _mode("sequential_epoch", drop_last=True))

    def test_amp_is_refused_under_it(self) -> None:
        config = _config_with("configs/mnist/fedavg.yaml", _mode("full_gradient"))
        config.runtime.use_amp = True
        self.assertEqual(amp_unsupported_sgd_engine_setting(config), "update_mode: full_gradient")


@unittest.skipUnless(_HAS_TRANSFORMERS, "transformers is an optional LLM dependency")
class NothingToAverageOverTest(unittest.TestCase):
    def test_a_split_with_no_active_token_is_refused(self) -> None:
        problem = _CausalLM()
        problem.targets = torch.zeros_like(problem.targets)
        with self.assertRaises(ValueError) as caught:
            problem.update("full_gradient", 3)
        self.assertIn("nothing its loss averages over", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


#: A shipped config per own-loop rule, for the config-level checks.
OWN_LOOP_CONFIGS = {
    "fedprox": "configs/femnist/fedprox.yaml",
    "scaffold": "configs/femnist/scaffold.yaml",
    "fedlalr": "configs/femnist/fedlalr.yaml",
    "local_sgd": "configs/dev/synthetic.yaml",
    "local_adamw": "configs/dev/tiny_causal_lm.yaml",
}
_OWN_LOOP_ROWS = 13


class _Float64Task(_Task):
    """Two-class cross-entropy on a linear model in float64, so rounding is all that differs."""

    def build_model(self, config: Any = None) -> nn.Module:
        torch.manual_seed(0)
        return nn.Linear(3, 2).double()


def _own_loop_data() -> dict[str, dict[str, Tensor]]:
    generator = torch.Generator().manual_seed(17)
    split = {
        "x": torch.randn(_OWN_LOOP_ROWS, 3, generator=generator, dtype=torch.float64),
        "y": torch.randint(0, 2, (_OWN_LOOP_ROWS,), generator=generator),
    }
    return {"train": split, "eval": split}


def _own_loop_client(
    rule: str, mode: str | None, batch_size: int, iterations: int = 3, **extra: Any
) -> Any:
    common: dict[str, Any] = {
        "client_id": "c0",
        "task": _Float64Task(),
        "model_config": {},
        "client_data": _own_loop_data(),
        "local_iterations": iterations,
        "batch_size": batch_size,
        "learning_rate": 0.1,
        "train_shuffle": False,
        "update_mode": mode,
    }
    if rule == "fedprox":
        return TorchFedProxClient(**common, proximal_mu=extra.get("proximal_mu", 0.3))
    if rule == "scaffold":
        return TorchScaffoldClient(**common)
    if rule == "local_sgd":
        return TorchSGDClient(
            **common,
            momentum=extra.get("momentum", 0.5),
            weight_decay=extra.get("weight_decay", 0.01),
            nesterov=False,
            learning_rate_schedule="constant",
            min_learning_rate=0.0,
        )
    if rule == "local_adamw":
        return TorchAdamWClient(
            **common,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            learning_rate_schedule="constant",
            min_learning_rate=0.0,
            total_rounds=1,
            max_local_steps=extra.get("max_local_steps"),
        )
    return TorchFedLALRClient(**common, beta1=0.9, beta2=0.999, epsilon=1e-8)


def _own_loop_fit(client: Any, control: float = 0.05) -> dict[str, dict[str, Tensor]]:
    """Every model-shaped state one round returns, from a fixed start and broadcast."""

    state = {k: v.detach().clone() for k, v in _Float64Task().build_model().state_dict().items()}
    payload = {
        "model_state": state,
        "server_control": {k: torch.full_like(v, control) for k, v in state.items()},
        "momentum_state": {k: torch.full_like(v, 0.01) for k, v in state.items()},
        "second_moment_state": {k: torch.full_like(v, 1e-4) for k, v in state.items()},
    }
    result = client.fit(FitRequest(round_id=1, client_id="c0", payload=payload))
    return {
        key: value
        for key, value in result.payload.items()
        if isinstance(value, dict) and value and all(isinstance(v, Tensor) for v in value.values())
    }


class OwnLoopRulesTest(unittest.TestCase):
    """The rules with a loop of their own under `full_gradient` and `sequential_epoch`."""

    def assert_states_close(self, got: dict, want: dict, tolerance: float) -> None:
        self.assertEqual(set(got), set(want))
        for key in want:
            for name in want[key]:
                with self.subTest(state=key, parameter=name):
                    self.assertLess(_relative(got[key][name], want[key][name]), tolerance)

    def test_full_gradient_is_its_own_step_on_one_batch_of_the_whole_split(self) -> None:
        for rule in OWN_LOOP_CONFIGS:
            with self.subTest(rule=rule):
                one_batch = _own_loop_fit(
                    _own_loop_client(rule, "sequential_epoch", _OWN_LOOP_ROWS)
                )
                for batch_size in (1, 5, _OWN_LOOP_ROWS, 2 * _OWN_LOOP_ROWS):
                    full = _own_loop_fit(_own_loop_client(rule, "full_gradient", batch_size))
                    self.assert_states_close(full, one_batch, 1e-12)
                # The mode is a different computation wherever a pass has more
                # than one batch, so the check above is not passing vacuously.
                passes = _own_loop_fit(_own_loop_client(rule, "sequential_epoch", 5))
                self.assertGreater(
                    _relative(passes["model_state"]["weight"], one_batch["model_state"]["weight"]),
                    1e-6,
                )

    def test_unset_is_sequential_epoch(self) -> None:
        for rule in OWN_LOOP_CONFIGS:
            with self.subTest(rule=rule):
                self.assertEqual(_own_loop_client(rule, None, 5).update_mode, "sequential_epoch")
                torch.testing.assert_close(
                    _own_loop_fit(_own_loop_client(rule, None, 5)),
                    _own_loop_fit(_own_loop_client(rule, "sequential_epoch", 5)),
                    rtol=0,
                    atol=0,
                )

    def test_with_nothing_to_correct_it_is_fedavgs_full_gradient_bit_for_bit(self) -> None:
        fedavg = FedAvgClient(
            client_id="c0",
            task=_Float64Task(),
            model_config={},
            client_data=_own_loop_data(),
            local_iterations=3,
            batch_size=5,
            learning_rate=0.1,
            train_shuffle=False,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            learning_rate_schedule="constant",
            min_learning_rate=0.0,
            update_mode="full_gradient",
            frozen_gradient_weighting="examples",
        )
        want = _own_loop_fit(fedavg)["model_state"]
        for rule, client, control in (
            ("fedprox", _own_loop_client("fedprox", "full_gradient", 5, proximal_mu=0.0), 0.05),
            ("scaffold", _own_loop_client("scaffold", "full_gradient", 5), 0.0),
            (
                "local_sgd",
                _own_loop_client("local_sgd", "full_gradient", 5, momentum=0.0, weight_decay=0.0),
                0.05,
            ),
        ):
            with self.subTest(rule=rule):
                got = _own_loop_fit(client, control=control)["model_state"]
                torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.fast
class OwnLoopConfigTest(unittest.TestCase):
    """What the own-loop rules accept and refuse at load."""

    def test_both_modes_and_none_load(self) -> None:
        for rule, source in OWN_LOOP_CONFIGS.items():
            for mode in ("sequential_epoch", "full_gradient"):
                with self.subTest(rule=rule, mode=mode):
                    _config_with(source, _mode(mode))
            with self.subTest(rule=rule, mode=None):
                self.assertNotIn("update_mode", _config_with(source, lambda _: None).client.extra)

    def test_the_engine_only_modes_are_refused(self) -> None:
        for rule, source in OWN_LOOP_CONFIGS.items():
            for mode in ("single_batch", "frozen_batch_gradients"):
                with self.subTest(rule=rule, mode=mode):
                    with self.assertRaises(RunRefused) as caught:
                        _config_with(source, _mode(mode))
                    # causal_lm refuses the frozen mode first, by POST-F19's check.
                    self.assertRegex(
                        str(caught.exception),
                        "client.update_mode must be one of|frozen_batch_gradients combines",
                    )

    def test_drop_last_and_amp_are_refused_under_full_gradient(self) -> None:
        for rule, source in OWN_LOOP_CONFIGS.items():
            with self.subTest(rule=rule):
                with self.assertRaises(RunRefused) as caught:
                    _config_with(source, _mode("full_gradient", drop_last=True))
                self.assertIn("client.drop_last", str(caught.exception))
        # These compose with AMP in their own loop, and not under full_gradient,
        # which steps through _GradientOnlyOptimizer.
        for rule in ("fedprox", "scaffold", "local_sgd", "local_adamw"):
            with self.subTest(rule=rule, amp=True):
                config = _config_with(OWN_LOOP_CONFIGS[rule], _mode("full_gradient"))
                config.runtime.use_amp = True
                self.assertEqual(
                    amp_unsupported_sgd_engine_setting(config), "update_mode: full_gradient"
                )
                config = _config_with(OWN_LOOP_CONFIGS[rule], _mode("sequential_epoch"))
                config.runtime.use_amp = True
                self.assertIsNone(amp_unsupported_sgd_engine_setting(config))


class LocalAdamWStepCapTest(unittest.TestCase):
    """`max_local_steps` under `full_gradient`: an iteration is one step, so it caps them."""

    def test_the_cap_stops_at_that_many_full_gradient_steps(self) -> None:
        capped = _own_loop_client("local_adamw", "full_gradient", 5, max_local_steps=2)
        uncapped = _own_loop_client("local_adamw", "full_gradient", 5, iterations=2)
        torch.testing.assert_close(_own_loop_fit(capped), _own_loop_fit(uncapped), rtol=0, atol=0)


def _delta_sgd(mode: str, batch_size: int, iterations: int) -> tuple[Tensor, list[float]]:
    """Parameters after one Delta-SGD round on the float64 problem, and its step sizes."""

    task = _Float64Task()
    model = task.build_model()
    result = run_delta_sgd_update_mode(
        task=task,
        model=model,
        train_loader=task.build_dataloader(_own_loop_data()["train"], {"batch_size": batch_size}),
        local_iterations=iterations,
        update_mode=mode,
        frozen_gradient_weighting="examples",
        client_id="c0",
        eta_0=0.2,
        theta_0=1.0,
        gamma=2.0,
        delta=0.1,
    )
    return _flat(model), result.step_sizes


class DeltaSGDFullGradientTest(unittest.TestCase):
    """delta_sgd under `full_gradient`: its rule on the exact gradient of the split."""

    def test_it_is_the_rule_over_one_batch_of_the_whole_split(self) -> None:
        want, want_steps = _delta_sgd("sequential_epoch", _OWN_LOOP_ROWS, 3)
        self.assertEqual(len(want_steps), 3)
        # The rule chose its own second and third step sizes.
        self.assertNotEqual(want_steps[1], want_steps[0])
        for batch_size in (1, 5, _OWN_LOOP_ROWS, 2 * _OWN_LOOP_ROWS):
            with self.subTest(batch_size=batch_size):
                got, got_steps = _delta_sgd("full_gradient", batch_size, 3)
                self.assertLess(_relative(got, want), 1e-12)
                for step, (a, b) in enumerate(zip(got_steps, want_steps, strict=True)):
                    self.assertLess(abs(a - b) / b, 1e-12, f"step {step}")
        passes, _ = _delta_sgd("sequential_epoch", 5, 3)
        self.assertGreater(_relative(passes, want), 1e-6)

    def test_one_step_a_round_is_fedavg_at_eta_0(self) -> None:
        """eta resets to eta_0 every round, so a one-step round cannot adapt."""

        delta, steps = _delta_sgd("full_gradient", 5, 1)
        self.assertEqual(steps, [0.2])
        task = _Float64Task()
        model = task.build_model()
        run_sgd_update_mode(
            task=task,
            model=model,
            train_loader=task.build_dataloader(_own_loop_data()["train"], {"batch_size": 5}),
            local_iterations=1,
            learning_rate=0.2,
            update_mode="full_gradient",
            frozen_gradient_weighting="examples",
            client_id="c0",
        )
        self.assertTrue(torch.equal(delta, _flat(model)))
