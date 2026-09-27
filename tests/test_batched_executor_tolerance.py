"""The batched executor computes what the sequential one computes, to summation order.

``runtime.performance.executor: batched`` trains a round's sampled clients
together (``fedbrew/core/batched_executor.py``). The same configuration is run
through both executors here, and what is compared is everything a run
computes: every round's global model (a checkpoint per round), every
persistent client state those checkpoints hold, and every non-timing cell of
the three CSVs. The design's tolerance (chapter 11 §9):

- ``|a - b| / max(|b|, 1e-300) <= 1e-12`` per CSV cell, and
  ``max |a - b| / max(max |b|, 1e-300) <= 1e-12`` per tensor, in float64;
  the identity and count columns -- ``round_id``, ``client_id``,
  ``num_clients``, ``num_examples``, ``optimizer_steps`` -- equal. A tensor
  is measured against its own scale rather than element by element because
  an element whose exact value is 0 holds rounding residue: on fed-lasso's
  first ``full_gradient`` round one coordinate is 1.08e-19 sequentially and
  9.5e-20 batched, beside a largest coordinate of 1.2e-2 (measured
  2026-09-27) -- 1.2e-18 of the tensor, and 0.13 of the element;
- **bit-identical** when every chunk holds one client
  (``executor_chunk_bytes: 1``) or a round samples one: nothing is then
  vmapped, and the batched arithmetic is the sequential arithmetic;
- the same clients selected at participation below 1;
- local iterations above 1, over shuffled batches, on every run below.

The matrix, per update rule, is the five linear examples' shipped arms, every
FedAvg update mode (each frozen weighting, with and without ``max_grad_norm``),
ragged clients whose split sizes differ (padded, masked batches and buckets of
different shapes), and partial participation.
"""

from __future__ import annotations

import copy
import csv
import math
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.core.runner import run
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from tests.test_reproducibility import TIMING

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 4
TOLERANCE = 1e-12
EXAMPLES = ("fed-lasso", "drift-quad", "simplex-lsq", "nonconvex-simplex", "pl-1d", "fed-lasso-l2")
#: Columns that are identities or counts, compared for equality.
EXACT_COLUMNS = {"round_id", "client_id", "phase", "num_clients", "num_examples", "optimizer_steps"}
#: Compared only where the executors must be bit-identical. fed-lasso's
#: ``exact_zeros`` counts the coordinates that are bit for bit 0.0, and a
#: coordinate whose exact value is 0 -- its clients come in exact +/- pairs --
#: comes out 0.0 or 1e-19 by summation order: 1 against 4 at round 1 under
#: AdamW (measured 2026-09-27). The count is a function of summation order
#: itself; the coordinates it counts are compared, to tolerance, in the model.
ORDER_COUNTED = ("exact_zeros",)
CSVS = ("round_metrics.csv", "client_update_metrics.csv", "client_metrics.csv")

_generated: dict[str, Path] = {}
_root = tempfile.TemporaryDirectory()


def example_manifest(name: str) -> Path:
    """The example's data, generated once per process from its shipped generator config."""

    if name not in _generated:
        from fedbrew.data.generate import generate_from_config

        generator = yaml.safe_load(
            (REPO_ROOT / "data" / "configs" / "examples" / f"{name}.yaml").read_text()
        )
        generator["dataset"]["output_dir"] = str(Path(_root.name) / "data" / name)
        path = Path(_root.name) / f"{name}-data.yaml"
        path.write_text(yaml.safe_dump(generator), encoding="utf-8")
        _generated[name] = Path(generate_from_config(path))
    return _generated[name]


def example_config(name: str, arm: str = "fedavg") -> dict[str, Any]:
    """A shipped arm of a linear example, on this process's data, checkpointing every round."""

    config = yaml.safe_load((REPO_ROOT / "configs" / "examples" / name / f"{arm}.yaml").read_text())
    config["data"]["path"] = str(example_manifest(name))
    config["defaults"]["global_rounds"] = ROUNDS
    config["runtime"]["quiet"] = True
    config["runtime"]["checkpointing"].update(
        enabled=True, save_last=True, save_every_round=True, keep_last=None
    )
    config["client_statistics"] = {**config.get("client_statistics", {}), "per_client_csv": True}
    return config


def ragged_clients() -> Any:
    """Train splits of 16, 13, 10, 7 and 4 rows, by client number: ragged batches and buckets."""

    real = ManifestFederatedDataset.get_client_data

    def truncated(self: ManifestFederatedDataset, client_id: str) -> dict[str, Any]:
        shard = real(self, client_id)
        keep = 16 - 3 * (int(client_id.rsplit("_", 1)[-1]) % 5)
        train = {**shard["train"]}
        for key in ("x", "y"):
            if key in train:
                train[key] = train[key][:keep]
        return {**shard, "train": train}

    return mock.patch.object(ManifestFederatedDataset, "get_client_data", truncated)


class ExecutorRuns(unittest.TestCase):
    """Runs one configuration through both executors and compares what they wrote."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self._count = 0

    def tearDown(self) -> None:
        self._directory.cleanup()

    def run_config(self, config: dict[str, Any], executor: str, **performance: Any) -> Path:
        self._count += 1
        config = copy.deepcopy(config)
        output = self.root / f"run{self._count}-{executor}"
        config["experiment"]["output_dir"] = str(output)
        config["runtime"].setdefault("performance", {}).update(executor=executor, **performance)
        path = self.root / f"run{self._count}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run(path, args=None)
        return output

    def both(
        self,
        config: dict[str, Any],
        *,
        data: Callable[[], Any] = nullcontext,
        **performance: Any,
    ) -> tuple[Path, Path]:
        """(batched, sequential) output directories for one configuration."""

        with data():
            sequential = self.run_config(config, "sequential")
            batched = self.run_config(config, "batched", **performance)
        return batched, sequential

    def assertAgree(
        self,
        batched: Path,
        sequential: Path,
        *,
        exact: bool = False,
        tolerance: float = TOLERANCE,
    ) -> None:
        """Every non-timing cell and every checkpoint agree, to ``tolerance`` or bit for bit."""

        for name in CSVS:
            self._compare_csv(batched / name, sequential / name, exact, tolerance)
        checkpoints = sorted((sequential / "checkpoints").glob("round_*.pt"))
        self.assertEqual(len(checkpoints), ROUNDS)
        for path in checkpoints:
            self._compare_checkpoint(batched / "checkpoints" / path.name, path, exact, tolerance)

    def _compare_csv(self, batched: Path, sequential: Path, exact: bool, tolerance: float) -> None:
        if not sequential.exists():
            self.assertFalse(batched.exists(), batched.name)
            return
        rows_b, rows_s = _rows(batched), _rows(sequential)
        self.assertEqual(len(rows_b), len(rows_s), batched.name)
        for number, (row_b, row_s) in enumerate(zip(rows_b, rows_s, strict=True)):
            self.assertEqual(list(row_b), list(row_s), batched.name)
            for column, value_s in row_s.items():
                if column in TIMING or value_s == row_b[column]:
                    continue
                if not exact and column.endswith(ORDER_COUNTED):
                    continue
                where = f"{batched.name} row {number} {column}: {row_b[column]} vs {value_s}"
                self.assertFalse(exact or column in EXACT_COLUMNS, where)
                self.assertLessEqual(
                    _relative(float(row_b[column]), float(value_s)), tolerance, where
                )

    def _compare_checkpoint(
        self, batched: Path, sequential: Path, exact: bool, tolerance: float
    ) -> None:
        loaded_b = torch.load(batched, weights_only=False)
        loaded_s = torch.load(sequential, weights_only=False)
        model = loaded_s["model_state"]
        pairs = [("model_state", loaded_b["model_state"], model)]
        for client, state in loaded_s["client_states"].items():
            for key, value in state.items():
                if isinstance(value, dict):
                    pairs.append((f"{client}.{key}", loaded_b["client_states"][client][key], value))
        for where, states_b, states_s in pairs:
            self.assertEqual(list(states_b), list(states_s), where)
            for key, tensor_s in states_s.items():
                tensor_b = states_b[key]
                label = f"{sequential.name} {where}.{key}"
                if exact:
                    self.assertTrue(torch.equal(tensor_b, tensor_s), label)
                else:
                    scale = max(float(tensor_s.abs().max()), 1e-300)
                    if where != "model_state" and key in model:
                        # A persistent client state is compared at least at its
                        # model tensor's scale: SCAFFOLD's c_i is (x - y_i) / (K
                        # lr) plus earlier terms, a difference of two models,
                        # and it tends to 0 as clients agree -- one bias's c_i
                        # was 9.7e-17 at round 4 of the CNN run below, all
                        # rounding -- so its error is the models' (1e-16 of
                        # their 0.37) and not a fraction of its own size, which
                        # was 2.4 there (measured 2026-09-27).
                        scale = max(scale, float(model[key].abs().max()))
                    error = float((tensor_b - tensor_s).abs().max()) / scale
                    self.assertLessEqual(error, tolerance, label)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def fedavg_modes() -> Iterator[tuple[str, dict[str, Any]]]:
    """Every FedAvg update mode, each frozen weighting, with and without clipping."""

    for mode in ("single_batch", "sequential_epoch", "frozen_batch_gradients", "full_gradient"):
        weightings = (
            ("examples", "uniform", "sum") if mode == "frozen_batch_gradients" else ("examples",)
        )
        for weighting in weightings:
            for clip in (None, 0.05):
                client = {
                    "update_mode": mode,
                    "frozen_gradient_weighting": weighting,
                    "batch_size": 3,
                }
                if clip is not None:
                    client["max_grad_norm"] = clip
                yield f"{mode}/{weighting}/clip={clip}", client


def with_client(config: dict[str, Any], **client: Any) -> dict[str, Any]:
    edited = copy.deepcopy(config)
    edited["client"].update(client)
    return edited


class EveryLinearExampleTest(ExecutorRuns):
    """The shipped FedAvg arm of each, and fed-lasso's smooth control.

    Shuffled epochs, eight clients.
    """

    def test_each_example(self) -> None:
        for name in EXAMPLES:
            with self.subTest(example=name):
                self.assertAgree(*self.both(example_config(name)))

    def test_each_example_one_client_per_chunk_is_bit_identical(self) -> None:
        for name in EXAMPLES:
            with self.subTest(example=name):
                batched, sequential = self.both(example_config(name), executor_chunk_bytes=1)
                self.assertAgree(batched, sequential, exact=True)


class EveryFedAvgModeTest(ExecutorRuns):
    """fed-lasso at batch size 3: batches of 3, 3, 3, 3, 3 and 1."""

    def test_every_mode(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                self.assertAgree(*self.both(with_client(example_config("fed-lasso"), **client)))

    def test_every_mode_one_client_per_chunk_is_bit_identical(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                config = with_client(example_config("fed-lasso"), **client)
                self.assertAgree(*self.both(config, executor_chunk_bytes=1), exact=True)


class RaggedClientsTest(ExecutorRuns):
    """Clients of 16, 13, 10, 7 and 4 rows: padded batches, masks, and several buckets."""

    def test_every_mode(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                config = with_client(example_config("fed-lasso"), **client)
                self.assertAgree(*self.both(config, data=ragged_clients))

    def test_one_client_per_chunk_is_bit_identical(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                config = with_client(example_config("fed-lasso"), **client)
                batched, sequential = self.both(config, data=ragged_clients, executor_chunk_bytes=1)
                self.assertAgree(batched, sequential, exact=True)


class PartialParticipationTest(ExecutorRuns):
    def test_the_same_clients_are_selected(self) -> None:
        config = example_config("fed-lasso")
        config["server"]["participation_rate"] = 0.5
        batched, sequential = self.both(config)
        # client_id is an exact column, so the selection is compared row by row.
        self.assertAgree(batched, sequential)
        per_round: dict[str, int] = {}
        for row in _rows(batched / "client_update_metrics.csv"):
            per_round[row["round_id"]] = per_round.get(row["round_id"], 0) + 1
        self.assertEqual(list(per_round.values()), [4] * ROUNDS)

    def test_one_sampled_client_is_bit_identical(self) -> None:
        config = example_config("fed-lasso")
        config["server"]["participation_rate"] = 0.125
        self.assertAgree(*self.both(config), exact=True)


def rule_arms() -> Iterator[tuple[str, dict[str, Any]]]:
    """local_sgd, local_adamw, FedProx and SCAFFOLD on fed-lasso, each under both of its modes."""

    arms: list[tuple[str, dict[str, Any]]] = [
        (
            "local_sgd",
            {
                "update_rule": "local_sgd",
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 0.01,
                "learning_rate_schedule": "cosine",
                "min_learning_rate": 0.001,
            },
        ),
        (
            "local_sgd/heavy-ball",
            {
                "update_rule": "local_sgd",
                "momentum": 0.5,
                "nesterov": False,
                "weight_decay": 0.0,
                "learning_rate_schedule": "constant",
                "min_learning_rate": 0.0,
            },
        ),
        (
            "local_adamw",
            {
                "update_rule": "local_adamw",
                "learning_rate": 0.01,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-8,
                "learning_rate_schedule": "cosine",
                "min_learning_rate": 0.001,
                "max_local_steps": 7,
            },
        ),
        ("fedprox", {"update_rule": "fedprox", "proximal_mu": 0.5}),
        ("scaffold", {"update_rule": "scaffold"}),
    ]
    for label, client in arms:
        for mode in ("sequential_epoch", "full_gradient"):
            yield f"{label}/{mode}", {**client, "update_mode": mode}


def rule_config(client: dict[str, Any]) -> dict[str, Any]:
    """The smooth control's FedAvg arm with its rule replaced.

    ``fed-lasso-l2``, the same problem with the ridge penalty, and not the
    lasso: `sign(x)`, the L1 term's subgradient, is 0 at exactly 0 and +-1 a
    rounding error away, so a coordinate that cancels to 0.0 in one summation
    order and to -4.3e-19 in the other takes steps `lam` apart from there.
    Under AdamW, which normalises the step, the two runs were 1.3e-3 apart
    in that coordinate at round 2 (measured 2026-09-27). That departure is the
    kink's; a smooth objective shows what the executors do. The engine's
    options a rule does not take go.
    """

    config = example_config("fed-lasso-l2")
    for option in (
        "momentum",
        "weight_decay",
        "nesterov",
        "learning_rate_schedule",
        "min_learning_rate",
        "frozen_gradient_weighting",
    ):
        config["client"].pop(option, None)
    config["client"].update(client)
    if client["update_rule"] == "scaffold":
        config["server"]["strategy"] = "scaffold"
    return config


class EveryRuleTest(ExecutorRuns):
    """Stage b: the rules with their own loop, their optimizer state and SCAFFOLD's ``c_i``.

    Every checkpoint holds every client's ``c_i``, so SCAFFOLD's persistent
    state is compared round by round with the model.
    """

    def test_every_rule(self) -> None:
        for label, client in rule_arms():
            with self.subTest(rule=label):
                self.assertAgree(*self.both(rule_config(client)))

    def test_every_rule_on_ragged_clients(self) -> None:
        for label, client in rule_arms():
            with self.subTest(rule=label):
                self.assertAgree(*self.both(rule_config(client), data=ragged_clients))

    def test_every_rule_one_client_per_chunk_is_bit_identical(self) -> None:
        for label, client in rule_arms():
            with self.subTest(rule=label):
                batched, sequential = self.both(
                    rule_config(client), data=ragged_clients, executor_chunk_bytes=1
                )
                self.assertAgree(batched, sequential, exact=True)

    def test_scaffold_at_partial_participation(self) -> None:
        """A client's ``c_i`` carries over the rounds it sits out."""

        config = rule_config({"update_rule": "scaffold", "update_mode": "sequential_epoch"})
        config["server"]["participation_rate"] = 0.5
        self.assertAgree(*self.both(config))


# ---------------------------------------------------------------------------
# Stage c: the classification task, MLP and CNN
# ---------------------------------------------------------------------------

#: The options of the shared engine a rule with its own step refuses.
ENGINE_OPTIONS = (
    "momentum",
    "weight_decay",
    "nesterov",
    "learning_rate_schedule",
    "min_learning_rate",
)

#: The float32 bound, for the classification task as it ships. float32 rounds
#: at 6e-8, and the batched products and reductions sum in another order; over
#: three rounds of the MNIST MLP at 1000 clients the largest difference was
#: 4.0e-5 relative, in one client's fit_loss, and 1.9e-5 of a tensor's scale
#: (measured 2026-09-27). The float64 fixture below holds the same code to
#: TOLERANCE.
FLOAT32_TOLERANCE = 1e-4


def classification_config(**client: Any) -> dict[str, Any]:
    """Synthetic classification: eight clients of 20 rows, an MLP, every pass every round."""

    from tests.test_reproducibility import _config

    config = _config(Path(_root.name) / "unused", rounds=ROUNDS, checkpoint=True)
    config["runtime"]["checkpointing"]["keep_last"] = None
    config["runtime"]["quiet"] = True
    config["model"]["dropout"] = 0.0
    config["data"].update(num_clients=8, samples_per_client=20)
    config["client"].update(batch_size=3, **client)
    config["client_statistics"] = {"per_client_csv": True}
    return config


def cnn_config(**client: Any) -> dict[str, Any]:
    """The same federation as one-channel 32x32 images, and the small CNN."""

    config = classification_config(**client)
    config["data"]["input_dim"] = 32 * 32
    config["model"] = {"name": "small_cnn", "input_channels": 1, "hidden_dim": 16, "num_classes": 2}
    return config


@contextmanager
def images() -> Iterator[None]:
    """Synthetic rows reshaped to (N, 1, 32, 32), for the CNN."""

    from fedbrew.data.synthetic_classification import SyntheticClassificationDataset as Data

    real_client, real_global = Data.get_client_data, Data.get_global_data

    def shaped(split: Any) -> Any:
        if isinstance(split, dict) and "X" in split:
            return {**split, "X": split["X"].reshape(-1, 1, 32, 32)}
        return split

    def client_data(self: Any, client_id: str) -> dict[str, Any]:
        return {key: shaped(value) for key, value in real_client(self, client_id).items()}

    def global_data(self: Any, split: str | None = "test") -> dict[str, Any]:
        return shaped(real_global(self, split))

    with (
        mock.patch.object(Data, "get_client_data", client_data),
        mock.patch.object(Data, "get_global_data", global_data),
    ):
        yield


@contextmanager
def float64_classification() -> Iterator[None]:
    """The classification task in float64: its rows widened to, and its models built in, it.

    The three places the task fixes float32 -- the resident rows, the
    extracted batches and the model -- and nothing else, so this is the
    shipped arithmetic at the precision the tolerance is stated in.
    """

    from fedbrew.tasks.classification import torch_classification as module

    real_construct = module.TorchClassificationTask._construct_model

    def rows(features: Any, targets: Any, device: Any) -> tuple[Any, Any]:
        return features.to(device).double(), targets.to(device).long()

    def extract(data: Any) -> tuple[Any, Any]:
        features, targets = module._raw_tensors(data)
        return features.double(), targets.long()

    def construct(self: Any, config: Any) -> Any:
        return real_construct(self, config).double()

    with (
        mock.patch.object(module, "_resident_rows", rows),
        mock.patch.object(module, "_extract_tensors", extract),
        mock.patch.object(module.TorchClassificationTask, "_construct_model", construct),
    ):
        yield


def classification_arms() -> Iterator[tuple[str, dict[str, Any]]]:
    """FedAvg's four modes, clipped, and the four rules with their own loop."""

    for mode in ("single_batch", "sequential_epoch", "frozen_batch_gradients", "full_gradient"):
        yield (
            f"fedavg/{mode}",
            {
                "update_rule": "fedavg",
                "update_mode": mode,
                "frozen_gradient_weighting": "examples",
                "max_grad_norm": 0.5,
            },
        )
    yield "local_sgd", {"momentum": 0.9, "nesterov": True, "weight_decay": 0.01}
    yield (
        "local_adamw",
        {
            "update_rule": "local_adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.99,
            "epsilon": 1e-8,
            "momentum": None,
            "nesterov": None,
        },
    )
    yield "fedprox", {"update_rule": "fedprox", "proximal_mu": 0.1}
    yield "scaffold", {"update_rule": "scaffold"}


def classification_rule_config(client: dict[str, Any], *, cnn: bool = False) -> dict[str, Any]:
    """The synthetic run with ``client``'s rule; a rule drops the engine options it refuses."""

    config = (cnn_config if cnn else classification_config)()
    rule = client.get("update_rule", "local_sgd")
    refused = {
        "local_adamw": ("momentum", "nesterov"),
        "fedprox": ENGINE_OPTIONS,
        "scaffold": ENGINE_OPTIONS,
    }.get(rule, ())
    for option in refused:
        config["client"].pop(option, None)
    config["client"].update({key: value for key, value in client.items() if value is not None})
    if rule == "scaffold":
        config["server"]["strategy"] = "scaffold"
    return config


def ragged_classification() -> Any:
    """Train splits of 20, 17, 14, 11 and 8 rows, by client number."""

    from fedbrew.data.synthetic_classification import SyntheticClassificationDataset as Data

    real = Data.get_client_data

    def truncated(self: Any, client_id: str) -> dict[str, Any]:
        data = real(self, client_id)
        keep = 20 - 3 * (int(client_id.rsplit("_", 1)[-1]) % 5)
        train = {key: value[:keep] for key, value in data["train"].items()}
        return {**data, "train": train, "num_train_examples": keep}

    return mock.patch.object(Data, "get_client_data", truncated)


class ClassificationFloat64Test(ExecutorRuns):
    """Stage c: the MLP and the CNN through the shipped task, in float64, to TOLERANCE."""

    def test_the_mlp_under_every_rule(self) -> None:
        for label, client in classification_arms():
            with self.subTest(rule=label):
                self.assertAgree(
                    *self.both(classification_rule_config(client), data=float64_classification)
                )

    def test_the_mlp_on_ragged_clients(self) -> None:
        @contextmanager
        def data() -> Iterator[None]:
            with float64_classification(), ragged_classification():
                yield

        for label, client in classification_arms():
            with self.subTest(rule=label):
                self.assertAgree(*self.both(classification_rule_config(client), data=data))

    def test_the_cnn(self) -> None:
        @contextmanager
        def data() -> Iterator[None]:
            with float64_classification(), images():
                yield

        for label, client in classification_arms():
            if label not in {"fedavg/sequential_epoch", "fedavg/full_gradient", "scaffold"}:
                continue
            with self.subTest(rule=label):
                self.assertAgree(
                    *self.both(classification_rule_config(client, cnn=True), data=data)
                )


class ClassificationFloat32Test(ExecutorRuns):
    """The task as it ships, in float32: to FLOAT32_TOLERANCE, and exactly with one client."""

    def test_the_mlp_and_the_cnn_one_client_per_chunk_are_bit_identical(self) -> None:
        for label, client in classification_arms():
            with self.subTest(model="mlp", rule=label):
                batched, sequential = self.both(
                    classification_rule_config(client), executor_chunk_bytes=1
                )
                self.assertAgree(batched, sequential, exact=True)
        with self.subTest(model="cnn"):
            config = classification_rule_config({"update_mode": "sequential_epoch"}, cnn=True)
            batched, sequential = self.both(config, data=images, executor_chunk_bytes=1)
            self.assertAgree(batched, sequential, exact=True)

    def test_the_mlp_within_the_float32_bound(self) -> None:
        for label, client in classification_arms():
            with self.subTest(rule=label):
                batched, sequential = self.both(classification_rule_config(client))
                self.assertAgree(batched, sequential, tolerance=FLOAT32_TOLERANCE)


class BatchedRunsAreDeterministicTest(ExecutorRuns):
    def test_two_batched_runs_are_identical(self) -> None:
        config = with_client(example_config("fed-lasso"), update_mode="single_batch")
        with ragged_clients():
            first = self.run_config(config, "batched")
            second = self.run_config(config, "batched")
        self.assertAgree(first, second, exact=True)


if __name__ == "__main__":
    unittest.main()
