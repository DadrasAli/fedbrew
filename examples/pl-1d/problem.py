"""A one-dimensional Polyak-Lojasiewicz objective, run through the real loop.

The problem
-----------
The global objective is the non-convex function that satisfies the PL inequality
in Karimi, Nutini and Schmidt (arXiv:1608.04636, section 2.2)::

    F(x) = x^2 + 3 sin^2(x)          F'(x) = 2x + 3 sin(2x)

`F` is not convex -- `F''(x) = 2 + 6 cos(2x)` is negative wherever
`cos(2x) < -1/3` -- but `F'` has exactly one zero, at `x = 0`, so

    x* = 0        F* = 0

both known in closed form, and `F` satisfies the PL inequality
`(1/2) |F'(x)|^2 >= mu (F(x) - F*)` with `mu = 1/32`.

Client `i` gets its own objective, tilted by a per-client `shift` `s_i`::

    f_i(x) = F(x) + s_i x            f_i'(x) = F'(x) + s_i

The shifts are laid out in exact +/- pairs, so they sum to zero in floating
point and the uniform mean over clients is `F` itself -- which is what keeps
`x*` and `F*` exact for the federated problem and not only for one client.
Client `i`'s own minimiser solves `F'(x) = -s_i`, so it is *not* at 0: the
clients genuinely disagree, and the disagreement is a knob (`shift_scale`).

There is no data. A "batch" carries one number, `s_i`, and the gradient is the
analytic expression above, returned by a custom autograd Function so that every
shipped client rule gets the same exact gradient, whether it calls
``TaskAdapter.train_step`` or drives ``loss.backward()`` through its own
optimizer wrapper.

What this file registers
------------------------
Three names, at ``register()`` time, and nothing at import:

    generators  "pl_1d"      writes the shards and the manifest
    tasks       "pl_1d"      PL1DTask
    models      "pl_scalar"  a single-parameter nn.Module

There is no dataset backend. The shifts are *generated* -- from the spec in a
generator config, rather than from a corpus -- and a run reads them through
the shipped ``manifest_dataset`` like every other run in the repository. That
is the whole reason this example no longer needs a ``run.py`` that imports
itself: ``fedbrew generate`` writes the data, ``fedbrew run`` reads it, and
neither knows this problem is analytic.

The reference optimum travels with the data. `x*`, `F*`, `mu` and the shifts
themselves are properties of the generated shards, so the generator writes
them into the manifest under ``reference``; the task reads them from
``dataset_metadata`` and ``run.json`` records them, which is how a run says
what it was scored against.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, optim

from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import save_client_shard, save_split_client_shard
from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.tasks.base import TaskAdapter

#: The optimum, in closed form. Both are exact for the *federated* objective,
#: not only for one client, because the shifts sum to zero.
X_STAR = 0.0
F_STAR = 0.0

#: The PL constant of F, from Karimi et al. Not used by the run; recorded so a
#: rate can be checked against the theory it is supposed to match. It is a
#: valid constant, not the tight one: minimising |F'(x)|^2 / (2(F(x) - F*))
#: over [-8, 8] gives about 0.1755 at x ~ +/-2.2, so a rate checked against
#: this number is conservative by roughly 5.6x. Nothing asserts either
#: value; the citation is what is recorded.
PL_MU = 1.0 / 32.0

#: Every tensor here is float64. The point of the example is to watch
#: F(x) - F* fall by ten orders of magnitude; in float32 it would stop at the
#: precision floor around 1e-7 and the curve would read as a plateau in the
#: algorithm rather than in the arithmetic.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def objective(x: float, shift: float = 0.0) -> float:
    """`f_i(x) = x^2 + 3 sin^2(x) + shift * x`, as a plain float."""

    return x * x + 3.0 * math.sin(x) ** 2 + shift * x


def gradient(x: float, shift: float = 0.0) -> float:
    """`f_i'(x) = 2x + 3 sin(2x) + shift`, as a plain float.

    The analytic derivative of :func:`objective`, differentiated by hand:
    `d/dx [3 sin^2 x] = 6 sin x cos x = 3 sin 2x`. :class:`_PLObjective`
    returns exactly this from its backward pass, so nothing in the run relies
    on autograd rediscovering it.
    """

    return 2.0 * x + 3.0 * math.sin(2.0 * x) + shift


def optimality_gap(x: float) -> float:
    """`F(x) - F*`, the quantity a PL rate bounds."""

    return objective(x) - F_STAR


def distance_to_optimum(x: float) -> float:
    """`||x - x*||`, which for a scalar iterate is `|x|`."""

    return abs(x - X_STAR)


class _PLObjective(torch.autograd.Function):  # type: ignore[misc]
    """`f_i(x)` forward, hand-written `f_i'(x)` backward.

    Written as an autograd Function rather than left to autograd on the closed
    form so that the gradient every rule sees is the analytic one, bit for bit,
    on every path into the model: ``train_step``, and the SCAFFOLD, FedProx and
    Delta-SGD paths that call ``backward()`` on their own.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, shifts: Tensor) -> Tensor:
        """Return `f_i(x)` for each shift in the batch; shape `(N,)`."""

        ctx.save_for_backward(x, shifts)
        return x * x + 3.0 * torch.sin(x) ** 2 + shifts * x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        """Return `sum_j grad_output_j * f_j'(x)`, summed over the batch."""

        x, shifts = ctx.saved_tensors
        per_row = 2.0 * x + 3.0 * torch.sin(2.0 * x) + shifts
        return (grad_output * per_row).sum().reshape(x.shape), None


# ---------------------------------------------------------------------------
# The model: one parameter
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("x_init",)


class PLScalarModel(nn.Module):  # type: ignore[misc]
    """The iterate `x`, as a one-element `nn.Parameter`.

    A model with no data-dependent structure at all. All that makes it a legal
    model here is that its state dict is a mapping of tensors, which is what
    ``WeightedStateAccumulator`` and the federated-state group require.
    """

    def __init__(self, x_init: float = 2.5) -> None:
        """Place the iterate at `x_init`.

        Args:
            x_init: Starting point, identical on every client because the
                server broadcasts its own initial state before round 1.
        """

        super().__init__()
        self.x = nn.Parameter(torch.tensor([float(x_init)], dtype=DTYPE))

    def forward(self, shifts: Tensor) -> Tensor:
        """Return `f_i(x)` per row of `shifts`."""

        return _PLObjective.apply(self.x, shifts)

    @property
    def iterate(self) -> float:
        """The scalar the whole example is about."""

        return float(self.x.detach().reshape(()))


def build_pl_scalar(config: Mapping[str, Any] | None = None) -> PLScalarModel:
    """Registry builder for `model.name: pl_scalar`.

    The model block carries the starting point and nothing about the problem:
    `x*` and `F*` are fixed by the objective, and the shifts are the data. So
    there is nothing here to cross-check against the manifest, and the task
    does not either.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _MODEL_KEYS, "pl_scalar")
    return PLScalarModel(x_init=float(values.get("x_init", 2.5)))


# ---------------------------------------------------------------------------
# The problem: one shift per client
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """The whole problem, as two numbers."""

    num_clients: int = 8
    #: Largest per-client tilt. 0.0 makes every client's objective the global
    #: one, which is the homogeneous control.
    shift_scale: float = 2.0

    def __post_init__(self) -> None:
        """Refuse a spec that cannot express disagreement."""

        if self.num_clients < 2:
            raise ValueError("partition.num_clients must be at least 2")
        if self.shift_scale < 0.0:
            raise ValueError("problem.shift_scale must be non-negative")

    def shifts(self) -> list[float]:
        """The per-client tilts, in exact +/- pairs so they sum to zero.

        Built as pairs rather than as `scale * (2i/(n-1) - 1)` because the
        federated objective is only `F` if the mean shift is *exactly* zero;
        the linear form leaves a residue of a few ulp, which puts `x*` and
        `F*` a little off the values this module documents as exact.
        """

        half = self.num_clients // 2
        values: list[float] = []
        for index in range(half):
            magnitude = self.shift_scale * (index + 1) / half
            values.extend((-magnitude, magnitude))
        if self.num_clients % 2:
            values.append(0.0)
        return values


# ---------------------------------------------------------------------------
# The generator: the shifts, written as shards a run reads like any other
# ---------------------------------------------------------------------------

#: The name of this problem's generator and task, as configs write it.
DATASET_NAME = "pl_1d"


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """What ``fedbrew generate`` prints when this generator finishes."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


def _spec_from_config(config: Mapping[str, Any]) -> ProblemSpec:
    """Read the spec out of a generator config's ``problem`` section.

    ``partition.num_clients`` rather than a ``problem.clients`` of our own:
    the client count is the one dial every generator states in the same
    place, and stating it twice is how two spellings of one number drift
    apart.
    """

    problem = dict(config.get("problem", {}))
    partition = dict(config.get("partition", {}))
    if "num_clients" not in partition:
        raise ValueError("pl_1d needs partition.num_clients: it is the number of shifts")
    return ProblemSpec(
        num_clients=int(partition["num_clients"]),
        shift_scale=float(problem.get("shift_scale", 2.0)),
    )


def reference_of(spec: ProblemSpec) -> dict[str, Any]:
    """Everything a run on this data is scored against, in closed form.

    Written into the manifest, copied into ``run.json`` by
    ``run_metadata.build_dataset_provenance``, and read back by
    :class:`PL1DTask`. The shifts are here as well as in the shards because
    they are the whole of what makes the clients differ, and a reader of
    ``run.json`` should not need to open a shard to see them.
    """

    return {
        "problem": {"clients": spec.num_clients, "shift_scale": spec.shift_scale},
        "shifts": spec.shifts(),
        "x_star": X_STAR,
        "f_star": F_STAR,
        "pl_mu": PL_MU,
    }


def generate_pl_1d_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, plus the manifest a run reads.

    The four arguments are the generator contract (chapter 12 §4). Two of
    them do nothing here, and say so rather than being quietly dropped:

    ``seed``
        The shifts are a deterministic function of the spec -- exact +/-
        pairs -- so there is no draw for a seed to fix. Recorded in the
        manifest anyway, because "this data was generated at seed 42" is what
        every other dataset's provenance says and an absent seed would read
        as an omission.

    ``client_splits``
        The ratios describe a cut, and there is nothing to cut: `f_i` is not
        estimated from samples, it *is* the client. All three splits hold the
        same single row, which the manifest declares as
        ``client_test_source: identical_to_train`` so that preflight tells a
        reader their ``test_*`` and ``central_test_*`` numbers are training
        numbers.
    """

    del client_splits
    spec = _spec_from_config(config)
    shifts = spec.shifts()
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    clients: list[dict[str, Any]] = []
    for index, shift in enumerate(shifts):
        client_id = f"client_{index}"
        # One row of one feature: the shift. The second column is a dummy
        # target nothing reads -- f_i needs no label -- but a shard is an
        # (x, y) pair everywhere else, and manifest_validation checks for both.
        row = torch.tensor([[shift]], dtype=DTYPE)
        target = torch.zeros(1, dtype=DTYPE)
        save_split_client_shard(
            shards_dir / f"{client_id}.pt", row, target, row, target, row, target
        )
        clients.append(
            {
                "client_id": client_id,
                "shard": f"shards/{client_id}.pt",
                "num_examples": 3,
                "num_train_examples": 1,
                "num_eval_examples": 1,
                "num_test_examples": 1,
                "shift": shift,
            }
        )

    # The server's central pass evaluates F in one go, so the pooled shard is
    # every client's shift stacked -- which is exactly the federated
    # objective, because every client carries the same weight.
    save_client_shard(
        shards_dir / "global_test.pt",
        torch.tensor(shifts, dtype=DTYPE).reshape(-1, 1),
        torch.zeros(len(shifts), dtype=DTYPE),
    )
    _write_partition_stats(output_dir, spec, clients)

    manifest = {
        "dataset_name": DATASET_NAME,
        "format": "torch_shards",
        "client_shard_format": "split_v2",
        "client_test_source": IDENTICAL_TO_TRAIN,
        "num_clients": len(clients),
        "input_dim": 1,
        "clients_file": "clients.jsonl",
        "shards_dir": "shards",
        "global_test": "shards/global_test.pt",
        "partition_stats_file": "partition_stats.json",
        "client_stats_file": "client_stats.csv",
        "partition_strategy": "analytic",
        "partition_key": "shift_index",
        "seed": seed,
        "reference": reference_of(spec),
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients)
    return GenerationSummary(
        manifest_path=manifest_path,
        num_clients=len(clients),
        num_examples=len(clients),
        num_test_examples=len(clients),
    )


def _write_partition_stats(
    output_dir: Path,
    spec: ProblemSpec,
    clients: Sequence[Mapping[str, Any]],
) -> None:
    """Write the two files ``fedbrew inspect-data`` reads beside the manifest.

    No label counts: an objective has no labels, and an empty
    ``global_label_counts`` would read as a partition with none rather than a
    problem with none.
    """

    payload = {
        "dataset_name": DATASET_NAME,
        "partition_strategy": "analytic",
        "num_clients": len(clients),
        "total_examples": sum(int(client["num_examples"]) for client in clients),
        "min_examples_per_client": 3,
        "max_examples_per_client": 3,
        "mean_examples_per_client": 3.0,
        "shift_scale": spec.shift_scale,
        "clients": [dict(client) for client in clients],
    }
    # allow_nan=False: the shifts are measured values, and Python's json
    # writes a non-finite one as the bare token NaN, which is not JSON and
    # which the next reader chokes on rather than noticing.
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "client_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", "num_examples", "shift"])
        for client in clients:
            writer.writerow(
                [client["client_id"], client["num_examples"], f"{client['shift']:.17g}"]
            )


# ---------------------------------------------------------------------------
# The task adapter
# ---------------------------------------------------------------------------


class PL1DTask(TaskAdapter):
    """Bridge between the scalar objective and the generic FL orchestration.

    Implements the five abstract methods, plus ``evaluate_model`` for the
    server's central pass. The three attributes below the constructor's
    docstring are not part of ``TaskAdapter``: they exist because client rules
    used to read them off the task directly, and one still does.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
        device: str = "cpu",
        **unused: Any,
    ) -> None:
        """Build the adapter.

        The arguments are the task contract (chapter 12 §1.4). Only ``device``
        is used: the model block carries the starting point and nothing about
        the problem, and the reference optimum is a constant of the objective
        rather than of the data, so there are no two halves to cross-check.
        The rest -- ``batch_size``, ``eval_batch_size``, ``dataloader_config``,
        ``reuse_model`` -- arrive in ``unused`` and are ignored, because a
        scalar has no throughput question and the loader is built per call.

        Args:
            model_config: The resolved ``model`` block. Carries `x_init`.
            dataset_metadata: The manifest, as ``manifest_dataset`` reports
                it. Carries ``reference``, written by the generator.
            device: The resolved ``runtime.device``.
            **unused: The rest of the contract.
        """

        del unused, model_config
        self.device = torch.device(device)
        self.reference = dict((dataset_metadata or {}).get("reference") or {})

        # --- this task's own helpers, not part of TaskAdapter ---
        #
        # `_criterion` and `_move_batch` are used by `train_step` and
        # `eval_step` below and by nothing outside this class. They carry the
        # classification task's private names because SCAFFOLD and FedProx
        # used to reach through the adapter and call them directly instead of
        # calling `train_step`; both now wrap the optimizer and go through
        # `train_step` like every other rule, so the names are free. Kept as
        # they are because they read fine and renaming them proves nothing.
        # `_criterion` ignores its second argument.
        self._criterion = _mean_of_batch
        # Delta-SGD, FedLALR, SCAFFOLD and FedProx read this to decide whether
        # to refuse `runtime.use_amp: true`. They read it with
        # `getattr(task, "_scaler", None)`, so a task that never defines it is
        # refused nothing and this line is not required. It is here to put the
        # answer where a reader looking for it will find it.
        self._scaler: Any = None

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> PLScalarModel:
        """Build the scalar model named by the `model` config block."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        factory = models.get(str(values.get("name", "pl_scalar")))
        model = factory(values)
        return model.to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Cut one client's shifts into batches.

        The shifts arrive as a shard's ``x``, because that is what
        ``manifest_dataset`` hands over and what every other dataset in the
        repository stores.

        A list, not a generator: ``single_batch`` re-iterates the loader when
        it runs out, so the loader has to be re-iterable.

        The second element of each batch is a dummy target. Nothing reads it --
        `f_i` needs no label -- but every client rule destructures a batch into
        two, so a one-element batch would fail in five places.
        """

        loader_config = _loader_config(config)
        shifts = _shifts_of(data).to(self.device)
        batch_size = max(1, int(loader_config.get("batch_size", len(shifts)) or len(shifts)))
        if bool(loader_config.get("shuffle", False)):
            shifts = shifts[_permutation(len(shifts), loader_config.get("seed"))]
        batches = [
            shifts[start : start + batch_size] for start in range(0, len(shifts), batch_size)
        ]
        if bool(loader_config.get("drop_last", False)) and len(batches) > 1:
            batches = [batch for batch in batches if len(batch) == batch_size]
        return [(batch, torch.zeros_like(batch)) for batch in batches]

    def train_step(
        self,
        model: PLScalarModel,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Take one optimizer step on the batch's mean objective."""

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        shifts, targets = self._move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = self._criterion(model(shifts), targets)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def eval_step(self, model: PLScalarModel, batch: Any) -> dict[str, float]:
        """Measure the batch's mean objective, and the iterate it was measured at."""

        model.eval()
        shifts, _ = self._move_batch(batch)
        with torch.no_grad():
            values = model(shifts)
        return {
            "loss": float(values.mean()),
            "total": float(shifts.numel()),
            # Carried per batch because compute_metrics is handed the outputs
            # and nothing else, and two of the three metrics it returns are
            # functions of the iterate rather than of the data.
            "iterate": model.iterate,
        }

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Fold eval-step outputs into the three numbers this task reports.

        ``loss``
            The example-weighted mean of `f_i(x)` over whatever was evaluated.
            On the central pass that is every client, so it *is* `F(x)`, and
            since `F* = 0` it is also the optimality gap. On a client pass it
            is that one client's tilted objective.
        There is deliberately no ``accuracy``. A scalar objective has none,
        and this task used to report a hit indicator -- 1.0 inside a
        ``tolerance`` ball around `x*` -- only because three places in the
        loop demanded a number every task could not have. They no longer do,
        so the indicator and the ``model.tolerance`` key that fed it are both
        gone. README, "The metric surface".

        ``optimality_gap`` / ``distance_to_optimum``
            `F(x) - F*` and `||x - x*||`, both functions of the iterate alone
            and both measured against the *global* objective, whichever client
            happens to be reporting them.
        """

        records = [record for record in outputs if isinstance(record, Mapping)]
        if not records:
            return {"loss": 0.0, "optimality_gap": 0.0, "distance_to_optimum": 0.0}

        total = sum(float(record.get("total", 0.0)) for record in records)
        if total:
            loss = (
                sum(
                    float(record.get("loss", 0.0)) * float(record.get("total", 0.0))
                    for record in records
                )
                / total
            )
        else:
            loss = sum(float(record.get("loss", 0.0)) for record in records) / len(records)

        # Every record was produced by the same model in the same pass, so any
        # of them carries the iterate; the last is as good as the first.
        iterate = float(records[-1].get("iterate", math.nan))
        return {
            "loss": loss,
            "optimality_gap": optimality_gap(iterate),
            "distance_to_optimum": distance_to_optimum(iterate),
        }

    # -- the server's central pass -----------------------------------------

    def evaluate_model(self, model: PLScalarModel, data: Any) -> dict[str, float]:
        """Measure the global model on every client's objective at once.

        ``FedAvgServer.evaluate_global`` prefixes what this returns with
        ``global_`` and hands it to ``loop._evaluate_central_test_set``, which
        passes through every finite numeric key it is given. So
        ``central_test_loss`` is `F(x) - F*` for the aggregated iterate -- the
        curve this example exists to draw -- and ``optimality_gap`` and
        ``distance_to_optimum`` arrive beside it as
        ``central_test_optimality_gap`` and
        ``central_test_distance_to_optimum``.
        """

        outputs = [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        return self.compute_metrics(outputs)

    # -- this task's own batch splitter -------------------------------------

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Split a batch into (shifts, ignored targets), both on the device."""

        shifts, targets = batch
        return shifts.to(self.device), targets.to(self.device)


def _mean_of_batch(outputs: Tensor, targets: Tensor | None = None) -> Tensor:
    """The batch objective: the mean of `f_i(x)` over the batch's rows."""

    del targets
    return outputs.mean()


def _shifts_of(data: Any) -> Tensor:
    """The shifts in one split of a shard, as a flat `(N,)` vector.

    A shard row is one feature wide, so the stored ``x`` is `(N, 1)`; the
    objective broadcasts a scalar iterate against a flat vector of shifts.
    """

    if isinstance(data, Mapping):
        for key in ("x", "X", "shifts"):
            value = data.get(key)
            if isinstance(value, Tensor):
                return value.to(DTYPE).reshape(-1)
    if isinstance(data, Tensor):
        return data.to(DTYPE).reshape(-1)
    raise ValueError("pl_1d data must be a shard mapping carrying an 'x' tensor")


def _loader_config(config: Mapping[str, Any] | bool | None) -> dict[str, Any]:
    if isinstance(config, bool):
        return {"shuffle": config}
    return dict(config or {})


def _permutation(count: int, seed: Any) -> Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.randperm(count, generator=generator)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: The three registry names this example adds. ``DATASET_NAME`` above is the
#: generator's; the task shares the name, because a run config's
#: ``dataset.name`` and its task are the same problem seen from two commands.
TASK_NAME = "pl_1d"
MODEL_NAME = "pl_scalar"


def register() -> None:
    """Register the generator, the task and the model.

    Called once by ``fedbrew.core.extensions``, for the entry a config names
    under ``experiment.extensions`` or ``dataset.extensions``. Nothing is
    registered at import, so this module can be imported for
    :class:`ProblemSpec` alone, without touching the registries.
    """

    from fedbrew.core import registry

    registry.generators.register(
        DATASET_NAME,
        generate_pl_1d_from_config,
        sections={"problem": {"shift_scale"}},
    )
    registry.tasks.register(TASK_NAME, lambda **kwargs: PL1DTask(**kwargs))
    registry.models.register(MODEL_NAME, build_pl_scalar, task=TASK_NAME)


def _self_check() -> None:
    """Assert the three claims this module's docstring makes about the problem.

    Cheap enough to run at import: it is four evaluations and one backward
    pass. Each is load-bearing -- if the shifts do not sum to zero the
    federated optimum is not at 0, if the manifest's reference is not this
    module's optimum then every gap is measured from the wrong place, and if
    the backward pass disagrees with :func:`gradient` then the run is not
    solving the documented problem.
    """

    spec = ProblemSpec()
    if sum(spec.shifts()) != 0.0:
        raise AssertionError("shifts must sum to exactly zero")
    if reference_of(spec)["f_star"] != F_STAR:
        raise AssertionError("the manifest's reference optimum is not this module's")

    model = PLScalarModel(x_init=1.3)
    shifts = torch.tensor([-0.7, 0.4], dtype=DTYPE)
    _mean_of_batch(model(shifts)).backward()
    measured = model.x.grad
    if measured is None:
        raise AssertionError("the backward pass left no gradient on the iterate")
    expected = sum(gradient(1.3, float(shift)) for shift in shifts) / len(shifts)
    if abs(float(measured.reshape(())) - expected) > 1e-15:
        raise AssertionError("autograd disagrees with the analytic gradient")


_self_check()
