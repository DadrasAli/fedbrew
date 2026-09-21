"""Least squares over the probability simplex, run through the real loop.

The problem
-----------
Client `i` holds an `m x d` design `H` and its own targets `y_i`, and the
federated problem is a *constrained* one::

    minimise   F(x) = mean_i (1/2m) ||H x - y_i||^2
    subject to x in Delta = { x : x >= 0, sum_j x_j = 1 }

`H` is the `d x d` Sylvester-Hadamard matrix with `m = d` rows, so `H^T H = m I`
exactly and the smooth part reduces, exactly, to `(1/2) ||x - theta_i||^2` with
`theta_i = H^T y_i / m` client `i`'s ordinary-least-squares solution. The
`theta_i` are laid out around a pooled `theta_bar` in exact +/- pairs, so::

    F(x) = (1/2) ||x - theta_bar||^2 + zeta^2 / 2

The unconstrained minimiser is `theta_bar`, and the whole point of the example
is that **`theta_bar` is not in the simplex**. The ``infeasibility`` dial puts it
outside: at the default it has negative coordinates and sums to 1.5. Because the
design is orthonormal, the constrained minimiser is then exactly the Euclidean
projection::

    x* = Proj_Delta(theta_bar)        F* = (1/2) ||x* - theta_bar||^2 + zeta^2/2

and :func:`project_simplex` computes it in `O(d log d)` by sorting -- a finite,
exact algorithm, not an iteration. So `x*` and `F*` are known before the run.

At the default dials `x*` is exactly `x_true`, the planted point: the
displacement is constant on the even coordinates, the planted support is even,
and the projection's threshold lands on exactly that constant. So this example
has one ground truth where ``examples/fed-lasso`` has two, and
``distance_to_optimum`` is also the distance to the planted signal.
``_self_check`` asserts it rather than leaving it to this paragraph.

What goes wrong, and why the gap goes negative
-----------------------------------------------
No shipped algorithm can respect a constraint. There is no projection step, no
Frank-Wolfe step and no mirror map anywhere in fedbrew, so every arm minimises
`F` over all of `R^d` and converges neatly to `theta_bar`, which is not a
solution. The failure is not slow convergence and not divergence -- the arms
converge, and they converge to the wrong set.

The consequence for a results table is worth stating in one line, because it is
the shape of a mistake rather than a subtlety:

    F(theta_bar) - F* = -(1/2) ||x* - theta_bar||^2 < 0

**Every arm ends with a negative optimality gap.** Read as a column of numbers,
they beat the optimum. They beat it by leaving the feasible set, and a table
that reports the objective without reporting the constraint says the opposite of
what happened. This module therefore reports the violation beside the objective
in every pass -- ``constraint_violation``, ``simplex_sum``, ``min_coordinate`` --
and a ``feasible_gap`` that projects the iterate first and is the number a
reader actually wants.

What this file registers
------------------------
Three names, at ``register()`` time, and nothing at import:

    generators  "simplex_lsq"     writes the shards and the manifest
    tasks       "simplex_lsq"     SimplexLSQTask
    models      "simplex_vector"  SimplexModel, a d-vector on the simplex

There is no dataset backend. The design and the targets are *generated* -- from
the spec in a generator config -- and a run reads them through the shipped
``manifest_dataset`` like every other run in the repository. The reference
optimum travels with the data: `x*`, `F*` and the negative floor it implies are
properties of the generated shards, so the generator writes them into the
manifest under ``reference``, the task reads them from ``dataset_metadata``,
and ``run.json`` records them.
"""

from __future__ import annotations

import csv
import json
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

#: float64 throughout: `constraint_violation` and `simplex_sum - 1` are the two
#: numbers the example is about, and both are exactly 0 at initialisation. In
#: float32 the sum would drift off 1 by ~1e-7 from rounding alone and the
#: violation column would report the arithmetic rather than the algorithm.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# The feasible set
# ---------------------------------------------------------------------------


def project_simplex(values: Tensor) -> Tensor:
    """Euclidean projection onto `{x >= 0, sum x = 1}`.

    The sort-and-threshold algorithm (Duchi et al., ICML 2008, Figure 1): sort
    descending, find the largest `rho` with
    `u_rho - (cumsum_rho - 1)/rho > 0`, and shift by that threshold. Finite and
    exact -- `O(d log d)` and no iteration -- which is what lets this example
    state its optimum in closed form.

    This is the operator the problem needs and the repository does not have.
    It is here to *define* `x*`, not to run: nothing under ``fedbrew/clients/``
    or ``fedbrew/servers/`` projects anything, and this example adds no client
    rule that would.
    """

    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    counts = torch.arange(1, len(values) + 1, dtype=values.dtype, device=values.device)
    candidates = sorted_values - (cumulative - 1.0) / counts
    support = int((candidates > 0).sum())
    threshold = (cumulative[support - 1] - 1.0) / support
    return torch.clamp(values - threshold, min=0.0)


def constraint_violation(values: Tensor) -> float:
    """`||x - Proj_Delta(x)||`: how far the iterate is from feasible.

    Zero if and only if `x` is in the simplex. Reported every round because the
    objective alone cannot say whether an iterate is a candidate solution.
    """

    return float(torch.linalg.vector_norm(values - project_simplex(values)))


def negative_mass(values: Tensor) -> float:
    """`sum_j max(0, -x_j)`: how much probability mass sits below zero."""

    return float(torch.clamp(-values, min=0.0).sum())


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def objective(x: Tensor, features: Tensor, targets: Tensor) -> float:
    """`f_i(x) = (1/2m) ||H x - y_i||^2`, as a plain float."""

    residual = features @ x - targets
    return 0.5 * float(residual @ residual) / len(targets)


def gradient(x: Tensor, features: Tensor, targets: Tensor) -> Tensor:
    """`(1/m) H^T (H x - y_i)`, the analytic gradient.

    A gradient, not a subgradient: the objective is smooth, and everything
    non-smooth about this problem is in the feasible set, which no arm sees.
    `_self_check` verifies autograd returns exactly this.
    """

    residual = features @ x - targets
    return features.T @ residual / len(targets)


def _hadamard_row(index: int, dim: int) -> list[float]:
    """Row `index` of the Sylvester-Hadamard matrix, as +/-1 entries."""

    return [-1.0 if bin(index & column).count("1") % 2 else 1.0 for column in range(dim)]


def hadamard_matrix(dim: int) -> Tensor:
    """The `dim x dim` Sylvester-Hadamard matrix; `H^T H = dim I` exactly."""

    return torch.tensor([_hadamard_row(row, dim) for row in range(dim)], dtype=DTYPE)


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("x_init",)


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """The whole problem, as four numbers, plus every reference it implies."""

    num_clients: int = 8
    #: `d`, and also `m`: the design is the full `d x d` Hadamard matrix. A
    #: power of two, so the design is orthogonal and `x*` is a projection.
    dim: int = 16
    #: How far the *unconstrained* optimum sits outside the simplex. 0.0 leaves
    #: `theta_bar` at the planted point, which is already feasible, and then the
    #: constraint does nothing and no arm is wrong -- that is the control.
    infeasibility: float = 0.5
    #: `zeta`: the per-client spread of `theta_i` about `theta_bar`. The mean is
    #: exact regardless, so this moves the clients apart without moving `x*`.
    heterogeneity: float = 0.4

    def __post_init__(self) -> None:
        """Refuse a spec that cannot express what it claims to."""

        if self.num_clients < 2:
            raise ValueError("partition.num_clients must be at least 2")
        if self.dim < 4 or self.dim & (self.dim - 1):
            raise ValueError("problem.dim must be a power of two of at least 4")
        if self.infeasibility < 0.0:
            raise ValueError("problem.infeasibility must be non-negative")
        if self.heterogeneity < 0.0:
            raise ValueError("problem.heterogeneity must be non-negative")
        if self.num_clients // 2 > self.dim - 2:
            raise ValueError(
                f"{self.num_clients} clients need {self.num_clients // 2} offset "
                f"directions, and d={self.dim} supplies {self.dim - 2} once the "
                "row that displaces theta_bar is taken."
            )

    # -- the ground truth ---------------------------------------------------

    def truth(self) -> Tensor:
        """`x_true`: the planted point of the simplex.

        Mass `1/2, 1/4, 1/8, 1/8` on four evenly spaced coordinates. Every
        value is a negative power of two, so the coordinates sum to exactly 1.0
        in floating point and the planted point is exactly feasible -- which is
        what makes `constraint_violation` at initialisation a meaningful zero
        rather than a rounding residue.
        """

        values = torch.zeros(self.dim, dtype=DTYPE)
        weights = (0.5, 0.25, 0.125, 0.125)
        for order, weight in enumerate(weights):
            values[(order * self.dim) // len(weights)] = weight
        return values

    # -- the data -----------------------------------------------------------

    def design(self) -> Tensor:
        """`H`, shared by every client. `H^T H = m I` exactly."""

        return hadamard_matrix(self.dim)

    def displacement(self) -> Tensor:
        """What moves `theta_bar` off the simplex, scaled by ``infeasibility``.

        Hadamard row 1 -- half `+1`, half `-1` -- plus a uniform term. The first
        drives coordinates negative, the second moves the sum: `sum theta_bar`
        is exactly `1 + infeasibility`. Both constraints are violated, so
        neither ``min_coordinate`` nor ``simplex_sum`` is a column that happens
        to look fine.
        """

        alternating = torch.tensor(_hadamard_row(1, self.dim), dtype=DTYPE)
        uniform = torch.ones(self.dim, dtype=DTYPE) / self.dim
        return self.infeasibility * (alternating / (self.dim**0.5) + uniform)

    def client_offsets(self) -> Tensor:
        """The per-client offsets, shape `(n, d)`, in exact +/- pairs.

        Hadamard rows from 2 up, so they are orthogonal to each other and to
        the row that displaces `theta_bar`; the pairing is what makes their sum
        exactly zero and `theta_bar` -- and so `x*` and `F*` -- exact.
        """

        scale = self.heterogeneity / (self.dim**0.5)
        rows: list[list[float]] = []
        for pair in range(self.num_clients // 2):
            direction = [scale * sign for sign in _hadamard_row(pair + 2, self.dim)]
            rows.append(direction)
            rows.append([-value for value in direction])
        if self.num_clients % 2:
            rows.append([0.0] * self.dim)
        return torch.tensor(rows, dtype=DTYPE)

    def pooled_ols(self) -> Tensor:
        """`theta_bar = x_true + displacement`: the unconstrained optimum."""

        return self.truth() + self.displacement()

    def client_ols(self) -> Tensor:
        """The `theta_i = theta_bar + u_i`, shape `(n, d)`."""

        return self.pooled_ols() + self.client_offsets()

    def client_targets(self) -> Tensor:
        """The `y_i = H theta_i`, shape `(n, m)`. This is the stored data."""

        return self.client_ols() @ self.design().T

    # -- the reference optimum ----------------------------------------------

    def optimum(self) -> Tensor:
        """`x* = Proj_Delta(theta_bar)`, the exact constrained minimiser."""

        return project_simplex(self.pooled_ols())

    def objective_at(self, x: Tensor) -> float:
        """`F(x)`, computed from the stored data rather than the reduced form."""

        features = self.design()
        targets = self.client_targets()
        return sum(objective(x, features, row) for row in targets) / len(targets)

    def optimal_objective(self) -> float:
        """`F* = F(x*)`: the best value any *feasible* point attains."""

        return self.objective_at(self.optimum())

    def unconstrained_objective(self) -> float:
        """`F(theta_bar)`: what an arm that ignores the constraint converges to."""

        return self.objective_at(self.pooled_ols())

    def negative_gap_floor(self) -> float:
        """`F(theta_bar) - F*`, which is negative. What the gap column reaches.

        This is the number a results table would report as an arm beating the
        optimum. It equals `-(1/2) ||x* - theta_bar||^2` up to the rounding in
        the materialised targets.
        """

        return self.unconstrained_objective() - self.optimal_objective()

    def heterogeneity_residual(self) -> float:
        """`zeta^2 / 2`: the part of `F*` no `x` removes, feasible or not."""

        offsets = self.client_offsets()
        return 0.5 * float((offsets * offsets).sum(dim=1).mean())

    def optimum_support_size(self) -> int:
        """How many coordinates of `x*` are non-zero.

        The projection sets the rest to exact zeros, which is the second thing
        no arm reproduces: an unconstrained iterate has none.
        """

        return int((self.optimum() > 0.0).sum())


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class SimplexModel(nn.Module):  # type: ignore[misc]
    """The iterate `x`, as a `d`-element `nn.Parameter`.

    Nothing here keeps `x` on the simplex, and nothing anywhere else does
    either. A model that reparameterised through a softmax would make the
    problem unconstrained and would be solving a different one; this example is
    about what happens when the constraint is stated and nothing enforces it.
    """

    def __init__(self, dim: int, x_init: str = "uniform") -> None:
        """Start at a feasible point.

        Args:
            dim: Problem dimension `d`, from ``model.input_dim``.
            x_init: ``uniform`` starts at the barycentre `1/d`, which is in the
                simplex and, for a power-of-two `d`, sums to exactly 1.0 in
                floating point. ``zeros`` starts outside it, at a point whose
                sum is 0. The default is feasible so that
                ``constraint_violation`` is exactly 0 at round 0 and non-zero
                from round 1 onward -- the shortest statement of what is
                missing.
        """

        super().__init__()
        if x_init == "uniform":
            start = torch.full((dim,), 1.0 / dim, dtype=DTYPE)
        elif x_init == "zeros":
            start = torch.zeros(dim, dtype=DTYPE)
        else:
            raise ValueError(f"model.x_init must be 'uniform' or 'zeros', got {x_init!r}")
        self.x = nn.Parameter(start)

    def forward(self, features: Tensor) -> Tensor:
        """Return the predictions `H_B x` for a batch of rows."""

        return features @ self.x

    @property
    def iterate(self) -> Tensor:
        """The vector the whole example is about, detached."""

        return self.x.detach().clone()


def build_simplex_vector(config: Mapping[str, Any] | None = None) -> SimplexModel:
    """Registry builder for `model.name: simplex_vector`.

    Builds from the model block alone. Whether `d` agrees with the data is
    checked by :class:`SimplexLSQTask`, the one object handed both the model
    block and the manifest.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _MODEL_KEYS, "simplex_vector")
    dim = values.get("input_dim")
    if dim is None:
        raise ValueError("model.input_dim is required for simplex_vector: it is the problem's d")
    return SimplexModel(dim=int(dim), x_init=str(values.get("x_init", "uniform")))


# ---------------------------------------------------------------------------
# The generator: one regression per client, written as shards a run reads
# ---------------------------------------------------------------------------

#: The name of this problem's generator and task, as configs write it.
DATASET_NAME = "simplex_lsq"


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
        raise ValueError("simplex_lsq needs partition.num_clients: the number of regressions")
    return ProblemSpec(
        num_clients=int(partition["num_clients"]),
        dim=int(problem.get("dim", 16)),
        infeasibility=float(problem.get("infeasibility", 0.5)),
        heterogeneity=float(problem.get("heterogeneity", 0.4)),
    )


def _spec_from_reference(reference: Mapping[str, Any]) -> ProblemSpec:
    """Rebuild the spec from the dials the manifest's ``reference`` records.

    The generator and the task are the same module, and the spec is a
    deterministic function of its dials, so the task rebuilds exactly what the
    generator wrote -- the design, the targets, `x*` -- rather than carrying
    `n x m` floats through JSON to get the same tensors back.
    """

    problem = reference["problem"]
    return ProblemSpec(
        num_clients=int(problem["clients"]),
        dim=int(problem["dim"]),
        infeasibility=float(problem["infeasibility"]),
        heterogeneity=float(problem["heterogeneity"]),
    )


def reference_of(spec: ProblemSpec) -> dict[str, Any]:
    """Everything a run on this data is scored against, in closed form.

    Written into the manifest, copied into ``run.json`` by
    ``run_metadata.build_dataset_provenance``, and read back by
    :class:`SimplexLSQTask`. ``negative_gap_floor`` is here because it is the
    number the gap column reaches, and a reader of ``run.json`` should be able
    to see that the run was *expected* to end below zero.
    """

    pooled = spec.pooled_ols()
    return {
        "problem": {
            "clients": spec.num_clients,
            "dim": spec.dim,
            "infeasibility": spec.infeasibility,
            "heterogeneity": spec.heterogeneity,
        },
        "rows_per_client": spec.dim,
        "x_true": spec.truth().tolist(),
        "x_star": spec.optimum().tolist(),
        "f_star": spec.optimal_objective(),
        "unconstrained_objective": spec.unconstrained_objective(),
        "negative_gap_floor": spec.negative_gap_floor(),
        "heterogeneity_residual": spec.heterogeneity_residual(),
        "unconstrained_optimum_sum": float(pooled.sum()),
        "unconstrained_optimum_min": float(pooled.min()),
        "optimum_support_size": spec.optimum_support_size(),
        "constrained": True,
        "projection_available_in_fedbrew": False,
    }


def generate_simplex_lsq_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, plus the manifest a run reads.

    The four arguments are the generator contract (chapter 12 §4). Two of
    them do nothing here, and say so rather than being quietly dropped:

    ``seed``
        The targets are a deterministic function of the spec -- a planted
        point and Hadamard rows in exact +/- pairs -- so there is no draw for
        a seed to fix. Recorded in the manifest anyway, because every other
        dataset's provenance says what seed it was made at.

    ``client_splits``
        The ratios describe a cut, and there is nothing to cut: `f_i` is
        defined over all `m` of a client's rows. All three splits hold the
        same rows, which the manifest declares as
        ``client_test_source: identical_to_train``.
    """

    del client_splits
    spec = _spec_from_config(config)
    features = spec.design()
    targets = spec.client_targets()
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    clients: list[dict[str, Any]] = []
    for index, client_targets in enumerate(targets):
        client_id = f"client_{index}"
        x = features.clone()
        y = client_targets.clone()
        save_split_client_shard(shards_dir / f"{client_id}.pt", x, y, x, y, x, y)
        clients.append(
            {
                "client_id": client_id,
                "shard": f"shards/{client_id}.pt",
                "num_examples": 3 * spec.dim,
                "num_train_examples": spec.dim,
                "num_eval_examples": spec.dim,
                "num_test_examples": spec.dim,
                "offset_norm": float(torch.linalg.vector_norm(spec.client_offsets()[index])),
            }
        )

    save_client_shard(
        shards_dir / "global_test.pt",
        features.repeat(len(targets), 1),
        targets.reshape(-1).clone(),
    )
    _write_partition_stats(output_dir, spec, clients)

    manifest = {
        "dataset_name": DATASET_NAME,
        "format": "torch_shards",
        "client_shard_format": "split_v2",
        "client_test_source": IDENTICAL_TO_TRAIN,
        "num_clients": len(clients),
        "input_dim": spec.dim,
        "clients_file": "clients.jsonl",
        "shards_dir": "shards",
        "global_test": "shards/global_test.pt",
        "partition_stats_file": "partition_stats.json",
        "client_stats_file": "client_stats.csv",
        "partition_strategy": "analytic",
        "partition_key": "offset_index",
        "seed": seed,
        "reference": reference_of(spec),
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients)
    return GenerationSummary(
        manifest_path=manifest_path,
        num_clients=len(clients),
        num_examples=len(clients) * spec.dim,
        num_test_examples=len(clients) * spec.dim,
    )


def _write_partition_stats(
    output_dir: Path,
    spec: ProblemSpec,
    clients: Sequence[Mapping[str, Any]],
) -> None:
    """Write the two files ``fedbrew inspect-data`` reads beside the manifest."""

    payload = {
        "dataset_name": DATASET_NAME,
        "partition_strategy": "analytic",
        "num_clients": len(clients),
        "total_examples": sum(int(client["num_examples"]) for client in clients),
        "min_examples_per_client": 3 * spec.dim,
        "max_examples_per_client": 3 * spec.dim,
        "mean_examples_per_client": 3.0 * spec.dim,
        "infeasibility": spec.infeasibility,
        "heterogeneity": spec.heterogeneity,
        "negative_gap_floor": spec.negative_gap_floor(),
        "clients": [dict(client) for client in clients],
    }
    # allow_nan=False: every float here is measured, and Python's json writes a
    # non-finite one as the bare token NaN, which is not JSON.
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "client_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", "num_examples", "offset_norm"])
        for client in clients:
            writer.writerow(
                [client["client_id"], client["num_examples"], f"{client['offset_norm']:.17g}"]
            )


# ---------------------------------------------------------------------------
# The task adapter
# ---------------------------------------------------------------------------


class SimplexLSQTask(TaskAdapter):
    """Bridge between the constrained problem and the generic FL orchestration.

    `x*` and `F*` are properties of the problem, and they arrive in the
    manifest's ``reference``, written by the generator.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
        device: str = "cpu",
        **unused: Any,
    ) -> None:
        """Build the adapter, and refuse a run whose two halves disagree.

        The arguments are the task contract (chapter 12 §1.4); the ones this
        problem does not use -- ``batch_size``, ``eval_batch_size``,
        ``dataloader_config``, ``reuse_model`` -- arrive in ``unused`` and are
        ignored, because the loader is built per call.

        Args:
            model_config: The resolved ``model`` block. Carries `d`.
            dataset_metadata: The manifest, as ``manifest_dataset`` reports it.
                Carries ``reference``, written by the generator.
            device: The resolved ``runtime.device``.
            **unused: The rest of the contract.

        Raises:
            ValueError: If the model block's `d` is not the generated data's.
                A `d` mismatch would at least fail on a shape, but it fails
                here with the two numbers named instead.
        """

        del unused
        self.device = torch.device(device)
        self.reference = dict((dataset_metadata or {}).get("reference") or {})
        if "problem" not in self.reference:
            raise ValueError(
                "simplex_lsq task needs a manifest written by its own generator: the "
                "reference optimum lives in the manifest's `reference` and this data "
                "has none."
            )
        spec = _spec_from_reference(self.reference)
        _check_model_against_reference(model_config or {}, spec)
        self.spec = spec
        self._optimum = spec.optimum().to(self.device)
        self._optimal_objective = spec.optimal_objective()
        # Built once. `ProblemSpec` is frozen and rebuilds the design and the
        # targets on every call, and `eval_step` evaluates the global objective
        # twice per batch -- at the iterate and at its projection.
        self._design = spec.design().to(self.device)
        self._targets = spec.client_targets().to(self.device)
        self._criterion = _least_squares
        # Read as `getattr(task, "_scaler", None)` by four client rules.
        self._scaler: Any = None

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> SimplexModel:
        """Build the vector model named by the `model` config block."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        factory = models.get(str(values.get("name", "simplex_vector")))
        model = factory(values)
        return model.to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Cut one client's rows into batches. A list, so it is re-iterable.

        The rows arrive as a shard's ``x`` and ``y``, because that is what
        ``manifest_dataset`` hands over.
        """

        loader_config = _loader_config(config)
        features, targets = _rows_of(data)
        features = features.to(self.device)
        targets = targets.to(self.device)
        rows = len(targets)
        batch_size = max(1, int(loader_config.get("batch_size", rows) or rows))
        if bool(loader_config.get("shuffle", False)):
            order = _permutation(rows, loader_config.get("seed"))
            features, targets = features[order], targets[order]
        batches = [
            (features[start : start + batch_size], targets[start : start + batch_size])
            for start in range(0, rows, batch_size)
        ]
        if bool(loader_config.get("drop_last", False)) and len(batches) > 1:
            batches = [batch for batch in batches if len(batch[1]) == batch_size]
        return batches

    def train_step(
        self,
        model: SimplexModel,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Take one *unconstrained* step on the batch's least-squares objective.

        The constraint appears nowhere in here, because there is nowhere for it
        to appear: ``train_step`` is handed an optimizer and returns a loss, and
        a feasible-set projection is neither.
        """

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        features, targets = self._move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = self._criterion(model(features), targets)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def eval_step(self, model: SimplexModel, batch: Any) -> dict[str, float]:
        """Measure the batch's objective, and six properties of the iterate."""

        model.eval()
        features, targets = self._move_batch(batch)
        with torch.no_grad():
            loss = self._criterion(model(features), targets)
            iterate = model.iterate
            projected = project_simplex(iterate)
        return {
            "loss": float(loss),
            "total": float(len(targets)),
            # The objective, and then the three numbers that say whether the
            # objective is a claim about anything.
            "optimality_gap": self._global_objective(iterate) - self._optimal_objective,
            "feasible_gap": self._global_objective(projected) - self._optimal_objective,
            "distance_to_optimum": float(torch.linalg.vector_norm(iterate - self._optimum)),
            "constraint_violation": float(torch.linalg.vector_norm(iterate - projected)),
            "simplex_sum": float(iterate.sum()),
            "min_coordinate": float(iterate.min()),
            "negative_mass": negative_mass(iterate),
        }

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Fold eval-step outputs into the eight numbers this task reports.

        ``loss`` / ``optimality_gap``
            `F(x)` and `F(x) - F*`. **The gap goes negative**, because the
            unconstrained minimiser beats every feasible point and that is where
            the arms go. It is reported anyway, and reported first, because it
            is what a table of this problem would otherwise show alone.

        ``feasible_gap``
            `F(Proj_Delta(x)) - F*`: project the iterate, then measure. Never
            negative, and the number a reader actually wants -- how good this
            run is once someone makes it legal.

        ``distance_to_optimum``
            `||x - x*||` against the constrained optimum.

        ``constraint_violation`` / ``simplex_sum`` / ``min_coordinate`` /
        ``negative_mass``
            `||x - Proj_Delta(x)||`, which is 0 exactly when `x` is feasible,
            and the three readings that say how it is infeasible. All four are 0,
            1 and 1/d at initialisation and never again.
        """

        records = [record for record in outputs if isinstance(record, Mapping)]
        names = (
            "loss",
            "optimality_gap",
            "feasible_gap",
            "distance_to_optimum",
            "constraint_violation",
            "simplex_sum",
            "min_coordinate",
            "negative_mass",
        )
        if not records:
            return dict.fromkeys(names, 0.0)

        weights = [float(record.get("total", 0.0)) for record in records]
        total = sum(weights)

        def pooled(name: str) -> float:
            values = [float(record.get(name, 0.0)) for record in records]
            if total:
                return sum(v * w for v, w in zip(values, weights, strict=True)) / total
            return sum(values) / len(values)

        return {name: pooled(name) for name in names}

    # -- the server's central pass -----------------------------------------

    def evaluate_model(self, model: SimplexModel, data: Any) -> dict[str, float]:
        """Measure the global model on every client's rows at once."""

        outputs = [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        return self.compute_metrics(outputs)

    # -- this task's own helpers --------------------------------------------

    def _global_objective(self, x: Tensor) -> float:
        """`F(x)` over the cached design and targets; the same value as
        ``ProblemSpec.objective_at``, without rebuilding the data."""

        residual = self._design @ x - self._targets
        return float(0.5 * (residual * residual).sum(dim=1).mean()) / len(self._design)

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Split a batch into (features, targets), both on the device."""

        features, targets = batch
        return features.to(self.device), targets.to(self.device)


def _least_squares(outputs: Tensor, targets: Tensor) -> Tensor:
    """`(1/2|B|) ||H_B x - y_B||^2`, as a scalar tensor.

    Two arguments, unlike ``examples/fed-lasso``: the objective here is smooth
    and depends on the parameters only through the outputs, so it fits the
    `criterion(outputs, targets)` shape that ``TorchClassificationTask`` uses.
    The part that does not fit is the feasible set, and the feasible set is not
    a loss term at all.
    """

    residual = outputs - targets
    return 0.5 * (residual * residual).mean()


def _rows_of(data: Any) -> tuple[Tensor, Tensor]:
    """The design rows and targets in one split of a shard."""

    if isinstance(data, Mapping):
        features = data.get("x", data.get("X", data.get("features")))
        targets = data.get("y", data.get("Y", data.get("targets")))
        if isinstance(features, Tensor) and isinstance(targets, Tensor):
            return features.to(DTYPE), targets.to(DTYPE)
    raise ValueError("simplex_lsq data must be a shard mapping carrying 'x' and 'y' tensors")


def _check_model_against_reference(model_config: Mapping[str, Any], spec: ProblemSpec) -> None:
    """Refuse a model block that describes a different problem than the data."""

    dim = int(model_config.get("input_dim", 0))
    if dim == spec.dim:
        return
    raise ValueError(
        f"model config describes d={dim}, but the generated data is d={spec.dim} "
        "(manifest reference). The targets come from the shards and the dimension "
        "from the model block, so a disagreement is two different problems."
    )


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
TASK_NAME = "simplex_lsq"
MODEL_NAME = "simplex_vector"


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
        generate_simplex_lsq_from_config,
        sections={"problem": {"dim", "infeasibility", "heterogeneity"}},
    )
    registry.tasks.register(TASK_NAME, lambda **kwargs: SimplexLSQTask(**kwargs))
    registry.models.register(MODEL_NAME, build_simplex_vector, task=TASK_NAME)


def _self_check() -> None:
    """Assert the seven claims this module's docstring makes about the problem."""

    spec = ProblemSpec()
    design = spec.design()
    rows = spec.dim

    if not torch.equal(design.T @ design, rows * torch.eye(rows, dtype=DTYPE)):
        raise AssertionError("the design is not orthogonal: H^T H != m I")

    truth = spec.truth()
    if float(truth.sum()) != 1.0 or float(truth.min()) < 0.0:
        raise AssertionError("the planted point is not exactly on the simplex")

    offsets = spec.client_offsets()
    if float(offsets.sum(dim=0).abs().max()) != 0.0:
        raise AssertionError("client offsets must sum to exactly zero")

    recovered = spec.client_targets() @ design / rows
    if float((recovered - spec.client_ols()).abs().max()) > 1e-14:
        raise AssertionError("stored targets do not reproduce the OLS solutions")

    pooled = spec.pooled_ols()
    if float(pooled.min()) >= 0.0 or abs(float(pooled.sum()) - 1.0) < 1e-9:
        raise AssertionError("the unconstrained optimum should violate both constraints")

    optimum = spec.optimum()
    if abs(float(optimum.sum()) - 1.0) > 1e-14 or float(optimum.min()) < 0.0:
        raise AssertionError("the projection did not land on the simplex")
    # By construction, not by luck: `displacement` is constant on the even
    # coordinates, `truth`'s support is even, and the projection's threshold
    # therefore lands on exactly that constant. So `x*` is the planted point and
    # the example has one ground truth rather than two. Asserted rather than
    # asserted-in-prose, because it is a property of the default dials and a
    # change to them should say so here rather than in a stale docstring.
    if not torch.equal(optimum, truth):
        raise AssertionError("at the default dials x* should be exactly the planted point")
    if project_simplex(optimum) is not optimum and constraint_violation(optimum) > 1e-14:
        raise AssertionError("the projection is not idempotent")
    if spec.negative_gap_floor() >= 0.0:
        raise AssertionError("the unconstrained optimum should beat the constrained one")

    if _spec_from_reference(reference_of(spec)) != spec:
        raise AssertionError("the spec does not survive the round trip through the manifest")

    model = SimplexModel(dim=spec.dim)
    features, targets = design, spec.client_targets()[0]
    _least_squares(model(features), targets).backward()
    measured = model.x.grad
    if measured is None:
        raise AssertionError("the backward pass left no gradient on the iterate")
    expected = gradient(model.iterate, features, targets)
    if float((measured - expected).abs().max()) > 1e-15:
        raise AssertionError("autograd disagrees with the analytic gradient")


_self_check()
