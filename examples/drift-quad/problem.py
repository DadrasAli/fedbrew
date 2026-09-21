"""Separable strongly convex quadratics with two dials, run through the real loop.

The problem
-----------
Every client optimises a quadratic over `x` in `R^d` with the *same* diagonal
curvature `A` and its *own* linear offset `b_i`::

    f_i(x) = (1/2) x^T A x - b_i^T x        grad f_i(x) = A x - b_i

`A = diag(a_0, ..., a_{d-1})` with `a_j = kappa^(j/(d-1))`, so `a_0 = 1` and
`a_{d-1} = kappa` exactly: `mu = 1`, `L = kappa`, and the condition number of
the problem *is* the ``condition_number`` dial. The Hessian is diagonal, so the
problem separates: the `d` coordinates are `d` independent scalar quadratics
that never mix, and every claim below can be read off one coordinate at a time.

The offsets are laid out in exact +/- pairs, so they sum to zero in floating
point and the uniform mean over clients is::

    F(x) = (1/2) x^T A x     grad F(x) = A x     x* = 0     F* = 0

both exact for the *federated* objective, not only for one client. Client `i`'s
own minimiser is `x_i* = A^{-1} b_i`, which is not at 0: the clients genuinely
disagree, and the disagreement is the second dial. It is the standard
gradient-dissimilarity constant, measured at the optimum::

    zeta^2 = (1/n) sum_i ||grad f_i(x*) - grad F(x*)||^2 = (1/n) sum_i ||b_i||^2

and since every `||b_i||` is set to ``dissimilarity``, `zeta` is exactly that
dial rather than a consequence of it.

Why two dials
-------------
They are the two things a federated method can fix, and they are separable
here in the same way the coordinates are:

* ``condition_number`` sets the *rate*. Local SGD needs `eta < 2/L = 2/kappa`
  to be stable in the stiffest coordinate, and then the softest coordinate
  contracts by `(1 - eta)^K` per round -- so `kappa` divides the progress of
  every SGD arm. Server-side preconditioning (the FedOpt family) attacks this.
* ``dissimilarity`` sets the *floor*. Under partial participation the sampled
  mean offset is not zero, and the round's fixed point is displaced by
  `A^{-1} b_S` -- a bias that no client learning rate removes. SCAFFOLD's
  control variates attack this.

Neither dial is attacked by the other's method, which is what makes a run of
this example legible. See the README for the measured 2x2.

One consequence is worth stating because it is easy to build a benchmark that
hides it: with a *shared* `A` the local map is affine, so under **full**
participation `mean_i (I - eta A)^K (x - x_i*) + x_i*` collapses to
`(I - eta A)^K x` and FedAvg is exactly unbiased -- the drift cancels, and the
``dissimilarity`` dial does nothing at all. Client drift on this problem is a
statement about *partial* participation. ``participation_rate: 1.0`` is the
control that shows it.

Structure the run can be checked against
----------------------------------------
:class:`ProblemSpec` exposes every one of these in closed form: the curvature
`A`, the offsets `b_i`, the global optimum `x* = 0` with `F* = 0`, each
client's own optimum `A^{-1} b_i`, the drift radius `max_i ||x_i*||`, and the
largest stable client learning rate `2/L`.

There is no data. A "batch" carries one offset vector `b_i`, and the gradient
is the analytic expression above, returned by a custom autograd Function so
that every shipped client rule gets the same exact gradient, whether it calls
``TaskAdapter.train_step`` or drives ``loss.backward()`` through its own
optimizer wrapper.

What this file registers
------------------------
Three names, at ``register()`` time, and nothing at import:

    generators  "drift_quad"    writes the shards and the manifest
    tasks       "drift_quad"    DriftQuadTask
    models      "quad_vector"   QuadraticModel, a single d-vector parameter

There is no dataset backend. The offsets are *generated* -- from a seed and
the spec in a generator config, rather than from a corpus -- and a run reads
them through the shipped ``manifest_dataset`` like every other run in the
repository. That is the whole reason this example no longer needs a
``run.py`` that imports itself: ``fedbrew generate`` writes the data,
``fedbrew run`` reads it, and neither knows this problem is analytic.

The reference optimum travels with the data. `x*`, `F*`, the dials and every
floor they imply are properties of the generated shards, so the generator
writes them into the manifest under ``reference``; the task reads them from
``dataset_metadata`` and ``run.json`` records them, which is how a run says
what it was scored against. :class:`DriftQuadTask` is also the one place that
sees both channels the problem is split across -- the curvature arrives in
``model.extra`` because that is where a run config can carry it, the offsets
arrive in the shards -- so it is where they are cross-checked.
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

#: The optimum's objective value, in closed form and exact for the federated
#: objective because the offsets sum to zero. `x*` is the zero vector, whose
#: dimension depends on the spec; :func:`x_star` builds it.
F_STAR = 0.0

#: The strong-convexity constant. `a_0 = kappa^0 = 1` for every kappa, so this
#: is fixed and the ``condition_number`` dial moves `L` alone.
MU = 1.0

#: Every tensor here is float64. The point of the example is to watch an arm
#: that has no drift floor fall past the floor the other arms sit on; in
#: float32 both would land on the precision floor and the comparison would
#: read as a tie.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def x_star(dim: int) -> Tensor:
    """`x* = 0`, as a `dim`-vector."""

    return torch.zeros(dim, dtype=DTYPE)


def objective(x: Tensor, curvature: Tensor, offset: Tensor) -> float:
    """`f_i(x) = (1/2) x^T A x - b_i^T x`, as a plain float."""

    return float(0.5 * (curvature * x * x).sum() - (offset * x).sum())


def gradient(x: Tensor, curvature: Tensor, offset: Tensor) -> Tensor:
    """`grad f_i(x) = A x - b_i`.

    The analytic derivative of :func:`objective`. :class:`_QuadraticObjective`
    returns exactly this from its backward pass, so nothing in the run relies
    on autograd rediscovering it.
    """

    return curvature * x - offset


def optimality_gap(x: Tensor, curvature: Tensor) -> float:
    """`F(x) - F* = (1/2) x^T A x`, the quantity every rate here bounds."""

    return float(0.5 * (curvature * x * x).sum()) - F_STAR


def distance_to_optimum(x: Tensor) -> float:
    """`||x - x*||`, which is `||x||` because `x*` is the origin."""

    return float(torch.linalg.vector_norm(x))


class _QuadraticObjective(torch.autograd.Function):  # type: ignore[misc]
    """`f_i(x)` forward, hand-written `A x - b_i` backward.

    Written as an autograd Function rather than left to autograd on the closed
    form so that the gradient every rule sees is the analytic one, bit for bit,
    on every path into the model: ``train_step``, and the SCAFFOLD, FedProx and
    FedLALR paths that wrap the optimizer around it.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, curvature: Tensor, offsets: Tensor) -> Tensor:
        """Return `f_i(x)` for each offset row; shape `(N,)`."""

        ctx.save_for_backward(x, curvature, offsets)
        quadratic = 0.5 * (curvature * x * x).sum()
        return quadratic - offsets @ x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        """Return `sum_r grad_output_r * (A x - b_r)`, summed over the batch."""

        x, curvature, offsets = ctx.saved_tensors
        weight = grad_output.sum()
        return weight * (curvature * x) - grad_output @ offsets, None, None


# ---------------------------------------------------------------------------
# The model: one d-vector, plus the curvature it is measured against
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("condition_number", "x_init")


def curvature_of(dim: int, condition_number: float) -> Tensor:
    """The diagonal of `A`: `a_j = kappa^(j/(d-1))`, geometrically spaced.

    Exact at both ends -- `kappa**0.0` is 1 and `kappa**1.0` is `kappa` -- so
    `mu` and `L` are the documented numbers and not almost them.
    """

    if dim < 2:
        raise ValueError("dim must be at least 2")
    if condition_number < 1.0:
        raise ValueError("condition_number must be at least 1")
    exponents = [index / (dim - 1) for index in range(dim)]
    return torch.tensor([condition_number**power for power in exponents], dtype=DTYPE)


class QuadraticModel(nn.Module):  # type: ignore[misc]
    """The iterate `x`, as a `d`-element `nn.Parameter`, plus `A`.

    `A` is a **non-persistent** buffer. A persistent one would appear in
    ``state_dict``, and ``fedbrew.core.torch_utils.get_model_state`` is
    ``state_dict``: the curvature would then be uploaded, averaged and counted
    into ``communicated_parameters`` every round -- harmless numerically, since
    every client holds the same `A`, and wrong in the one column that is
    supposed to say what a round costs. A constant that is not state belongs
    outside the state dict.
    """

    def __init__(
        self,
        dim: int,
        condition_number: float = 100.0,
        x_init: float = 1.0,
    ) -> None:
        """Place the iterate at equal objective energy in every coordinate.

        Args:
            dim: Problem dimension `d`, from ``model.input_dim``.
            condition_number: `kappa`. Sets `A`; see :func:`curvature_of`.
            x_init: The per-coordinate *energy* scale of the starting point:
                `x_0[j] = x_init / sqrt(a_j)`, so every coordinate contributes
                exactly `x_init^2 / 2` to `F(x_0)` and the initial gap is
                `d x_init^2 / 2`. Identical on every client, because the server
                broadcasts its own initial state before round 1.

                Not `x_init` in every coordinate, which is what this was and
                which quietly decides the comparison: a per-coordinate
                normalised method -- FedAdam with `beta1 = 0`, and any sign-like
                rule -- takes the same step in every coordinate, so from a
                start where every coordinate is the same distance from 0 it can
                land on `x*` in one round no matter what `kappa` is. Measured:
                `central_test_loss` of 2.7e-12 after round 1, and 3.3e-136 at
                round 200, against FedAvg's 2.4e-34. That is a fact about the
                starting point, not about the method, and it is exactly the
                kind of number that reads as a result. Geometric spacing
                removes the coincidence: the coordinates start a factor
                `sqrt(kappa)` apart in distance.
        """

        super().__init__()
        curvature = curvature_of(dim, float(condition_number))
        self.x = nn.Parameter(float(x_init) / torch.sqrt(curvature))
        self.register_buffer("curvature", curvature, persistent=False)

    def forward(self, offsets: Tensor) -> Tensor:
        """Return `f_i(x)` per row of `offsets`."""

        return _QuadraticObjective.apply(self.x, self.curvature, offsets)

    @property
    def iterate(self) -> Tensor:
        """The vector the whole example is about, detached."""

        return self.x.detach().clone()


def build_quad_vector(config: Mapping[str, Any] | None = None) -> QuadraticModel:
    """Registry builder for `model.name: quad_vector`.

    Builds from the model block alone. Whether the `d` and `kappa` it names
    are the ones the *data* was generated with is a question about two config
    blocks at once, and a builder is handed one of them;
    :meth:`DriftQuadTask.__init__` answers it, because the task is handed
    both.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _MODEL_KEYS, "quad_vector")
    dim = values.get("input_dim")
    if dim is None:
        raise ValueError("model.input_dim is required for quad_vector: it is the problem's d")
    return QuadraticModel(
        dim=int(dim),
        condition_number=float(values.get("condition_number", 100.0)),
        x_init=float(values.get("x_init", 1.0)),
    )


# ---------------------------------------------------------------------------
# The dataset: one offset vector per client
# ---------------------------------------------------------------------------


def _hadamard_row(index: int, dim: int) -> list[float]:
    """Row `index` of the Sylvester-Hadamard matrix, as +/-1 entries.

    `H[k][j] = (-1)^popcount(k & j)`. For a power-of-two `dim` the rows are
    mutually orthogonal, so distinct clients disagree in orthogonal directions
    and the heterogeneity has no accidental structure; row 0 is the all-ones
    vector and is skipped, so no client's disagreement is a pure rescaling of
    the mean. For any other `dim` the entries are still exactly +/-1 -- so `||b_i||` is still
    exact -- but the rows are no longer orthogonal.
    """

    return [-1.0 if bin(index & column).count("1") % 2 else 1.0 for column in range(dim)]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """The whole problem, as four numbers."""

    num_clients: int = 8
    #: `d`. A power of two keeps the offset directions orthogonal.
    dim: int = 16
    #: `kappa = L / mu`, with `mu` pinned to 1. The rate dial.
    condition_number: float = 100.0
    #: `zeta`, the gradient-dissimilarity constant, in gradient units.
    #: 0.0 makes every client's objective `F`, which is the homogeneous
    #: control. The floor dial.
    dissimilarity: float = 1.0

    def __post_init__(self) -> None:
        """Refuse a spec that cannot express what it claims to."""

        if self.num_clients < 2:
            raise ValueError("problem.clients must be at least 2")
        if self.dim < 2:
            raise ValueError("problem.dim must be at least 2")
        if self.condition_number < 1.0:
            raise ValueError("problem.condition_number must be at least 1")
        if self.dissimilarity < 0.0:
            raise ValueError("problem.dissimilarity must be non-negative")
        if self.num_clients // 2 > self.dim - 1:
            raise ValueError(
                f"{self.num_clients} clients need {self.num_clients // 2} offset "
                f"directions, and d={self.dim} supplies {self.dim - 1}. Raise "
                "problem.dim or lower problem.clients."
            )

    def curvature(self) -> Tensor:
        """The diagonal of `A`; `mu = 1`, `L = condition_number`."""

        return curvature_of(self.dim, self.condition_number)

    def offsets(self) -> Tensor:
        """The per-client offsets `b_i`, shape `(n, d)`.

        Built in exact +/- pairs, from Hadamard rows scaled to norm
        ``dissimilarity``. The pairing is what makes `sum_i b_i` exactly zero
        in floating point, and that is what keeps `x* = 0` and `F* = 0` exact
        for the federated objective rather than approximately true. A
        `scale * (2i/(n-1) - 1)` layout leaves a residue of a few ulp, which
        moves the optimum off the value this module documents.
        """

        scale = self.dissimilarity / math.sqrt(self.dim)
        rows: list[list[float]] = []
        for pair in range(self.num_clients // 2):
            direction = [scale * sign for sign in _hadamard_row(pair + 1, self.dim)]
            rows.append(direction)
            rows.append([-value for value in direction])
        if self.num_clients % 2:
            rows.append([0.0] * self.dim)
        return torch.tensor(rows, dtype=DTYPE)

    def client_optima(self) -> Tensor:
        """Each client's own minimiser `x_i* = A^{-1} b_i`, shape `(n, d)`.

        The ground-truth structure a run is measured against: an arm that
        drifts ends its local training here, and an arm that does not ends it
        at `x* = 0`. ``fit_distance_to_optimum`` and
        ``fit_distance_to_client_optimum`` are the two distances, and they
        trade off exactly.
        """

        return self.offsets() / self.curvature()

    def drift_radius(self) -> float:
        """`max_i ||x_i*||`: how far client drift can carry a local iterate."""

        return float(torch.linalg.vector_norm(self.client_optima(), dim=1).max())

    def realised_dissimilarity(self) -> float:
        """`zeta` as actually built: `sqrt(mean_i ||b_i||^2)`.

        Equals ``dissimilarity`` up to floating-point rounding for an even
        client count, and is smaller for an odd one, where the unpaired client
        gets `b = 0`. Reported rather than assumed, because `zeta` is the dial
        every claim about the floor is stated in.
        """

        offsets = self.offsets()
        return float(torch.sqrt((offsets * offsets).sum(dim=1).mean()))

    def initial_gap(self, x_init: float = 1.0) -> float:
        """`F(x_0) - F* = d x_init^2 / 2`, for :class:`QuadraticModel`'s start.

        Independent of `kappa` by construction, which is what makes the two
        dials comparable across settings: changing the condition number
        changes how hard the descent is, not how far down it has to go.
        """

        return 0.5 * self.dim * x_init * x_init

    def max_stable_learning_rate(self) -> float:
        """`2/L`: above this, local GD diverges in the stiffest coordinate."""

        return 2.0 / self.condition_number


# ---------------------------------------------------------------------------
# The generator: the offsets, written as shards a run reads like any other
# ---------------------------------------------------------------------------

#: The name of this problem's generator and task, as configs write it.
DATASET_NAME = "drift_quad"


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
        raise ValueError("drift_quad needs partition.num_clients: it is the number of offsets")
    return ProblemSpec(
        num_clients=int(partition["num_clients"]),
        dim=int(problem.get("dim", 16)),
        condition_number=float(problem.get("condition_number", 100.0)),
        dissimilarity=float(problem.get("dissimilarity", 1.0)),
    )


def reference_of(spec: ProblemSpec) -> dict[str, Any]:
    """Everything a run on this data is scored against, in closed form.

    Written into the manifest, copied into ``run.json`` by
    ``run_metadata.build_dataset_provenance``, and read back by
    :class:`DriftQuadTask`. It carries the dials as well as the optimum
    because a floor is only meaningful beside the dial that sets it, and
    because the task cross-checks the model config against them.
    """

    return {
        "problem": {
            "dim": spec.dim,
            "condition_number": spec.condition_number,
            "dissimilarity": spec.dissimilarity,
            "clients": spec.num_clients,
        },
        "x_star": x_star(spec.dim).tolist(),
        "f_star": F_STAR,
        "mu": MU,
        "smoothness": spec.condition_number,
        "max_stable_learning_rate": spec.max_stable_learning_rate(),
        "drift_radius": spec.drift_radius(),
        "realised_dissimilarity": spec.realised_dissimilarity(),
        "client_optima": spec.client_optima().tolist(),
    }


def generate_drift_quad_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, plus the manifest a run reads.

    The four arguments are the generator contract (chapter 12 §4). Two of
    them do nothing here, and say so rather than being quietly dropped:

    ``seed``
        The offsets are a deterministic function of the spec -- Hadamard rows
        in exact +/- pairs -- so there is no draw for a seed to fix. Recorded
        in the manifest anyway, because "this data was generated at seed 42"
        is what every other dataset's provenance says and an absent seed
        would read as an omission.

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
    offsets = spec.offsets()
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    clients: list[dict[str, Any]] = []
    for index, offset in enumerate(offsets):
        client_id = f"client_{index}"
        row = offset.reshape(1, -1).clone()
        # The second column is a dummy target. Nothing reads it -- f_i needs
        # no label -- but a shard is an (x, y) pair everywhere else, and
        # manifest_validation checks for both.
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
                "offset_norm": float(torch.linalg.vector_norm(offset)),
            }
        )

    # The server's central pass evaluates F in one go, so the pooled shard is
    # every client's offset stacked -- which is exactly the federated
    # objective, because every client carries the same weight.
    save_client_shard(
        shards_dir / "global_test.pt", offsets.clone(), torch.zeros(len(offsets), dtype=DTYPE)
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
        num_examples=len(clients),
        num_test_examples=len(clients),
    )


def _write_partition_stats(
    output_dir: Path,
    spec: ProblemSpec,
    clients: Sequence[Mapping[str, Any]],
) -> None:
    """Write the two files ``fedbrew inspect-data`` reads beside the manifest.

    No label counts: a regression has no labels, and an empty
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
        "condition_number": spec.condition_number,
        "dissimilarity": spec.dissimilarity,
        "realised_dissimilarity": spec.realised_dissimilarity(),
        "drift_radius": spec.drift_radius(),
        "clients": [dict(client) for client in clients],
    }
    # allow_nan=False: every float here is measured -- the offset norms, the
    # realised zeta, the drift radius -- and Python's json writes a
    # non-finite one as the bare token NaN, which is not JSON and which the
    # next reader chokes on rather than noticing. The shipped partition_stats
    # writers are exempt from that rule because they hold only counts and
    # validated ratios; this one is not.
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


class DriftQuadTask(TaskAdapter):
    """Bridge between the quadratic family and the generic FL orchestration.

    Implements the five abstract methods, plus ``evaluate_model`` for the
    server's central pass. Structurally identical to ``examples/pl-1d``'s
    adapter -- see that README §3 for why ``_criterion``, ``_move_batch`` and
    ``_scaler`` are here.
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
        ignored, because a 16-vector has no throughput question and the
        loader is built per call.

        Args:
            model_config: The resolved ``model`` block. Carries `d` and
                `kappa`.
            dataset_metadata: The manifest, as ``manifest_dataset`` reports
                it. Carries ``reference``, written by the generator.
            device: The resolved ``runtime.device``.
            **unused: The rest of the contract.

        Raises:
            ValueError: If the model block and the generated data describe
                different problems. They are the two channels this problem is
                split across -- the curvature travels in ``model.extra``
                because that is where a run config can carry it, the offsets
                travel in the shards -- and this is the one object handed
                both. A `d` mismatch would at least fail on a shape; a
                `kappa` mismatch would fail on nothing at all and draw a
                plausible curve for a condition number the config does not
                name.
        """

        del unused
        self.device = torch.device(device)
        self.reference = dict((dataset_metadata or {}).get("reference") or {})
        _check_model_against_reference(model_config or {}, self.reference)
        self._criterion = _mean_of_batch
        # Read as `getattr(task, "_scaler", None)` by four client rules, to
        # decide whether to refuse `runtime.use_amp: true`. Set explicitly so
        # the answer is where a reader will look for it.
        self._scaler: Any = None

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> QuadraticModel:
        """Build the vector model named by the `model` config block."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        factory = models.get(str(values.get("name", "quad_vector")))
        model = factory(values)
        return model.to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Cut one client's offsets into batches.

        The offsets arrive as a shard's ``x``, because that is what
        ``manifest_dataset`` hands over and what every other dataset in the
        repository stores.

        A list, not a generator: ``single_batch`` re-iterates the loader when
        it runs out, so the loader has to be re-iterable.

        The second element of each batch is a dummy target. Nothing reads it --
        `f_i` needs no label -- but every client rule destructures a batch into
        two, so a one-element batch would fail in five places.
        """

        loader_config = _loader_config(config)
        offsets = _offsets_of(data).to(self.device)
        rows = len(offsets)
        batch_size = max(1, int(loader_config.get("batch_size", rows) or rows))
        if bool(loader_config.get("shuffle", False)):
            offsets = offsets[_permutation(rows, loader_config.get("seed"))]
        batches = [offsets[start : start + batch_size] for start in range(0, rows, batch_size)]
        if bool(loader_config.get("drop_last", False)) and len(batches) > 1:
            batches = [batch for batch in batches if len(batch) == batch_size]
        return [(batch, torch.zeros(len(batch), dtype=DTYPE)) for batch in batches]

    def train_step(
        self,
        model: QuadraticModel,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Take one optimizer step on the batch's mean objective."""

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        offsets, targets = self._move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = self._criterion(model(offsets), targets)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def eval_step(self, model: QuadraticModel, batch: Any) -> dict[str, float]:
        """Measure the batch's mean objective, and the iterate it was measured at."""

        model.eval()
        offsets, _ = self._move_batch(batch)
        with torch.no_grad():
            values = model(offsets)
            iterate = model.iterate
            own_optima = offsets / model.curvature
            drift = torch.linalg.vector_norm(iterate - own_optima, dim=1).mean()
        return {
            "loss": float(values.mean()),
            "total": float(len(offsets)),
            # Carried per batch because compute_metrics is handed the outputs
            # and nothing else, and three of the four numbers it returns are
            # functions of the iterate rather than of the objective's value.
            "optimality_gap": optimality_gap(iterate, model.curvature),
            "distance_to_optimum": distance_to_optimum(iterate),
            "distance_to_client_optimum": float(drift),
        }

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Fold eval-step outputs into the four numbers this task reports.

        ``loss``
            The example-weighted mean of `f_i(x)` over whatever was evaluated.
            On the central pass that is every client, so it *is* `F(x)`, and
            since `F* = 0` it is also the optimality gap. On a client pass it
            is that one client's tilted objective, which can be negative.

        ``optimality_gap`` / ``distance_to_optimum``
            `F(x) - F*` and `||x - x*||`, both functions of the iterate alone
            and both measured against the *global* objective, whichever client
            happens to be reporting them.

        ``distance_to_client_optimum``
            `||x - A^{-1} b_i||`, averaged over the rows evaluated. The drift
            meter, and the counterpart of the one above: a local iterate that
            has drifted to its own minimiser has this near zero and
            ``distance_to_optimum`` near the drift radius, and an arm that
            corrects for drift has them the other way round.

        There is deliberately no ``accuracy``: a quadratic has none, and since
        ``loop._validate_client_evaluation`` stopped requiring one, nothing
        asks for it. pl-1d's README §4 has the history.
        """

        records = [record for record in outputs if isinstance(record, Mapping)]
        if not records:
            return {
                "loss": 0.0,
                "optimality_gap": 0.0,
                "distance_to_optimum": 0.0,
                "distance_to_client_optimum": 0.0,
            }

        weights = [float(record.get("total", 0.0)) for record in records]
        total = sum(weights)

        def pooled(name: str) -> float:
            values = [float(record.get(name, math.nan)) for record in records]
            if total:
                return sum(v * w for v, w in zip(values, weights, strict=True)) / total
            return sum(values) / len(values)

        return {
            "loss": pooled("loss"),
            "optimality_gap": pooled("optimality_gap"),
            "distance_to_optimum": pooled("distance_to_optimum"),
            "distance_to_client_optimum": pooled("distance_to_client_optimum"),
        }

    # -- the server's central pass -----------------------------------------

    def evaluate_model(self, model: QuadraticModel, data: Any) -> dict[str, float]:
        """Measure the global model on every client's objective at once.

        ``FedAvgServer.evaluate_global`` prefixes what this returns with
        ``global_`` and hands it to ``loop._evaluate_central_test_set``, which
        passes through every finite numeric key it is given. So
        ``central_test_loss`` is `F(x) - F*` for the aggregated iterate -- the
        curve this example exists to draw -- and the other three arrive beside
        it under the same prefix.
        """

        outputs = [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        return self.compute_metrics(outputs)

    # -- this task's own batch splitter -------------------------------------

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Split a batch into (offsets, ignored targets), both on the device."""

        offsets, targets = batch
        return offsets.to(self.device), targets.to(self.device)


def _mean_of_batch(outputs: Tensor, targets: Tensor | None = None) -> Tensor:
    """The batch objective: the mean of `f_i(x)` over the batch's rows."""

    del targets
    return outputs.mean()


def _offsets_of(data: Any) -> Tensor:
    """The offset rows in one split of a shard, as `(N, d)`."""

    if isinstance(data, Mapping):
        for key in ("x", "X", "offsets"):
            value = data.get(key)
            if isinstance(value, Tensor):
                return value.to(DTYPE)
    if isinstance(data, Tensor):
        return data.to(DTYPE)
    raise ValueError("drift_quad data must be a shard mapping carrying an 'x' tensor")


def _check_model_against_reference(
    model_config: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    """Refuse a model block that describes a different problem than the data."""

    problem = reference.get("problem")
    if not isinstance(problem, Mapping):
        # No reference: the data was not written by this generator, and there
        # is nothing to check against. The run still fails on a shape if the
        # dimensions disagree.
        return
    dim = int(model_config.get("input_dim", 0))
    condition_number = float(model_config.get("condition_number", 100.0))
    if dim == int(problem["dim"]) and condition_number == float(problem["condition_number"]):
        return
    raise ValueError(
        f"model config describes d={dim}, kappa={condition_number}, but the "
        f"generated data is d={problem['dim']}, kappa={problem['condition_number']} "
        f"(manifest reference). The offsets come from the shards and the "
        "curvature from the model block, so a disagreement is two different "
        "problems in one run."
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
TASK_NAME = "drift_quad"
MODEL_NAME = "quad_vector"


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
        generate_drift_quad_from_config,
        sections={"problem": {"dim", "condition_number", "dissimilarity"}},
    )
    registry.tasks.register(TASK_NAME, lambda **kwargs: DriftQuadTask(**kwargs))
    registry.models.register(MODEL_NAME, build_quad_vector, task=TASK_NAME)


def _self_check() -> None:
    """Assert the four claims this module's docstring makes about the problem.

    Cheap enough to run at import: a handful of 16-vector operations and one
    backward pass. Every one is load-bearing -- if the offsets do not sum to
    zero the federated optimum is not at 0, if `zeta` is not the dial then no
    statement about the floor means what it says, if the backward pass
    disagrees with :func:`gradient` the run is not solving this problem, and
    if `A^{-1} b_i` is not the client's minimiser the drift meter is measuring
    a distance to nothing.
    """

    spec = ProblemSpec()
    offsets = spec.offsets()
    curvature = spec.curvature()

    if float(offsets.sum(dim=0).abs().max()) != 0.0:
        raise AssertionError("offsets must sum to exactly zero")
    if abs(spec.realised_dissimilarity() - spec.dissimilarity) > 1e-12:
        raise AssertionError("realised zeta differs from the dissimilarity dial")
    if float(curvature[0]) != MU or float(curvature[-1]) != spec.condition_number:
        raise AssertionError("curvature does not span [mu, L] exactly")

    for index, own_optimum in enumerate(spec.client_optima()):
        residual = gradient(own_optimum, curvature, offsets[index])
        if float(residual.abs().max()) > 1e-15:
            raise AssertionError("A^{-1} b_i is not client i's minimiser")

    if reference_of(spec)["f_star"] != F_STAR:
        raise AssertionError("the manifest's reference optimum is not this module's")

    model = QuadraticModel(dim=spec.dim, condition_number=spec.condition_number, x_init=0.7)
    batch = offsets[:2]
    _mean_of_batch(model(batch)).backward()
    measured = model.x.grad
    if measured is None:
        raise AssertionError("the backward pass left no gradient on the iterate")
    expected = sum(gradient(model.iterate, curvature, row) for row in batch) / len(batch)
    if float((measured - expected).abs().max()) > 1e-15:
        raise AssertionError("autograd disagrees with the analytic gradient")


_self_check()
