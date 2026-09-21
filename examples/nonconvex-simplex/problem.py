"""A non-convex quadratic over the probability simplex: Motzkin-Straus.

The problem
-----------
Client `i` holds its own noisy view `A_i` of a graph's adjacency matrix, and the
federated problem is::

    minimise   F(x) = mean_i -(1/2) x^T A_i x = -(1/2) x^T A x
    subject to x in Delta = { x : x >= 0, sum_j x_j = 1 }

The `A_i` differ by symmetric, zero-diagonal perturbations laid out in exact +/-
pairs, so they sum to zero and the mean is the true adjacency `A` exactly. No
client sees the graph; only the average is it.

`A` is symmetric with a zero diagonal and indefinite -- its trace is 0 and it is
not the zero matrix, so it has eigenvalues of both signs. `F` is therefore a
genuinely non-convex quadratic, and the constrained problem has a value known in
closed form by the **Motzkin-Straus theorem** (Canad. J. Math. 1965)::

    max_{x in Delta} x^T A x = 1 - 1/omega(G)

with `omega(G)` the clique number. So::

    F* = -(1/2) (1 - 1/omega)        x* = uniform on a maximum clique

The graph is a disjoint union of a `K_5` and a star `K_{1,25}` on 31 vertices,
so `omega = 5`, `F* = -0.4`, and `x*` is `1/5` on the clique's five vertices.
The star holds a second set of local optima at `-0.25`: any `x` with half its
mass on the hub and the other half over the leaves is worth `-0.25`, and no
nearby feasible point does better. The set is one connected face containing
the star's 25 edges, not 25 isolated optima.

Why this graph, and not two cliques
------------------------------------
Because the leading eigenvector must not point at the answer. `K_5`'s adjacency
has spectral radius 4; `K_{1,25}`'s is `sqrt(25) = 5`. The clique number is the
`K_5`'s and the spectral radius is the star's, so a method that follows the top
eigenvector goes to the wrong component. Two disjoint cliques would put the
spectral radius and the clique number on the same vertices and every failure
below would accidentally look like a success.

What goes wrong
---------------
Two things, and the second is the one worth having.

**The unconstrained problem is unbounded below.** `A` has a positive eigenvalue,
so `-(1/2) x^T A x -> -inf` along it, and there is no minimum to converge to. No
shipped algorithm can respect a constraint -- there is no projection, no
Frank-Wolfe step and no mirror map anywhere in fedbrew -- so every arm runs off
to infinity. The iterate grows by a factor of `1 + eta * lambda_max` per local
step, which is fast enough to be unmistakable and slow enough to stay finite for
the whole run.

**Nothing stops it.** ``divergence`` cannot see this. ``blowup_factor`` anchors
on the first strictly *positive* observation of the monitored metric, and this
objective is negative from round 1, so the anchor is never set and the detector
never arms. ``blowup_absolute`` is an upper ceiling and the value is heading
down. ``non_finite`` is the only one left, and it fires only once the arithmetic
overflows -- which at the shipped learning rate is round 531 for FedAvg. A
150-round run therefore **completes, reports success, and records a loss around
-9e85**.

**And projecting afterwards does not rescue it.** ``examples/simplex-lsq`` ends
with `Proj_Delta` of its iterate landing exactly on `x*`, which is a property of
that problem's geometry and not a method. Here the iterate diverges along the
star's eigenvector, whose largest coordinate is the hub, and the projection of a
large vector concentrates on its largest coordinate -- so `Proj_Delta(x)` tends
to the single hub vertex, which spans no edge and scores `F = 0`. That is worse
than the barycentre the run started from. The projected gap goes *up*, to `+0.4`.

What this file registers
------------------------
Three names, at ``register()`` time, and nothing at import:

    generators  "nonconvex_simplex"  writes the shards and the manifest
    tasks       "nonconvex_simplex"  NonconvexSimplexTask
    models      "simplex_point"      SimplexPointModel, a d-vector

There is no dataset backend. The client adjacencies are *generated* -- from
the spec in a generator config -- and a run reads them through the shipped
``manifest_dataset`` like every other run in the repository. `F*`, `x*`, the
two spectral radii and the maximal-clique count are properties of the
generated graph, so the generator writes them into the manifest under
``reference``, the task reads them from ``dataset_metadata``, and ``run.json``
records them.
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

#: float64 throughout. The iterate's norm spans 160 orders of magnitude over a
#: run; in float32 it would overflow before round 40 and the example would be
#: about the arithmetic instead of about the guard that does not fire.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# The feasible set
# ---------------------------------------------------------------------------


def project_simplex(values: Tensor) -> Tensor:
    """Euclidean projection onto `{x >= 0, sum x = 1}`.

    Sort-and-threshold (Duchi et al., ICML 2008), the
    same routine ``examples/simplex-lsq`` uses. Present to *define* the feasible
    set and to measure how far the run is from it -- nothing under
    ``fedbrew/clients/`` or ``fedbrew/servers/`` projects anything.
    """

    finite = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    sorted_values, _ = torch.sort(finite, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    counts = torch.arange(1, len(finite) + 1, dtype=finite.dtype, device=finite.device)
    candidates = sorted_values - (cumulative - 1.0) / counts
    support = int((candidates > 0).sum())
    threshold = (cumulative[support - 1] - 1.0) / support
    return torch.clamp(finite - threshold, min=0.0)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def objective(x: Tensor, adjacency: Tensor) -> float:
    """`f_i(x) = -(1/2) x^T A_i x`, as a plain float."""

    return -0.5 * float(x @ (adjacency @ x))


def gradient(x: Tensor, adjacency: Tensor) -> Tensor:
    """`-A_i x`, the analytic gradient.

    Exact because `A_i` is symmetric: the two halves of
    `d/dx [x^T A x] = (A + A^T) x` are the same matrix. `_self_check` verifies
    autograd returns this rather than assuming it.
    """

    return -(adjacency @ x)


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("x_init",)


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """The whole problem, as three numbers, plus every reference it implies."""

    num_clients: int = 8
    #: Size of the maximum clique. Sets `omega`, and so `F*` and `x*`.
    clique_size: int = 5
    #: Leaves of the decoy star. Its spectral radius is `sqrt(star_leaves)`, so
    #: this must exceed `(clique_size - 1)^2` for the star to out-rank the
    #: clique spectrally -- which is the whole point of it being there.
    star_leaves: int = 25
    #: Per-client perturbation of the adjacency, in exact +/- pairs. The mean is
    #: the true graph regardless, so this moves the clients apart without
    #: moving `F*`.
    heterogeneity: float = 0.1

    def __post_init__(self) -> None:
        """Refuse a spec that cannot express what it claims to."""

        if self.num_clients < 2:
            raise ValueError("problem.clients must be at least 2")
        if self.clique_size < 2:
            raise ValueError("problem.clique_size must be at least 2")
        if self.star_leaves < 1:
            raise ValueError("problem.star_leaves must be at least 1")
        if self.heterogeneity < 0.0:
            raise ValueError("problem.heterogeneity must be non-negative")
        if self.star_leaves <= (self.clique_size - 1) ** 2:
            raise ValueError(
                f"a star with {self.star_leaves} leaves has spectral radius "
                f"{self.star_leaves**0.5:.3g}, which does not exceed the clique's "
                f"{self.clique_size - 1}. The decoy would not be a decoy."
            )

    # -- the graph ----------------------------------------------------------

    @property
    def dim(self) -> int:
        """`d`: clique vertices, plus the star's hub and leaves."""

        return self.clique_size + 1 + self.star_leaves

    def clique_vertices(self) -> list[int]:
        """The maximum clique, which is the answer's support."""

        return list(range(self.clique_size))

    def hub_vertex(self) -> int:
        """The star's centre: the graph's highest-degree vertex, and a trap."""

        return self.clique_size

    def adjacency(self) -> Tensor:
        """`A`: `K_clique_size` and `K_{1,star_leaves}`, disjoint. 0/1, symmetric."""

        size = self.dim
        matrix = torch.zeros(size, size, dtype=DTYPE)
        for first in self.clique_vertices():
            for second in self.clique_vertices():
                if first != second:
                    matrix[first, second] = 1.0
        hub = self.hub_vertex()
        for leaf in range(hub + 1, size):
            matrix[hub, leaf] = 1.0
            matrix[leaf, hub] = 1.0
        return matrix

    def clique_spectral_radius(self) -> float:
        """`clique_size - 1`, the adjacency eigenvalue of a complete graph."""

        return float(self.clique_size - 1)

    def star_spectral_radius(self) -> float:
        """`sqrt(star_leaves)`, the adjacency eigenvalue of a star."""

        return float(self.star_leaves**0.5)

    def spectral_radius(self) -> float:
        """`lambda_max(A)`: the star's, by construction, not the clique's."""

        return max(self.clique_spectral_radius(), self.star_spectral_radius())

    def client_perturbations(self) -> Tensor:
        """The `E_i`, shape `(n, d, d)`, symmetric, zero-diagonal, summing to zero.

        Built as +/- pairs of a fixed pattern so the sum is exactly the zero
        matrix in floating point, which is what keeps `A` -- and so `F*`, `x*`
        and the Motzkin-Straus value -- exact for the federated objective.
        """

        size = self.dim
        rows: list[Tensor] = []
        for pair in range(self.num_clients // 2):
            base = torch.zeros(size, size, dtype=DTYPE)
            for first in range(size):
                for second in range(first + 1, size):
                    sign = -1.0 if (first + second + pair) % 2 else 1.0
                    base[first, second] = self.heterogeneity * sign
                    base[second, first] = self.heterogeneity * sign
            rows.extend((base, -base))
        if self.num_clients % 2:
            rows.append(torch.zeros(size, size, dtype=DTYPE))
        return torch.stack(rows)

    def client_adjacencies(self) -> Tensor:
        """The `A_i = A + E_i`, shape `(n, d, d)`. This is the stored data."""

        return self.adjacency() + self.client_perturbations()

    # -- the reference optimum ----------------------------------------------

    def clique_number(self) -> int:
        """`omega(G) = clique_size`: the star's largest clique is an edge."""

        return max(self.clique_size, 2)

    def optimum(self) -> Tensor:
        """`x*`: uniform on the maximum clique, `1/omega` on its vertices."""

        values = torch.zeros(self.dim, dtype=DTYPE)
        for vertex in self.clique_vertices():
            values[vertex] = 1.0 / self.clique_size
        return values

    def optimal_objective(self) -> float:
        """`F* = -(1/2)(1 - 1/omega)`, from Motzkin-Straus."""

        return -0.5 * (1.0 - 1.0 / self.clique_number())

    def star_local_objective(self) -> float:
        """`-(1/2)(1 - 1/2)`: what the decoy component is worth.

        The star's maximal cliques are its 25 edges, so its best simplex value
        is `1 - 1/2`. An arm that follows the spectral radius arrives here, and
        it is 0.15 short of `F*`.
        """

        return -0.5 * (1.0 - 1.0 / 2)

    def barycentre_objective(self) -> float:
        """`F(1/d)`: where the run starts, and a value a single vertex cannot beat."""

        uniform = torch.full((self.dim,), 1.0 / self.dim, dtype=DTYPE)
        return objective(uniform, self.adjacency())

    def objective_at(self, x: Tensor) -> float:
        """`F(x)`, averaged over the stored client matrices."""

        matrices = self.client_adjacencies()
        return sum(objective(x, matrix) for matrix in matrices) / len(matrices)

    def mass_on_clique(self, x: Tensor) -> float:
        """How much of `x`'s mass sits on the maximum clique. 1.0 at `x*`."""

        return float(x[self.clique_vertices()].sum())

    def maximal_clique_count(self) -> int:
        """Maximal cliques: the `K_5`, plus one per star edge.

        Not a count of local maximisers. The star's edges all lie on one
        connected face of maximisers at `-0.25` -- half the mass on the hub, the
        rest anywhere on the leaves -- so they are points of one optimum set,
        not separate optima.
        """

        return 1 + self.star_leaves


# ---------------------------------------------------------------------------
# The generator: one noisy adjacency per client, written as shards a run reads
# ---------------------------------------------------------------------------

#: The name of this problem's generator and task, as configs write it.
DATASET_NAME = "nonconvex_simplex"


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
        raise ValueError("nonconvex_simplex needs partition.num_clients: the number of views")
    return ProblemSpec(
        num_clients=int(partition["num_clients"]),
        clique_size=int(problem.get("clique_size", 5)),
        star_leaves=int(problem.get("star_leaves", 25)),
        heterogeneity=float(problem.get("heterogeneity", 0.1)),
    )


def _spec_from_reference(reference: Mapping[str, Any]) -> ProblemSpec:
    """Rebuild the spec from the dials the manifest's ``reference`` records.

    The generator and the task are the same module, and the spec is a
    deterministic function of its dials, so the task rebuilds exactly what the
    generator wrote -- the graph, the client views, `x*` -- rather than
    carrying `n x d x d` floats through JSON to get the same tensors back.
    """

    problem = reference["problem"]
    return ProblemSpec(
        num_clients=int(problem["clients"]),
        clique_size=int(problem["clique_size"]),
        star_leaves=int(problem["star_leaves"]),
        heterogeneity=float(problem["heterogeneity"]),
    )


def reference_of(spec: ProblemSpec) -> dict[str, Any]:
    """Everything a run on this data is scored against, in closed form.

    Written into the manifest, copied into ``run.json`` by
    ``run_metadata.build_dataset_provenance``, and read back by
    :class:`NonconvexSimplexTask`. The two spectral radii are here because the
    decoy is the whole design of the graph, and a reader of ``run.json``
    should be able to see that the leading eigenvector points away from the
    answer without opening this file.
    """

    return {
        "problem": {
            "clients": spec.num_clients,
            "clique_size": spec.clique_size,
            "star_leaves": spec.star_leaves,
            "heterogeneity": spec.heterogeneity,
        },
        "dim": spec.dim,
        "clique_number": spec.clique_number(),
        "x_star": spec.optimum().tolist(),
        "f_star": spec.optimal_objective(),
        "star_local_objective": spec.star_local_objective(),
        "barycentre_objective": spec.barycentre_objective(),
        "clique_spectral_radius": spec.clique_spectral_radius(),
        "star_spectral_radius": spec.star_spectral_radius(),
        "maximal_clique_count": spec.maximal_clique_count(),
        "constrained": True,
        "unconstrained_objective_bounded_below": False,
        "projection_available_in_fedbrew": False,
    }


def generate_nonconvex_simplex_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, plus the manifest a run reads.

    The four arguments are the generator contract (chapter 12 §4). Two of
    them do nothing here, and say so rather than being quietly dropped:

    ``seed``
        The client views are a deterministic function of the spec -- a fixed
        perturbation pattern in exact +/- pairs -- so there is no draw for a
        seed to fix. Recorded in the manifest anyway, because every other
        dataset's provenance says what seed it was made at.

    ``client_splits``
        The ratios describe a cut, and there is nothing to cut: a client's
        objective *is* its matrix. All three splits hold the same single
        matrix, which the manifest declares as
        ``client_test_source: identical_to_train``.
    """

    del client_splits
    spec = _spec_from_config(config)
    matrices = spec.client_adjacencies()
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    clients: list[dict[str, Any]] = []
    for index, matrix in enumerate(matrices):
        client_id = f"client_{index}"
        # One row, whose "feature" is a whole d x d matrix. The second element
        # is a dummy target nothing reads -- f_i needs no label -- but a shard
        # is an (x, y) pair everywhere else.
        row = matrix.reshape(1, spec.dim, spec.dim).clone()
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
                "edge_weight_sum": float(matrix.sum()),
            }
        )

    # The server's central pass evaluates F in one go, so the pooled shard is
    # every client's matrix stacked -- exactly the federated objective, because
    # every client carries the same weight.
    save_client_shard(
        shards_dir / "global_test.pt",
        matrices.clone(),
        torch.zeros(len(matrices), dtype=DTYPE),
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
        "partition_key": "view_index",
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
    """Write the two files ``fedbrew inspect-data`` reads beside the manifest."""

    payload = {
        "dataset_name": DATASET_NAME,
        "partition_strategy": "analytic",
        "num_clients": len(clients),
        "total_examples": sum(int(client["num_examples"]) for client in clients),
        "min_examples_per_client": 3,
        "max_examples_per_client": 3,
        "mean_examples_per_client": 3.0,
        "clique_size": spec.clique_size,
        "star_leaves": spec.star_leaves,
        "heterogeneity": spec.heterogeneity,
        "clique_spectral_radius": spec.clique_spectral_radius(),
        "star_spectral_radius": spec.star_spectral_radius(),
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
        writer.writerow(["client_id", "num_examples", "edge_weight_sum"])
        for client in clients:
            writer.writerow(
                [client["client_id"], client["num_examples"], f"{client['edge_weight_sum']:.17g}"]
            )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class SimplexPointModel(nn.Module):  # type: ignore[misc]
    """The iterate `x`, as a `d`-element `nn.Parameter`. Nothing constrains it."""

    def __init__(self, dim: int, x_init: str = "uniform") -> None:
        """Start at the barycentre, which is feasible.

        Args:
            dim: Problem dimension `d`, from ``model.input_dim``.
            x_init: ``uniform`` starts at `1/d`. Unlike
                ``examples/simplex-lsq``, `d` is 31 and not a power of two, so
                `d * (1/d)` is 1.0 only to rounding and the initial
                ``constraint_violation`` is around 1e-16 rather than exactly 0.
                That is the arithmetic, and it is six orders of magnitude below
                anything else this example reports.
        """

        super().__init__()
        if x_init != "uniform":
            raise ValueError(f"model.x_init must be 'uniform', got {x_init!r}")
        super().__setattr__("x", nn.Parameter(torch.full((dim,), 1.0 / dim, dtype=DTYPE)))

    def forward(self, graphs: Tensor) -> Tensor:
        """Return `f_i(x)` per matrix in the batch; shape `(N,)`."""

        return -0.5 * torch.einsum("j,njk,k->n", self.x, graphs, self.x)

    @property
    def iterate(self) -> Tensor:
        """The vector the whole example is about, detached."""

        return self.x.detach().clone()


def build_simplex_point(config: Mapping[str, Any] | None = None) -> SimplexPointModel:
    """Registry builder for `model.name: simplex_point`.

    Builds from the model block alone. Whether `d` agrees with the generated
    graph is checked by :class:`NonconvexSimplexTask`, the one object handed
    both the model block and the manifest.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _MODEL_KEYS, "simplex_point")
    dim = values.get("input_dim")
    if dim is None:
        raise ValueError("model.input_dim is required for simplex_point: it is the problem's d")
    return SimplexPointModel(dim=int(dim), x_init=str(values.get("x_init", "uniform")))


# ---------------------------------------------------------------------------
# The task adapter
# ---------------------------------------------------------------------------


class NonconvexSimplexTask(TaskAdapter):
    """Bridge between the constrained non-convex problem and the FL loop."""

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
        ignored, because one matrix per client has no throughput question and
        the loader is built per call.

        Args:
            model_config: The resolved ``model`` block. Carries `d`.
            dataset_metadata: The manifest, as ``manifest_dataset`` reports it.
                Carries ``reference``, written by the generator.
            device: The resolved ``runtime.device``.
            **unused: The rest of the contract.

        Raises:
            ValueError: If the model block's `d` is not the generated graph's.
                `d` is derived from the graph rather than chosen, so this is
                the one place a reader who edited the model block instead of
                the generator config finds out.
        """

        del unused
        self.device = torch.device(device)
        self.reference = dict((dataset_metadata or {}).get("reference") or {})
        if "problem" not in self.reference:
            raise ValueError(
                "nonconvex_simplex task needs a manifest written by its own "
                "generator: the reference optimum lives in the manifest's "
                "`reference` and this data has none."
            )
        spec = _spec_from_reference(self.reference)
        _check_model_against_reference(model_config or {}, spec)
        self.spec = spec
        self._optimum = spec.optimum().to(self.device)
        self._optimal_objective = spec.optimal_objective()
        # Built once. `ProblemSpec` is frozen and rebuilds the (n, d, d) stack
        # on every call, and `eval_step` evaluates the global objective twice
        # per batch -- once at the iterate and once at its projection -- so
        # calling through the spec made the run an order of magnitude slower
        # than the arithmetic warrants.
        self._matrices = spec.client_adjacencies().to(self.device)
        self._criterion = _mean_of_batch
        # Read as `getattr(task, "_scaler", None)` by four client rules.
        self._scaler: Any = None

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> SimplexPointModel:
        """Build the vector model named by the `model` config block."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        factory = models.get(str(values.get("name", "simplex_point")))
        model = factory(values)
        return model.to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Cut one client's matrices into batches. A list, so it is re-iterable.

        The second element of each batch is a dummy target nothing reads: `f_i`
        needs no label, but every client rule destructures a batch into two.
        """

        loader_config = _loader_config(config)
        graphs = _graphs_of(data).to(self.device)
        rows = len(graphs)
        batch_size = max(1, int(loader_config.get("batch_size", rows) or rows))
        if bool(loader_config.get("shuffle", False)):
            graphs = graphs[_permutation(rows, loader_config.get("seed"))]
        batches = [graphs[start : start + batch_size] for start in range(0, rows, batch_size)]
        if bool(loader_config.get("drop_last", False)) and len(batches) > 1:
            batches = [batch for batch in batches if len(batch) == batch_size]
        return [(batch, torch.zeros(len(batch), dtype=DTYPE)) for batch in batches]

    def train_step(
        self,
        model: SimplexPointModel,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Take one *unconstrained* step. There is nowhere to put the constraint."""

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        graphs, targets = self._move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = self._criterion(model(graphs), targets)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def eval_step(self, model: SimplexPointModel, batch: Any) -> dict[str, float]:
        """Measure the batch's objective, and seven properties of the iterate."""

        model.eval()
        graphs, _ = self._move_batch(batch)
        with torch.no_grad():
            values = model(graphs)
            iterate = model.iterate
            projected = project_simplex(iterate)
        return {
            "loss": float(values.mean()),
            "total": float(len(graphs)),
            "optimality_gap": self._global_objective(iterate) - self._optimal_objective,
            "feasible_gap": self._global_objective(projected) - self._optimal_objective,
            "distance_to_optimum": float(torch.linalg.vector_norm(iterate - self._optimum)),
            "constraint_violation": float(torch.linalg.vector_norm(iterate - projected)),
            "simplex_sum": float(iterate.sum()),
            "iterate_norm": float(torch.linalg.vector_norm(iterate)),
            "mass_on_clique": self.spec.mass_on_clique(projected),
        }

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Fold eval-step outputs into the eight numbers this task reports.

        ``loss`` / ``optimality_gap``
            `F(x)` and `F(x) - F*`, both heading to `-inf`. There is no
            unconstrained minimum, so these are not measuring convergence to
            anything; they are measuring how far the run has left.

        ``feasible_gap``
            `F(Proj_Delta(x)) - F*`. **This one goes up**, to `+0.4`, because a
            large iterate projects onto its single largest coordinate and the
            largest coordinate is the star's hub, which spans no edge. Post-hoc
            projection is not a rescue here, and this is the column that says so.

        ``iterate_norm`` / ``constraint_violation`` / ``simplex_sum``
            `||x||`, the distance to the simplex, and the sum. All three start at
            about 1, 0 and 1 and end around 1e80.

        ``mass_on_clique``
            How much of the *projected* iterate sits on the maximum clique. 1.0
            would mean the run found the answer; it goes to 0.
        """

        records = [record for record in outputs if isinstance(record, Mapping)]
        names = (
            "loss",
            "optimality_gap",
            "feasible_gap",
            "distance_to_optimum",
            "constraint_violation",
            "simplex_sum",
            "iterate_norm",
            "mass_on_clique",
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

    def evaluate_model(self, model: SimplexPointModel, data: Any) -> dict[str, float]:
        """Measure the global model on every client's matrix at once."""

        outputs = [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        return self.compute_metrics(outputs)

    # -- this task's own helpers --------------------------------------------

    def _global_objective(self, x: Tensor) -> float:
        """`F(x)` over the cached client matrices; the same value as
        ``ProblemSpec.objective_at`` and about a hundred times cheaper."""

        return float(-0.5 * torch.einsum("j,njk,k->n", x, self._matrices, x).mean())

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Split a batch into (graphs, ignored targets), both on the device."""

        graphs, targets = batch
        return graphs.to(self.device), targets.to(self.device)


def _mean_of_batch(outputs: Tensor, targets: Tensor | None = None) -> Tensor:
    """The batch objective: the mean of `f_i(x)` over the batch's matrices."""

    del targets
    return outputs.mean()


def _graphs_of(data: Any) -> Tensor:
    """The adjacency matrices in one split of a shard, as `(N, d, d)`.

    They arrive as a shard's ``x``, because that is what ``manifest_dataset``
    hands over and what every other dataset in the repository stores.
    """

    if isinstance(data, Mapping):
        for key in ("x", "X", "graphs"):
            value = data.get(key)
            if isinstance(value, Tensor):
                return value.to(DTYPE)
    if isinstance(data, Tensor):
        return data.to(DTYPE)
    raise ValueError("nonconvex_simplex data must be a shard mapping carrying an 'x' tensor")


def _check_model_against_reference(model_config: Mapping[str, Any], spec: ProblemSpec) -> None:
    """Refuse a model block that describes a different graph than the data."""

    dim = int(model_config.get("input_dim", 0))
    if dim == spec.dim:
        return
    raise ValueError(
        f"model config describes d={dim}, but the generated graph is d={spec.dim} "
        f"(manifest reference: clique {spec.clique_size} + hub + {spec.star_leaves} "
        "leaves). d is a property of the graph, so it is set by the generator "
        "config and never by this block."
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
TASK_NAME = "nonconvex_simplex"
MODEL_NAME = "simplex_point"


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
        generate_nonconvex_simplex_from_config,
        sections={"problem": {"clique_size", "star_leaves", "heterogeneity"}},
    )
    registry.tasks.register(TASK_NAME, lambda **kwargs: NonconvexSimplexTask(**kwargs))
    registry.models.register(MODEL_NAME, build_simplex_point, task=TASK_NAME)


def _self_check() -> None:
    """Assert the six claims this module's docstring makes about the problem."""

    spec = ProblemSpec()
    adjacency = spec.adjacency()

    if not torch.equal(adjacency, adjacency.T):
        raise AssertionError("the adjacency is not symmetric")
    if float(adjacency.diagonal().abs().max()) != 0.0:
        raise AssertionError("the adjacency has a non-zero diagonal")

    eigenvalues = torch.linalg.eigvalsh(adjacency)
    if float(eigenvalues.min()) >= 0.0 or float(eigenvalues.max()) <= 0.0:
        raise AssertionError("the adjacency should be indefinite: F is not convex")
    if abs(float(eigenvalues.max()) - spec.star_spectral_radius()) > 1e-12:
        raise AssertionError("the spectral radius should be the star's, not the clique's")
    if spec.star_spectral_radius() <= spec.clique_spectral_radius():
        raise AssertionError("the decoy does not out-rank the clique spectrally")

    perturbations = spec.client_perturbations()
    if float(perturbations.sum(dim=0).abs().max()) != 0.0:
        raise AssertionError("client perturbations must sum to exactly zero")

    optimum = spec.optimum()
    if abs(float(optimum.sum()) - 1.0) > 1e-14 or float(optimum.min()) < 0.0:
        raise AssertionError("x* is not on the simplex")
    if abs(spec.objective_at(optimum) - spec.optimal_objective()) > 1e-14:
        raise AssertionError("F(x*) does not match the Motzkin-Straus value")
    # Motzkin-Straus is a maximum, so nothing feasible may beat it. Checked
    # against the two competitors the graph was built to have.
    if spec.star_local_objective() <= spec.optimal_objective():
        raise AssertionError("the decoy should be worse than the clique, not better")
    if spec.barycentre_objective() <= spec.optimal_objective():
        raise AssertionError("the starting point should be worse than the optimum")

    if _spec_from_reference(reference_of(spec)) != spec:
        raise AssertionError("the spec does not survive the round trip through the manifest")

    model = SimplexPointModel(dim=spec.dim)
    batch = spec.client_adjacencies()[:2]
    _mean_of_batch(model(batch)).backward()
    measured = model.x.grad
    if measured is None:
        raise AssertionError("the backward pass left no gradient on the iterate")
    expected = sum(gradient(model.iterate, matrix) for matrix in batch) / len(batch)
    if float((measured - expected).abs().max()) > 1e-14:
        raise AssertionError("autograd disagrees with the analytic gradient")


_self_check()
