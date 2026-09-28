"""Federated logistic regression with an L1 penalty, over a planted sparse signal.

The problem
-----------
Rows `(a_i, b_i)` with `a_i` in `R^d` and `b_i` in `{-1, +1}`. The objective is::

    F(x) = (1/n) sum_{i=1}^n log(1 + exp(-b_i x' a_i)) + lam ||x||_1

over `x` in `R^d`, with **no constraint set**: `log(1 + exp(.))` is bounded
below by 0 and `lam ||x||_1 -> infinity`, so `F` is coercive and a minimiser
exists without a box. Writing `l(x)` for the first term::

    grad l(x) = -(1/n) sum_i b_i sigma(-b_i x' a_i) a_i
    F = l + lam ||.||_1,   l smooth and convex, the penalty neither.

The loss and the penalty are named (``problem.loss``, ``problem.penalty`` and
the model's keys of the same names), and a problem is the pair of them with
`lam`: :data:`LOSSES` and :data:`PENALTIES` hold the forms this file defines.

How it is federated
-------------------
Clients `c = 1..N` hold disjoint index sets `I_c` with `m_c = |I_c|` rows each,
and client `c` optimises::

    F_c(x) = (1/m_c) sum_{i in I_c} log(1 + exp(-b_i x' a_i)) + lam ||x||_1

with the **whole** penalty in every client, as ``examples/fed-lasso`` does.
Every client holds the same `m = rows_per_client` rows, so uniform and
example-weighted aggregation are the same average and both reproduce `F`
exactly. ``_self_check`` asserts both spellings against the stored rows.

The data
--------
No draw anywhere. The design and the labels come from low-discrepancy
sequences:

``design``
    `a_ij = Phi^-1(vdc(i, p_j))`, the Halton sequence through the normal
    quantile, with `p_j` the `j`-th prime.

``labels``
    `b_i = +1` if `sigma(x_true . a_i) > vdc(i, q)` else `-1`, with
    `q = p_d` -- **the first prime the design does not use**. A threshold
    sequence in a base a column already uses is a function of the same index
    in the same base, and correlates with that column: base 3 puts a false
    positive into the support at coordinate 1, and base 2 corrupts `x*` while
    the support still matches (measured 2026-09-20).

``x_true``
    ``sparsity`` non-zeros at evenly spaced coordinates, magnitudes halving,
    signs alternating -- fed-lasso's planting rule -- scaled by
    ``signal_scale``.

``clients``
    The rows sorted by their margin `x_true . a_i` (a stable sort) and dealt in
    blocks of ``partition_block`` rows, round robin: client `c` holds blocks
    `c, c + N, c + 2N, ...`. 1 is a stratified deal and `rows_per_client` one
    contiguous band of margins per client; the block is the heterogeneity dial.

The reference optimum
---------------------
`x*` has no closed form. It is solved once, at generation -- FISTA to identify
the support, then Newton on it -- and certified by its KKT residual on the
full vector, which generation refuses above :data:`CERTIFICATE`. `x*`, `F*`
and the residual are written into the manifest, and the task reads them as
values: the one structural difference from fed-lasso, whose task re-derives
its closed form.

What this file registers
------------------------
Three names, at ``register()`` time, and nothing at import::

    generators  "fed_logistic_l1"    writes the shards and the manifest
    tasks       "fed_logistic_l1"    FedLogisticL1Task
    models      "logistic_vector"    LogisticModel, a d-vector plus its penalty

The problem's `lam`, loss and penalty are on the model as well as in the data,
because they are part of the objective the client descends and
``model.extra`` is where a run config carries them; the task is the one object
handed both, so it is where they are cross-checked.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, optim

from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import (
    load_client_shard,
    save_client_shard,
    save_split_client_shard,
)
from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.tasks.base import (
    LoaderOrder,
    TaskAdapter,
    batch_row_numbers,
    listed_loader_order,
    row_count,
    row_mean,
    row_numbers,
)

#: Every tensor here is float64: so that ``exact_zeros`` means what it says, and
#: so that the reference solve reaches a KKT residual near 1e-17 rather than
#: near float32's epsilon.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Deterministic sequences: the design and the labels, with no draw
# ---------------------------------------------------------------------------


def primes(count: int) -> list[int]:
    """The first `count` primes, by trial division. `count` is `d + 1` here."""

    found: list[int] = []
    candidate = 2
    while len(found) < count:
        if all(candidate % prime for prime in found):
            found.append(candidate)
        candidate += 1
    return found


def van_der_corput(count: int, base: int) -> Tensor:
    """`vdc(i, base)` for `i = 1..count`, as a float64 tensor in `(0, 1)`.

    The radical inverse, digits accumulated lowest first, so the order of the
    float64 additions is fixed by the sequence and not by how the loop is
    written. Indices start at 1 because the normal quantile of 0 is `-inf`.
    """

    index = torch.arange(1, count + 1, dtype=torch.int64)
    value = torch.zeros(count, dtype=DTYPE)
    denominator = 1.0
    while bool(index.any()):
        denominator *= base
        value += (index % base).to(DTYPE) / denominator
        index = index // base
    return value


def halton_normal_design(rows: int, dim: int) -> Tensor:
    """`a_ij = Phi^-1(vdc(i, p_j))`: the design, as a pure function of `(n, d)`."""

    columns = [van_der_corput(rows, prime) for prime in primes(dim)]
    return torch.special.ndtri(torch.stack(columns, dim=1))


def label_base(dim: int) -> int:
    """`p_d`: the first prime the design does not use (see the module docstring)."""

    return primes(dim + 1)[dim]


def deal(key: Tensor, clients: int, rows_per_client: int, block: int) -> list[Tensor]:
    """Rows sorted by `key` (stable), dealt in blocks of `block`, round robin."""

    order = torch.argsort(key, stable=True)
    return [
        torch.cat(
            [
                order[(group * clients + client) * block : (group * clients + client + 1) * block]
                for group in range(rows_per_client // block)
            ]
        )
        for client in range(clients)
    ]


# ---------------------------------------------------------------------------
# The objective: a loss of the margin, and a penalty of the iterate
# ---------------------------------------------------------------------------


def _logistic(signed: Tensor, mask: Tensor | None = None) -> Tensor:
    """`mean log(1 + exp(z))` over the rows, `z = -b a.x`, through ``softplus``."""

    return row_mean(torch.nn.functional.softplus(signed), mask)


def _logistic_weights(signed: Tensor) -> Tensor:
    """`d loss_i / d z_i`: `sigma(z)`."""

    return torch.sigmoid(signed)


#: The losses of the margin, each as (the mean over a batch's rows of `z =
#: -b a.x`, its derivative in `z` per row, whether it is convex, and the bound
#: on its second derivative that makes `grad l` Lipschitz with `L = bound *
#: ||A||^2 / n`).
LOSSES: dict[str, tuple[Callable[..., Tensor], Callable[[Tensor], Tensor], bool, float]] = {
    "logistic": (_logistic, _logistic_weights, True, 0.25),
}


def _l1(x: Tensor, lam: float) -> Tensor:
    return lam * x.abs().sum()


def _l1_gradient(x: Tensor, lam: float) -> Tensor:
    # sign(0) = 0: the minimum-norm subgradient, and what autograd gives |x|.
    return lam * torch.sign(x)


#: The penalties, each as (its value, its (sub)gradient, whether it is convex,
#: and the bound on its second derivative, None where it has none).
PENALTIES: dict[
    str, tuple[Callable[[Tensor, float], Tensor], Callable[[Tensor, float], Tensor], bool, Any]
] = {
    "l1": (_l1, _l1_gradient, True, None),
}


#: The problems this example poses, as (loss, penalty) pairs: each has
#: generator configs and an arm, and is held by the tests batched against
#: sequential.
PROBLEMS: tuple[tuple[str, str], ...] = (("logistic", "l1"),)


def convex(loss: str, penalty: str) -> bool:
    """Whether a problem is convex, and so has a certified `F*`."""

    return LOSSES[loss][2] and PENALTIES[penalty][2]


def penalty_value(x: Tensor, lam: float, penalty: str = "l1") -> Tensor:
    """The penalty at `x`, as a scalar tensor."""

    return PENALTIES[penalty][0](x, lam)


def mean_loss(
    outputs: Tensor, labels: Tensor, loss: str = "logistic", mask: Tensor | None = None
) -> Tensor:
    """The mean loss of a batch's margins `outputs = A_B x`."""

    return LOSSES[loss][0](-labels * outputs, mask)


def smooth_gradient(x: Tensor, features: Tensor, labels: Tensor, loss: str = "logistic") -> Tensor:
    """`grad l(x) = -(1/n) A' (b * loss'(z))`, the analytic gradient of the mean loss."""

    weights = LOSSES[loss][1](-labels * (features @ x))
    return features.T @ (-labels * weights) / len(labels)


def objective(
    x: Tensor,
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    loss: str = "logistic",
    penalty: str = "l1",
) -> float:
    """`F(x)` over the rows given, as a plain float."""

    return float(mean_loss(features @ x, labels, loss)) + float(
        penalty_value(x, penalty_strength, penalty)
    )


def gradient(
    x: Tensor,
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    loss: str = "logistic",
    penalty: str = "l1",
) -> Tensor:
    """The analytic (sub)gradient of `F`: the smooth part plus the penalty's."""

    return smooth_gradient(x, features, labels, loss) + PENALTIES[penalty][1](x, penalty_strength)


def kkt_residual(
    x: Tensor, features: Tensor, labels: Tensor, penalty_strength: float, penalty: str = "l1"
) -> float:
    """How far `x` is from the optimality conditions of the logistic problem.

    For the L1 penalty, per coordinate: `|grad l(x)_k + lam sign(x_k)|` where
    `x_k != 0`, and `max(|grad l(x)_k| - lam, 0)` where it is 0. Zero exactly at
    the minimiser, and with `l` convex a point at residual `r` has
    `F(x) - F* <= r ||x - x*||_1`: the whole certificate.
    """

    smooth = smooth_gradient(x, features, labels)
    if penalty == "l1":
        violations = torch.where(
            x != 0.0,
            (smooth + penalty_strength * torch.sign(x)).abs(),
            torch.clamp(smooth.abs() - penalty_strength, min=0.0),
        )
        return float(violations.max())
    raise ValueError(f"no certificate for the {penalty!r} penalty")


def soft_threshold(values: Tensor, level: float) -> Tensor:
    """`S_t(v)_j = sign(v_j) max(0, |v_j| - t)`, the L1 proximal operator.

    Here to *compute* the reference solution and for nothing else: nothing
    under ``fedbrew/clients/`` or ``fedbrew/servers/`` applies it.
    """

    return torch.sign(values) * torch.clamp(values.abs() - level, min=0.0)


def support_of(values: Tensor, tolerance: float) -> set[int]:
    """The coordinates a run would call non-zero at `tolerance`."""

    return {index for index, value in enumerate(values.tolist()) if abs(value) > tolerance}


def lipschitz_of(features: Tensor) -> float:
    """`||A||_2^2 / 4n`: the Lipschitz constant of the logistic loss's gradient."""

    return float(torch.linalg.matrix_norm(features, 2)) ** 2 / (4.0 * features.shape[0])


# ---------------------------------------------------------------------------
# The reference solver: FISTA to find the support, Newton on it to finish
# ---------------------------------------------------------------------------

#: What a certified solve must reach on the full vector. Generation refuses a
#: reference above it: `x*` is what every optimality gap on the data is
#: measured against, and a wrong one is a different problem.
CERTIFICATE = 1.0e-12

#: How still the iterate must be, beside an unchanged support, before FISTA
#: hands over to the polish in a certified solve. Loose on purpose: FISTA is
#: there to identify the support, and the certificate on the full vector is
#: what makes the answer right.
SETTLED_MOVE = 1.0e-3

#: A certified solve's settle counters: the support must hold for this many
#: iterations, inside a budget of this many per block. Short blocks, because
#: each ends in a polish and the polish is where a coordinate leaves the
#: support (on Gisette at `lam = 5e-5`, blocks of 5,000 certify in three).
SETTLE_ITERATIONS = 1_000
CERTIFIED_FISTA_ITERATIONS = 5_000


def solve_reference(
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    iterations: int = 20_000,
    polish_steps: int = 60,
    settle: int | None = None,
    rounds: int = 12,
) -> tuple[Tensor, float]:
    """`(x*, kkt_residual(x*))` for the pooled L1-logistic problem.

    **FISTA** at the step `1/L` identifies the support -- the proximal step
    sets coordinates to exactly zero -- and **Newton on the support**, with its
    signs fixed, takes the residual to machine precision. The residual is
    computed on the full vector, so the off-support condition is checked
    rather than assumed.

    ``settle`` makes it a *certified* solve: FISTA stops once the support has
    held still, and the pair repeats, up to ``rounds`` times, while the
    residual is above :data:`CERTIFICATE` -- the next FISTA block is what lets
    a violating coordinate back in. The polish is a side branch off FISTA's own
    iterate, never a point FISTA resumes from: a point fitted under the wrong
    support is a worse start for a proximal method than the unpolished one.
    """

    dim = features.shape[1]
    step = 1.0 / lipschitz_of(features)
    robust = settle is not None
    current = torch.zeros(dim, dtype=DTYPE)
    polished, residual = current, float("inf")
    for _ in range(rounds if robust else 1):
        current = _fista(features, labels, penalty_strength, step, iterations, current, settle)
        polished = _polish(
            features, labels, penalty_strength, current.clone(), polish_steps, robust
        )
        residual = kkt_residual(polished, features, labels, penalty_strength)
        if not robust or residual <= CERTIFICATE:
            break
    return polished, residual


def _fista(
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    step: float,
    iterations: int,
    start: Tensor,
    settle: int | None = None,
) -> Tensor:
    """FISTA from `start`, for `iterations` steps or until the support settles.

    With ``settle`` it also restarts adaptively (O'Donoghue and Candes): when
    the momentum has carried the iterate uphill the weight is reset to 1.
    """

    current = start.clone()
    momentum = current.clone()
    weight = 1.0
    unchanged, moved = 0, 0.0
    for _ in range(iterations):
        candidate = momentum - step * smooth_gradient(momentum, features, labels)
        following = soft_threshold(candidate, step * penalty_strength)
        if settle is not None and float(((momentum - following) * (following - current)).sum()) > 0:
            weight = 1.0
        next_weight = 0.5 * (1.0 + (1.0 + 4.0 * weight * weight) ** 0.5)
        momentum = following + ((weight - 1.0) / next_weight) * (following - current)
        if settle is not None:
            same = bool(torch.equal(following != 0.0, current != 0.0))
            unchanged = unchanged + 1 if same else 0
            moved = float((following - current).abs().max())
        current, weight = following, next_weight
        if settle is not None and unchanged >= settle and moved < SETTLED_MOVE:
            break
    return current


def _polish(
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    current: Tensor,
    polish_steps: int,
    robust: bool = False,
) -> Tensor:
    """Newton on the identified support, with its signs held fixed.

    ``robust`` is for data whose support Hessian can be singular -- a corpus
    may carry a column twice, and Gisette does: the step is then taken in the
    Hessian's range (:func:`_newton_step`), and a coordinate the step pushes
    across its own sign leaves the support (:func:`_drop_flipped`).
    """

    rows = features.shape[0]
    support = torch.nonzero(current, as_tuple=False).ravel().tolist()
    if not support:
        return current
    signs = torch.sign(current[support])
    for _ in range(polish_steps):
        residual = smooth_gradient(current, features, labels)[support]
        residual = residual + penalty_strength * signs
        if float(residual.abs().max()) < 1e-18:
            break
        scores = features[:, support]
        probabilities = torch.sigmoid(-labels * (features @ current))
        curvature = probabilities * (1.0 - probabilities)
        hessian = (scores * curvature.unsqueeze(1)).T @ scores / rows
        current[support] -= _newton_step(hessian, residual, robust)
        if robust:
            support, signs = _drop_flipped(current, support, signs)
            if not support:
                break
    return current


def _drop_flipped(current: Tensor, support: list[int], signs: Tensor) -> tuple[list[int], Tensor]:
    """Zero every coordinate the last Newton step pushed across its own sign."""

    index = torch.tensor(support)
    flipped = torch.sign(current[index]) != signs
    if not bool(flipped.any()):
        return support, signs
    current[index[flipped]] = 0.0
    return index[~flipped].tolist(), signs[~flipped]


def _newton_step(hessian: Tensor, residual: Tensor, robust: bool) -> Tensor:
    """`H^-1 r`, or its minimum-norm reading when `H` is singular."""

    if not robust:
        return torch.linalg.solve(hessian, residual)
    values, vectors = torch.linalg.eigh(hessian)
    largest = float(values.max())
    if largest <= 0.0:
        return torch.zeros_like(residual)
    inverted = torch.where(values > largest * 1.0e-12, 1.0 / values, torch.zeros_like(values))
    return vectors @ (inverted * (vectors.T @ residual))


def certified_optimum(
    features: Tensor, labels: Tensor, penalty_strength: float
) -> tuple[Tensor, float]:
    """`(x*, residual)` of the L1 problem, run until it is certified."""

    return solve_reference(
        features,
        labels,
        penalty_strength,
        iterations=CERTIFIED_FISTA_ITERATIONS,
        settle=SETTLE_ITERATIONS,
    )


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("penalty_strength", "loss", "penalty", "support_tolerance", "x_init")


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """The whole problem, as its dials, plus everything they determine.

    Everything here but :meth:`optimum` is closed form and cheap. The optimum
    is a solve, computed once at generation and read back from the manifest.
    """

    num_clients: int = 32
    #: `d`. Picks how many primes the design uses.
    dim: int = 32
    #: `m`, the rows each client holds; `n = num_clients * rows_per_client`.
    rows_per_client: int = 64
    #: Non-zeros in `x_true`, at evenly spaced coordinates.
    sparsity: int = 3
    #: What the halving magnitudes are multiplied by. At 1.0 the smallest
    #: planted coefficient is 0.25 and the logistic loss shrinks it to 0.005 at
    #: the widest recovering `lam`; at 2.0 the recovering window is `lam` in
    #: `[0.02, 0.07]` (measured 2026-09-20).
    signal_scale: float = 2.0
    #: `lam`.
    penalty_strength: float = 0.03
    #: How the margin-sorted rows are dealt: blocks of this many, round robin.
    partition_block: int = 32
    #: The loss of the margin (:data:`LOSSES`).
    loss: str = "logistic"
    #: The penalty `lam` multiplies (:data:`PENALTIES`).
    penalty: str = "l1"

    def __post_init__(self) -> None:
        """Refuse a spec that cannot express what it claims to."""

        if self.num_clients < 2:
            raise ValueError("partition.num_clients must be at least 2")
        if self.dim < 2:
            raise ValueError("problem.dim must be at least 2")
        if self.rows_per_client < 1:
            raise ValueError("problem.rows_per_client must be at least 1")
        if not 1 <= self.sparsity <= self.dim:
            raise ValueError("problem.sparsity must be between 1 and problem.dim")
        if self.signal_scale <= 0.0:
            raise ValueError("problem.signal_scale must be positive")
        if self.penalty_strength < 0.0:
            raise ValueError("problem.penalty_strength must be non-negative")
        if self.loss not in LOSSES:
            raise ValueError(f"problem.loss must be one of {sorted(LOSSES)}, not {self.loss!r}")
        if self.penalty not in PENALTIES:
            raise ValueError(
                f"problem.penalty must be one of {sorted(PENALTIES)}, not {self.penalty!r}"
            )
        if self.partition_block < 1 or self.rows_per_client % self.partition_block:
            raise ValueError(
                f"problem.partition_block must divide problem.rows_per_client: "
                f"{self.partition_block} does not divide {self.rows_per_client}. The deal "
                "is whole blocks, so a remainder would give some clients fewer rows and "
                "break the equal m_c that makes uniform aggregation exactly F."
            )

    @property
    def rows(self) -> int:
        """`n = N m`, the pooled row count."""

        return self.num_clients * self.rows_per_client

    @property
    def certified(self) -> bool:
        """Whether this problem has a certified `F*`: a convex loss and penalty."""

        return convex(self.loss, self.penalty)

    # -- the ground truth ---------------------------------------------------

    def truth(self) -> Tensor:
        """`x_true`: `sparsity` non-zeros, evenly spaced, magnitudes halving."""

        values = torch.zeros(self.dim, dtype=DTYPE)
        for order in range(self.sparsity):
            index = (order * self.dim) // self.sparsity
            values[index] = self.signal_scale * (-1.0) ** order * 0.5**order
        return values

    def truth_support(self) -> set[int]:
        """The planted support, as coordinate indices."""

        return {(order * self.dim) // self.sparsity for order in range(self.sparsity)}

    # -- the data -----------------------------------------------------------

    def design(self) -> Tensor:
        """`A`, shape `(n, d)`, in source order."""

        return halton_normal_design(self.rows, self.dim)

    def labels(self) -> Tensor:
        """`b`, shape `(n,)`, in `{-1, +1}`: a deterministic Bernoulli draw."""

        probability = torch.sigmoid(self.design() @ self.truth())
        threshold = van_der_corput(self.rows, label_base(self.dim))
        return torch.where(probability > threshold, 1.0, -1.0).to(DTYPE)

    def margin_key(self) -> Tensor:
        """The per-row score the deal sorts by: `x_true . a_i`."""

        return self.design() @ self.truth()

    def client_indices(self) -> list[Tensor]:
        """Which rows each client holds: blocks of the margin order, round robin."""

        return deal(self.margin_key(), self.num_clients, self.rows_per_client, self.partition_block)

    # -- the reference optimum ----------------------------------------------

    def optimum(self, iterations: int = 20_000) -> tuple[Tensor, float]:
        """`(x*, its KKT residual)`. **A solve, not a formula.**

        The fixed-budget solve first; where it stops short of
        :data:`CERTIFICATE`, the certified one.
        """

        features, labels = self.design(), self.labels()
        optimum, residual = solve_reference(
            features, labels, self.penalty_strength, iterations=iterations
        )
        if residual > CERTIFICATE:
            optimum, residual = certified_optimum(features, labels, self.penalty_strength)
        return optimum, residual

    def objective_at(self, x: Tensor) -> float:
        """`F(x)`, over every row. Rebuilds the data; the task caches instead."""

        return objective(
            x, self.design(), self.labels(), self.penalty_strength, self.loss, self.penalty
        )

    def lipschitz(self) -> float:
        """`L = ||A||_2^2 / 4n`, the Lipschitz constant of the logistic `grad l`."""

        return lipschitz_of(self.design())

    def client_objective_at(self, x: Tensor) -> list[float]:
        """`F_c(x)` for each client, from the rows that client holds."""

        features, labels = self.design(), self.labels()
        return [
            objective(
                x, features[index], labels[index], self.penalty_strength, self.loss, self.penalty
            )
            for index in self.client_indices()
        ]

    def client_label_balance(self) -> list[float]:
        """Each client's fraction of `+1` labels: what the deal's block moves."""

        labels = self.labels()
        return [float((labels[index] > 0).to(DTYPE).mean()) for index in self.client_indices()]

    def client_gradient_dispersion(self, optimum: Tensor) -> float:
        """`zeta^2 = mean_c ||grad l_c(x*) - grad l(x*)||^2`."""

        features, labels = self.design(), self.labels()
        pooled = smooth_gradient(optimum, features, labels, self.loss)
        spread = [
            float(
                (
                    (smooth_gradient(optimum, features[index], labels[index], self.loss) - pooled)
                    ** 2
                ).sum()
            )
            for index in self.client_indices()
        ]
        return sum(spread) / len(spread)

    def client_support_sizes(self, iterations: int = 4_000) -> list[int]:
        """How many non-zeros each client's *own* L1-logistic solution has.

        A reported property of the partition, solved at a short budget with no
        polish, and not something a run is scored against.
        """

        features, labels = self.design(), self.labels()
        return [
            int(
                (
                    solve_reference(
                        features[index],
                        labels[index],
                        self.penalty_strength,
                        iterations=iterations,
                        polish_steps=0,
                    )[0]
                    != 0.0
                ).sum()
            )
            for index in self.client_indices()
        ]


# ---------------------------------------------------------------------------
# The model: one d-vector, and the penalty it is scored with
# ---------------------------------------------------------------------------


class LogisticModel(nn.Module):  # type: ignore[misc]
    """The iterate `x`, as a `d`-element `nn.Parameter`.

    `lam`, the loss and the penalty are plain attributes, not buffers: a
    buffer is in ``state_dict``, and would be uploaded and averaged every round.
    """

    def __init__(
        self,
        dim: int,
        penalty_strength: float = 0.03,
        support_tolerance: float = 1.0e-3,
        x_init: float = 0.0,
        loss: str = "logistic",
        penalty: str = "l1",
    ) -> None:
        """Place the iterate at `x_init` in every coordinate.

        Args:
            dim: Problem dimension `d`, from ``model.input_dim``.
            penalty_strength: `lam`, checked against the manifest's.
            support_tolerance: The threshold the support metrics use. Not part
                of the objective.
            x_init: Starting value in every coordinate.
            loss: The loss of the margin, checked against the manifest's.
            penalty: The penalty `lam` multiplies, checked against the
                manifest's.
        """

        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"model.loss must be one of {sorted(LOSSES)}, not {loss!r}")
        if penalty not in PENALTIES:
            raise ValueError(f"model.penalty must be one of {sorted(PENALTIES)}, not {penalty!r}")
        self.x = nn.Parameter(torch.full((dim,), float(x_init), dtype=DTYPE))
        self.penalty_strength = float(penalty_strength)
        self.support_tolerance = float(support_tolerance)
        self.loss_form = str(loss)
        self.penalty_form = str(penalty)

    def forward(self, features: Tensor) -> Tensor:
        """The margins `A_B x` for a batch of rows."""

        return features @ self.x

    def penalty(self) -> Tensor:
        """The penalty at the iterate, as a scalar tensor."""

        return penalty_value(self.x, self.penalty_strength, self.penalty_form)

    @property
    def iterate(self) -> Tensor:
        """The vector the whole example is about, detached."""

        return self.x.detach().clone()


def build_logistic_vector(config: Mapping[str, Any] | None = None) -> LogisticModel:
    """Registry builder for `model.name: logistic_vector`."""

    values = dict(config or {})
    reject_unknown_model_keys(values, _MODEL_KEYS, "logistic_vector")
    dim = values.get("input_dim")
    if dim is None:
        raise ValueError("model.input_dim is required for logistic_vector: it is the problem's d")
    return LogisticModel(
        dim=int(dim),
        penalty_strength=float(values.get("penalty_strength", 0.03)),
        support_tolerance=float(values.get("support_tolerance", 1.0e-3)),
        x_init=float(values.get("x_init", 0.0)),
        loss=str(values.get("loss", "logistic")),
        penalty=str(values.get("penalty", "l1")),
    )


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

#: The name of this problem's generator and task, as configs write it.
DATASET_NAME = "fed_logistic_l1"

#: The ``problem`` keys a generator config may state.
PROBLEM_KEYS = {
    "dim",
    "rows_per_client",
    "sparsity",
    "signal_scale",
    "penalty_strength",
    "partition_block",
    "loss",
    "penalty",
}


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """What ``fedbrew generate`` prints when this generator finishes."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


def _spec_from_config(config: Mapping[str, Any]) -> ProblemSpec:
    """Read the spec out of a generator config's ``problem`` and ``partition``."""

    problem = dict(config.get("problem", {}))
    partition = dict(config.get("partition", {}))
    if "num_clients" not in partition:
        raise ValueError(
            "fed_logistic_l1 needs partition.num_clients: it is how many ways the rows are dealt"
        )
    return ProblemSpec(
        num_clients=int(partition["num_clients"]),
        dim=int(problem.get("dim", 32)),
        rows_per_client=int(problem.get("rows_per_client", 64)),
        sparsity=int(problem.get("sparsity", 3)),
        signal_scale=float(problem.get("signal_scale", 2.0)),
        penalty_strength=float(problem.get("penalty_strength", 0.03)),
        partition_block=int(problem.get("partition_block", 32)),
        loss=str(problem.get("loss", "logistic")),
        penalty=str(problem.get("penalty", "l1")),
    )


def _spec_from_reference(reference: Mapping[str, Any]) -> ProblemSpec:
    """Rebuild the *dials* from the manifest's ``reference``.

    `x*` and `F*` are not among them: they are a solve, stored as values.
    """

    problem = reference["problem"]
    return ProblemSpec(
        num_clients=int(problem["clients"]),
        dim=int(problem["dim"]),
        rows_per_client=int(problem["rows_per_client"]),
        sparsity=int(problem.get("sparsity", 3)),
        signal_scale=float(problem.get("signal_scale", 2.0)),
        penalty_strength=float(problem["penalty_strength"]),
        partition_block=int(problem["partition_block"]),
        loss=str(problem.get("loss", "logistic")),
        penalty=str(problem.get("penalty", "l1")),
    )


def _problem_record(spec: ProblemSpec) -> dict[str, Any]:
    """The dials, as the manifest's ``reference.problem`` records them."""

    return {
        "clients": spec.num_clients,
        "dim": spec.dim,
        "rows_per_client": spec.rows_per_client,
        "sparsity": spec.sparsity,
        "signal_scale": spec.signal_scale,
        "penalty_strength": spec.penalty_strength,
        "partition_block": spec.partition_block,
        "loss": spec.loss,
        "penalty": spec.penalty,
    }


def reference_of(
    spec: ProblemSpec,
    iterations: int = 20_000,
    client_iterations: int = 4_000,
) -> dict[str, Any]:
    """Everything a run on this data is scored against.

    Written into the manifest, copied into ``run.json``, and read back by
    :class:`FedLogisticL1Task`. A problem that is not convex has no certified
    optimum, so its reference holds no ``x_star``, ``f_star`` or
    ``kkt_residual``, and its runs report no gap. The two budgets are lowered
    by ``_self_check``, which checks this block's plumbing and not the shipped
    accuracy.
    """

    features, labels = spec.design(), spec.labels()
    stacked = torch.cat(spec.client_indices())
    reference: dict[str, Any] = {
        "problem": _problem_record(spec),
        "rows": spec.rows,
        "rows_per_client": spec.rows_per_client,
        "label_base": label_base(spec.dim),
        "x_true": spec.truth().tolist(),
        "truth_support": sorted(spec.truth_support()),
        "lipschitz": spec.lipschitz(),
        "client_label_balance": spec.client_label_balance(),
    }
    if not spec.certified:
        return reference
    optimum, residual = spec.optimum(iterations=iterations)
    if residual > CERTIFICATE:
        raise ValueError(
            f"the reference solve reached a KKT residual of {residual:.3e}, above the "
            f"{CERTIFICATE:g} this generator certifies to. Every optimality gap on this data "
            "would be measured against a point that is not the minimiser."
        )
    reference.update(
        {
            "x_star": optimum.tolist(),
            # In the order the global shard stacks the rows, which is the order
            # the task's pooled objective sums in.
            "f_star": objective(
                optimum,
                features[stacked],
                labels[stacked],
                spec.penalty_strength,
                spec.loss,
                spec.penalty,
            ),
            "kkt_residual": residual,
            "optimum_support": sorted(support_of(optimum, 0.0)),
            "support_recoverable": support_of(optimum, 0.0) == spec.truth_support(),
            "client_gradient_dispersion": spec.client_gradient_dispersion(optimum),
        }
    )
    if spec.penalty == "l1":
        reference["client_support_sizes"] = spec.client_support_sizes(iterations=client_iterations)
    return reference


def generate_fed_logistic_l1_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, plus the manifest a run reads.

    ``seed`` fixes no draw -- the data is a deterministic function of the spec
    -- and is recorded because every dataset records the seed it was made at.
    ``client_splits`` describe a cut and there is nothing to cut: `F_c` is
    defined over all `m` of a client's rows, so all three splits hold them,
    declared as ``client_test_source: identical_to_train``.
    """

    del client_splits
    spec = _spec_from_config(config)
    features = spec.design()
    labels = spec.labels()
    partition = spec.client_indices()
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_of(spec)

    clients: list[dict[str, Any]] = []
    for index, rows in enumerate(partition):
        client_id = f"client_{index}"
        x = features[rows].clone()
        y = labels[rows].clone()
        save_split_client_shard(shards_dir / f"{client_id}.pt", x, y, x, y, x, y)
        entry = {
            "client_id": client_id,
            "shard": f"shards/{client_id}.pt",
            "num_examples": 3 * spec.rows_per_client,
            "num_train_examples": spec.rows_per_client,
            "num_eval_examples": spec.rows_per_client,
            "num_test_examples": spec.rows_per_client,
            "positive_label_fraction": reference["client_label_balance"][index],
        }
        if "client_support_sizes" in reference:
            entry["own_support_size"] = reference["client_support_sizes"][index]
        clients.append(entry)

    # The server's central pass evaluates F in one go, so the pooled shard is
    # every client's rows stacked: exactly the federated objective, because
    # every client holds the same m rows and so the same weight.
    stacked = torch.cat(partition)
    save_client_shard(
        shards_dir / "global_test.pt", features[stacked].clone(), labels[stacked].clone()
    )
    _write_partition_stats(output_dir, spec, clients, reference)

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
        "partition_key": "margin_block",
        "seed": seed,
        "reference": reference,
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients)
    return GenerationSummary(
        manifest_path=manifest_path,
        num_clients=len(clients),
        num_examples=spec.rows,
        num_test_examples=spec.rows,
    )


def _write_partition_stats(
    output_dir: Path,
    spec: ProblemSpec,
    clients: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> None:
    """Write the two files ``fedbrew inspect-data`` reads beside the manifest."""

    balance = list(reference["client_label_balance"])
    positives = int(round(sum(balance) * spec.rows_per_client))
    payload = {
        "dataset_name": DATASET_NAME,
        "partition_strategy": "analytic",
        "num_clients": len(clients),
        "total_examples": sum(int(client["num_examples"]) for client in clients),
        "min_examples_per_client": 3 * spec.rows_per_client,
        "max_examples_per_client": 3 * spec.rows_per_client,
        "mean_examples_per_client": 3.0 * spec.rows_per_client,
        "global_label_counts": {"-1": spec.rows - positives, "1": positives},
        "penalty_strength": spec.penalty_strength,
        "partition_block": spec.partition_block,
        "min_positive_label_fraction": min(balance),
        "max_positive_label_fraction": max(balance),
        "clients": [dict(client) for client in clients],
    }
    for key in ("client_gradient_dispersion", "kkt_residual"):
        if key in reference:
            payload[key] = reference[key]
    # allow_nan=False: every float here is measured, and a non-finite one is
    # not JSON.
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = ["client_id", "num_examples", "positive_label_fraction"]
    if "client_support_sizes" in reference:
        columns.append("own_support_size")
    with (output_dir / "client_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for client in clients:
            writer.writerow([client[column] for column in columns])


# ---------------------------------------------------------------------------
# The task adapter
# ---------------------------------------------------------------------------


class FedLogisticL1Task(TaskAdapter):
    """Bridge between the composite objective and the generic FL orchestration.

    Reads `x*` and `F*` out of the manifest as values, where the problem has
    them: there is no closed form to rebuild them from.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
        device: str = "cpu",
        **unused: Any,
    ) -> None:
        """Build the adapter, and refuse a run whose two halves disagree.

        Args:
            model_config: The resolved ``model`` block: `d`, `lam`, the loss
                and the penalty.
            dataset_metadata: The manifest, as ``manifest_dataset`` reports it.
                Carries ``reference``, written by the generator.
            device: The resolved ``runtime.device``.
            **unused: The rest of the task contract, ignored because the loader
                is built per call.

        Raises:
            ValueError: If the data carries no reference, a convex problem's
                data carries no optimum, or the model block and the data
                describe different problems.
        """

        del unused
        self.device = torch.device(device)
        self.reference = dict((dataset_metadata or {}).get("reference") or {})
        if "problem" not in self.reference:
            raise ValueError(
                "fed_logistic_l1 task needs a manifest written by its own generator: the "
                "problem lives in the manifest's `reference` and this data has none."
            )
        spec = _spec_from_reference(self.reference)
        _check_model_against_reference(model_config or {}, spec)
        missing = [key for key in ("x_star", "f_star") if key not in self.reference]
        if spec.certified and missing:
            raise ValueError(
                f"fed_logistic_l1 data for a convex problem is missing {missing}: x* is a "
                "solve done once at generation. Regenerate the data with this example's "
                "generator."
            )
        self.spec = spec
        self._optimum: Tensor | None = None
        self._optimal_objective: float | None = None
        if spec.certified:
            self._optimum = torch.tensor(self.reference["x_star"], dtype=DTYPE).to(self.device)
            self._optimal_objective = float(self.reference["f_star"])
        self._truth: Tensor | None = spec.truth().to(self.device)
        self._truth_support = spec.truth_support()
        pooled = _pooled_rows(spec, dataset_metadata or {})
        self._pooled_features = pooled[0].to(self.device)
        self._pooled_labels = pooled[1].to(self.device)
        self._support_mask = torch.tensor(
            [index in self._truth_support for index in range(spec.dim)], device=self.device
        )
        # What a client evaluation reports, and what the central pass adds to
        # it. `optimality_gap` is F over every row, the same in every client's
        # copy, so it is measured once a round by `evaluate_model`.
        self._names = (
            "loss",
            *(("distance_to_optimum",) if self._optimum is not None else ()),
            *(("distance_to_truth",) if self._truth is not None else ()),
            "support_size",
            "support_f1",
            "exact_zeros",
        )
        self._central_names = (
            "loss",
            *(("optimality_gap",) if self._optimal_objective is not None else ()),
            *self._names[1:],
        )
        # Read as `getattr(task, "_scaler", None)` by four client rules, to
        # decide whether to refuse `runtime.use_amp: true`.
        self._scaler: Any = None

    def pooled_objective(self, x: Tensor) -> float:
        """`F(x)` over every row, from the cached pooled rows."""

        spec = self.spec
        return objective(
            x,
            self._pooled_features,
            self._pooled_labels,
            spec.penalty_strength,
            spec.loss,
            spec.penalty,
        )

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> LogisticModel:
        """Build the vector model named by the `model` config block."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        factory = models.get(str(values.get("name", "logistic_vector")))
        model = factory(values)
        return model.to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Cut one client's rows into batches: a list, so it is re-iterable."""

        loader_config = _loader_config(config)
        features, labels = _rows_of(data)
        features = features.to(self.device)
        labels = labels.to(self.device)
        rows = len(labels)
        batch_size = max(1, int(loader_config.get("batch_size", rows) or rows))
        if bool(loader_config.get("shuffle", False)):
            order = _permutation(rows, loader_config.get("seed"))
            features, labels = features[order], labels[order]
        batches = [
            (features[start : start + batch_size], labels[start : start + batch_size])
            for start in range(0, rows, batch_size)
        ]
        if bool(loader_config.get("drop_last", False)) and len(batches) > 1:
            batches = [batch for batch in batches if len(batch[1]) == batch_size]
        return batches

    def train_step(
        self,
        model: LogisticModel,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Take one (sub)gradient step on the batch's composite objective.

        The penalty enters at full strength in every batch: the smooth part is
        an unbiased estimate of its full-data value and the penalty is exact.
        """

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = self.functional_loss(model, None, None, self._move_batch(batch))
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def evaluation_total(self, batch: Any) -> float | None:
        """eval_step's "total", the batch's rows, without evaluating."""

        _, labels = self._move_batch(batch)
        return float(len(labels))

    def eval_step(self, model: LogisticModel, batch: Any) -> dict[str, float]:
        """Measure the batch's objective, and the properties of the iterate."""

        model.eval()
        with torch.no_grad():
            outputs = self.functional_eval(model, None, None, self._move_batch(batch))
        return {name: float(value) for name, value in outputs.items()}

    # -- the batched executor (fedbrew.tasks.base.BatchableTask) --------------

    def loader_order(
        self, data: Any, config: Mapping[str, Any] | bool | None = None
    ) -> LoaderOrder:
        """What ``build_dataloader(data, config)`` yields, declared (``LoaderOrder``)."""

        return listed_loader_order(len(_rows_of(data)[1]), config)

    def split_rows(self, data: Any) -> tuple[Tensor, Tensor]:
        """A split's rows and labels, as ``build_dataloader`` slices them."""

        features, labels = _rows_of(data)
        return features.to(self.device), labels.to(self.device)

    def row_batches(
        self, data: Any, config: Mapping[str, Any] | bool | None = None
    ) -> list[Tensor]:
        """``build_dataloader(data, config)``'s batches as row indices, by the same code."""

        numbers = row_numbers(len(_rows_of(data)[1]))
        batches = self.build_dataloader({"x": numbers.unsqueeze(1), "y": numbers}, config)
        return [batch_row_numbers(numbered) for _, numbered in batches]

    def functional_loss(
        self,
        model: LogisticModel,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The batch's composite objective at ``params``, or at the model's own when None."""

        features, labels = batch
        iterate = model.x if params is None else params["x"]
        outputs = (
            model(features)
            if params is None
            else torch.func.functional_call(model, (dict(params), dict(buffers or {})), (features,))
        )
        loss = mean_loss(outputs, labels, model.loss_form, mask) + penalty_value(
            iterate, model.penalty_strength, model.penalty_form
        )
        return loss, {"loss": loss.detach()}

    def functional_eval(
        self,
        model: LogisticModel,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """The batch's objective, and the properties of the iterate, as tensors."""

        loss, _ = self.functional_loss(model, params, buffers, batch, mask)
        iterate = (model.x if params is None else params["x"]).detach()
        found = iterate.abs() > model.support_tolerance
        # Carried per batch because compute_metrics is handed the outputs and
        # nothing else, and all but the loss are functions of the iterate.
        measured = {"loss": loss.detach(), "total": row_count(batch[1], mask)}
        if self._optimum is not None:
            measured["distance_to_optimum"] = torch.linalg.vector_norm(iterate - self._optimum)
        if self._truth is not None:
            measured["distance_to_truth"] = torch.linalg.vector_norm(iterate - self._truth)
        measured["support_size"] = found.sum().to(DTYPE)
        measured["support_f1"] = self._support_f1(found)
        measured["exact_zeros"] = (iterate == 0.0).sum().to(DTYPE)
        return measured

    def _support_f1(self, found: Tensor) -> Tensor:
        """F1 of the support at the tolerance against the planted one; 0 if nothing hit."""

        hits = (found & self._support_mask).sum().to(DTYPE)
        precision = hits / found.sum().to(DTYPE)
        recall = hits / len(self._truth_support)
        f1 = 2.0 * precision * recall / (precision + recall)
        return torch.where(hits > 0, f1, torch.zeros_like(f1))

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Fold eval-step outputs into the numbers this task reports.

        ``loss`` is the example-weighted mean of the composite objective over
        what was evaluated -- `F(x)` on the central pass, and not the gap, since
        `F* != 0`. ``distance_to_optimum`` and ``distance_to_truth`` are
        `||x - x*||` (where the problem has a certified `x*`) and
        `||x - x_true||`, whose floor is `||x* - x_true||`. The support columns
        are measured at ``model.support_tolerance``, and ``exact_zeros`` counts
        coordinates that are bit-for-bit 0.0: `d` at initialisation and 0 from
        round 1 on every shipped arm, since none applies a proximal operator.
        """

        records = [record for record in outputs if isinstance(record, Mapping)]
        names = self._names
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

    def evaluate_model(self, model: LogisticModel, data: Any) -> dict[str, float]:
        """Measure the global model on every client's rows at once.

        Returned keys arrive as ``central_test_<name>``. ``optimality_gap`` is
        measured here and only here: `F(x)` is the same number for every
        client and batch.
        """

        outputs = [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        measured = self.compute_metrics(outputs)
        if self._optimal_objective is not None:
            measured["optimality_gap"] = (
                self.pooled_objective(model.iterate) - self._optimal_objective
            )
        return {name: measured[name] for name in self._central_names}

    # -- narrowing a split to given positions --------------------------------

    def count_examples(self, data: Any) -> int:
        """The rows `data` holds, which are what `select_examples` indexes."""

        return len(_rows_of(data)[1])

    def select_examples(self, data: Any, indices: Tensor) -> dict[str, Tensor]:
        """The rows of `data` at `indices`, in that order, as a shard's ``x`` and ``y``."""

        features, labels = _rows_of(data)
        return {"x": features[indices], "y": labels[indices]}

    # -- this task's own batch splitter -------------------------------------

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Split a batch into (features, labels), both on the device."""

        features, labels = batch
        return features.to(self.device), labels.to(self.device)


def _composite_loss(model: LogisticModel, outputs: Tensor, labels: Tensor) -> Tensor:
    """The batch's mean loss plus the penalty at the model's iterate."""

    return mean_loss(outputs, labels, model.loss_form) + model.penalty()


def _rows_of(data: Any) -> tuple[Tensor, Tensor]:
    """The feature rows and labels in one split of a shard."""

    if isinstance(data, Mapping):
        features = data.get("x", data.get("X", data.get("features")))
        labels = data.get("y", data.get("Y", data.get("targets")))
        if isinstance(features, Tensor) and isinstance(labels, Tensor):
            return features.to(DTYPE), labels.to(DTYPE)
    raise ValueError("fed_logistic_l1 data must be a shard mapping carrying 'x' and 'y' tensors")


def _pooled_rows(spec: ProblemSpec, metadata: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """Every row of the dataset, in the order the global shard stacks them.

    Read from the global shard beside the manifest when there is one -- it is
    the dealt data itself -- and rebuilt from the dials otherwise.
    """

    manifest_path = str(metadata.get("manifest_path") or "")
    global_test = str(metadata.get("global_test") or "")
    if manifest_path and global_test:
        path = Path(manifest_path).parent / global_test
        if path.is_file():
            return _rows_of(load_client_shard(path))
    stacked = torch.cat(spec.client_indices())
    return spec.design()[stacked], spec.labels()[stacked]


def _check_model_against_reference(model_config: Mapping[str, Any], spec: ProblemSpec) -> None:
    """Refuse a model block that describes a different problem than the data."""

    stated = (
        int(model_config.get("input_dim", 0)),
        float(model_config.get("penalty_strength", 0.03)),
        str(model_config.get("loss", "logistic")),
        str(model_config.get("penalty", "l1")),
    )
    data = (spec.dim, spec.penalty_strength, spec.loss, spec.penalty)
    if stated == data:
        return
    raise ValueError(
        "model config describes d={}, lam={}, loss={!r}, penalty={!r}, but the generated "
        "data is d={}, lam={}, loss={!r}, penalty={!r} (manifest reference). The rows come "
        "from the shards and the objective from the model block, so a disagreement scores "
        "the run against the wrong optimum.".format(*stated, *data)
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

TASK_NAME = "fed_logistic_l1"
MODEL_NAME = "logistic_vector"


def register() -> None:
    """Register the generator, the task and the model.

    Called once by ``fedbrew.core.extensions``, for the entry a config names
    under ``experiment.extensions`` or ``dataset.extensions``. Nothing is
    registered at import, so this module can be imported for
    :class:`ProblemSpec` alone.
    """

    from fedbrew.core import registry

    registry.generators.register(
        DATASET_NAME,
        generate_fed_logistic_l1_from_config,
        sections={"problem": PROBLEM_KEYS},
    )
    registry.tasks.register(TASK_NAME, lambda **kwargs: FedLogisticL1Task(**kwargs))
    registry.models.register(MODEL_NAME, build_logistic_vector, task=TASK_NAME)


def _self_check() -> None:
    """Assert the claims this module's docstring makes, on a small instance.

    d = 8, 4 clients of 16 rows: cheap at import, and every mechanism. If the
    label base collided with a feature column's, `x*` would be corrupted. If
    the federated decomposition were not exact, every gap would be measured
    against the wrong objective. If the partition dropped or duplicated a row,
    `F` would not be the pooled objective. If the certificate were not checked
    on the full vector, a polish off the support would look clean. If autograd
    disagreed with :func:`gradient`, the run would descend something else. And
    if the spec did not survive the manifest, the task would rebuild a
    different problem than the generator wrote.
    """

    spec = ProblemSpec(num_clients=4, dim=8, rows_per_client=16, partition_block=4)
    for check in (_check_data, _check_decomposition, _check_solve, _check_gradients):
        failure = check(spec)
        if failure is not None:
            raise AssertionError(failure)


def _check_data(spec: ProblemSpec) -> str | None:
    """The rows are finite, the labels +/-1, and the deal a partition into equal shares."""

    features, labels = spec.design(), spec.labels()
    if not bool(torch.isfinite(features).all()):
        return "the design holds a non-finite entry"
    if set(labels.tolist()) - {-1.0, 1.0}:
        return "labels must be exactly -1 or +1"
    if label_base(spec.dim) in primes(spec.dim):
        return "the label base is one the design already uses"
    partition = spec.client_indices()
    if sorted(torch.cat(partition).tolist()) != list(range(spec.rows)):
        return "the client partition is not a partition of the rows"
    if {len(index) for index in partition} != {spec.rows_per_client}:
        return "clients do not hold equal shares"
    return None


def _check_decomposition(spec: ProblemSpec) -> str | None:
    """Both aggregation weightings of the client objectives are the pooled `F`."""

    partition = spec.client_indices()
    probe = torch.linspace(-0.7, 0.9, spec.dim, dtype=DTYPE)
    per_client = spec.client_objective_at(probe)
    uniform = sum(per_client) / len(per_client)
    weighted = (
        sum(value * len(index) for value, index in zip(per_client, partition, strict=True))
        / spec.rows
    )
    pooled = spec.objective_at(probe)
    if abs(uniform - pooled) > 1e-14 or abs(weighted - pooled) > 1e-14:
        return "the client objectives do not average to the pooled objective"
    return None


def _check_solve(spec: ProblemSpec) -> str | None:
    """The reference solve is certified on the full vector, and survives the manifest."""

    features, labels = spec.design(), spec.labels()
    probe = torch.linspace(-0.7, 0.9, spec.dim, dtype=DTYPE)
    optimum, residual = solve_reference(features, labels, spec.penalty_strength, iterations=800)
    if residual > CERTIFICATE:
        return f"the reference solve is not certified: KKT residual {residual}"
    if kkt_residual(optimum + 0.1, features, labels, spec.penalty_strength) <= residual:
        return "the KKT residual does not increase away from the optimum"
    if spec.objective_at(optimum) > spec.objective_at(probe):
        return "the reference solve is not the better of two points"
    reference = reference_of(spec, iterations=400, client_iterations=200)
    if _spec_from_reference(reference) != spec:
        return "the spec does not survive the round trip through the manifest"
    stored = torch.tensor(reference["x_star"], dtype=DTYPE)
    stacked = torch.cat(spec.client_indices())
    at_stored = objective(stored, features[stacked], labels[stacked], spec.penalty_strength)
    if reference["f_star"] != at_stored:
        return "the stored f_star is not F at the stored x_star"
    return None


def _check_gradients(spec: ProblemSpec) -> str | None:
    """Autograd on the composite loss is the analytic gradient, for every loss and penalty."""

    features, labels = spec.design(), spec.labels()
    for loss in LOSSES:
        for penalty in PENALTIES:
            model = LogisticModel(
                dim=spec.dim,
                penalty_strength=spec.penalty_strength,
                x_init=0.3,
                loss=loss,
                penalty=penalty,
            )
            _composite_loss(model, model(features), labels).backward()
            measured = model.x.grad
            if measured is None:
                return "the backward pass left no gradient on the iterate"
            expected = gradient(
                model.iterate, features, labels, spec.penalty_strength, loss, penalty
            )
            if float((measured - expected).abs().max()) > 1e-15:
                return f"autograd disagrees with the analytic {loss}+{penalty}"
    return None


_self_check()
