"""Three federated problems on one construction, with heterogeneity dials that act one at a time.

The construction
----------------
``d`` coordinates, ``N`` clients (a power of two, ``2d <= N - 1``), a planted
centre ``x̂``, and one scalar profile ``φ``::

    f_i(x) = Σ_j a_ij φ(x_j − x̂_j) + (g + ζ_i)ᵀ(x − x̂) + λ‖x‖₁
    a_ij   = ā_j (1 + ε s_ij),              ā_j = κ^{j/(d−1)}
    ζ_ij   = (ζ*/√d)(ω s_ij + √(1−ω²) r_ij)

``s_·j`` and ``r_·j`` are columns ``1 + j`` and ``1 + d + j`` of the
Sylvester–Hadamard matrix of order N: ±1, summing to zero exactly, orthogonal
to each other. So ζ*² = (1/N) Σ‖ζ_i‖² exactly, and ω is exactly the Pearson
correlation across clients between a client's curvature perturbation and its
shift, in every coordinate. F is the uniform mean of the f_i.

The three members share their clients -- the same a_ij, ζ_i and x̂ -- so the
dials mean the same thing in each:

    member                     φ                          λ, g
    heterogeneous_quadratic    u²/2                        λ = 0, g = 0
    lasso                      u²/2                        λ > 0, g = −λ s*
    geman_mcclure              u²/2 − θ u²/(1 + u²)        λ = 0, g = 0

- **The heterogeneous quadratic** (smooth convex): x* = x̂, F* = 0, μ = 1,
  L = κ. FedAvg's fixed point at step α and K local steps is, per
  coordinate, ``x̂_j + Σ_i C_ij c_ij / Σ_i C_ij`` with ``C = 1 − (1 − αa)^K``
  and ``c = −ζ/a`` (Patel et al. 2024, Prop. 4).
- **The lasso** (nonsmooth convex): a lasso with a diagonal design, the
  planted version. ``s*_j = sign(x̂_j)`` on x̂'s support and ``±0.5``
  (alternating) off it; the linear term ``g = −λ s*`` keeps the minimiser at
  x̂ -- F's soft-threshold (Tibshirani 1996, §2.2) -- and so keeps the
  clients' gradient dissimilarity at x* equal to ζ_i. F* = λ‖x̂‖₁.
- **The Geman–McClure double well** (smooth nonconvex): a quadratic minus
  θ times the Geman–McClure function. Per coordinate the stationary points
  are 0 and ±u₀, ``u₀ = √(√(2θ) − 1)``; the 2^d minima x̂ + {±u₀}^d are all
  global, F* = φ(u₀) Σ_j ā_j.

The heterogeneity measures, in closed form and recorded in the manifest's
``reference``: ζ*² (first-order heterogeneity at the optimum),
``δ*_j = (1/N) Σ_i H_i(x*)_jj ζ_ij`` (the curvature-weighted gradient
dissimilarity at the optimum; ``ε ω ζ* ā_j / √d`` for the quadratic members),
``δ*ᵀH̄⁻¹δ*``, τ = 2εκ, and FedAvg's exact floor per (α, K). δ* is zero in
three controls -- iid (ζ* = ε = 0), shift-only (ε = 0) and decoupled
(ω = 0) -- and those are where FedAvg has no drift in the quadratic member.

The stochastic oracle
---------------------
Each client holds ``2d`` rows, and row r's loss gradient is
``∇f_i(x) + σ z_r`` with ``z_r ∈ {±h_m/√d}``, the rows of the order-d
Sylvester–Hadamard matrix and their negatives. So Σ_r z_r = 0 exactly, F is
the mean of the rows' losses, and a minibatch of b rows drawn **iid with
replacement** has ``E‖ξ‖² = σ²/b``, whatever x, the client or the member.
The noise is in the data, as ``σ z_r`` added to the row's linear term, so a
run samples it through its own loader.

The arms sample that oracle through ``client.sampling: with_replacement``,
fedbrew's own (``fedbrew/clients/sampling.py``): every training pass is
**one** minibatch of ``batch_size`` rows drawn uniformly with replacement from
the loader's seeded generator. So ``update_mode: single_batch`` takes K iid
minibatches in K local iterations, SCAFFOLD's ``sequential_epoch`` does too
(one pass is one batch), and minibatch SGD at batch K·b is
``local_iterations: 1`` at ``batch_size: K·b``. The task's own loader is the
other examples': every row, in batches of ``batch_size``, permuted once when
shuffled -- which is every evaluation pass, and a training pass left at
``without_replacement``.

What this file registers
------------------------
    generators  "heterogeneous_quadratic"         writes the shards and the manifest
    tasks       "heterogeneous_quadratic"         HeterogeneousQuadraticTask
    models      "heterogeneous_quadratic_vector"  the iterate, a d-vector

``check_identities`` holds every generated dataset to the identities above;
the README gives the numbers they produce at the dials' centre.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, optim

from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import save_client_shard, save_split_client_shard
from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.tasks.base import (
    LoaderOrder,
    TaskAdapter,
    listed_loader_order,
    row_count,
    row_mean,
    stacked_row_mean,
    stacked_row_weights,
)

#: Every tensor here is float64: the smooth members' floors go below 1e-30.
DTYPE = torch.float64

#: The members, by their established names.
QUADRATIC = "heterogeneous_quadratic"
LASSO = "lasso"
DOUBLE_WELL = "geman_mcclure"
MEMBERS = (QUADRATIC, LASSO, DOUBLE_WELL)

#: The α and K FedAvg's exact floor is recorded at in the manifest.
FLOOR_STEPS = (0.1, 0.01, 0.001)
FLOOR_LOCAL_STEPS = (1, 10, 100)

#: |x_j| at or below which an off-support coordinate counts as recovered as zero.
SUPPORT_TOLERANCE = 1.0e-3


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


def hadamard(order: int) -> Tensor:
    """The Sylvester–Hadamard matrix of ``order``, a power of two: H_{2n} = [[H, H], [H, −H]]."""

    if order < 1 or order & (order - 1):
        raise ValueError(f"a Sylvester–Hadamard matrix needs a power-of-two order, not {order}")
    matrix = torch.ones((1, 1), dtype=DTYPE)
    while matrix.shape[0] < order:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    return matrix


def soft_threshold(values: Tensor, level: Tensor | float) -> Tensor:
    """sign(v) max(|v| − t, 0), coordinate-wise."""

    return torch.sign(values) * torch.clamp(values.abs() - level, min=0.0)


def profile(member: str, u: Tensor, theta: float) -> Tensor:
    """φ(u): u²/2, or u²/2 − θ u²/(1 + u²) for the double well."""

    if member == DOUBLE_WELL:
        square = u * u
        return 0.5 * square - theta * square / (1.0 + square)
    return 0.5 * u * u


def profile_slope(member: str, u: Tensor, theta: float) -> Tensor:
    """φ'(u)."""

    if member == DOUBLE_WELL:
        return u * (1.0 - 2.0 * theta / (1.0 + u * u) ** 2)
    return u


def profile_curvature(member: str, u: Tensor, theta: float) -> Tensor:
    """φ''(u)."""

    if member == DOUBLE_WELL:
        square = u * u
        return 1.0 - 2.0 * theta * (1.0 - 3.0 * square) / (1.0 + square) ** 3
    return torch.ones_like(u)


def well(theta: float) -> float:
    """u₀ = √(√(2θ) − 1), the double well's minimum."""

    return math.sqrt(math.sqrt(2.0 * theta) - 1.0)


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """One member at one setting of the dials; everything else follows from these.

    ``lam`` is the lasso's λ and must be 0 for the other two members; ``theta``
    is read by the double well only.
    """

    member: str = QUADRATIC
    num_clients: int = 64
    dim: int = 16
    kappa: float = 10.0
    zeta_star: float = 1.0
    epsilon: float = 0.5
    omega: float = 1.0
    lam: float = 0.0
    theta: float = 1.0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        self._check_shape()
        self._check_values()

    def _check_shape(self) -> None:
        """The member, and the sizes the Hadamard construction needs."""

        if self.member not in MEMBERS:
            raise ValueError(f"problem.member must be one of {MEMBERS}, not {self.member!r}")
        for name, value in (("num_clients", self.num_clients), ("dim", self.dim)):
            if value < 2 or value & (value - 1):
                raise ValueError(f"problem.{name} must be a power of two, not {value}")
        if 2 * self.dim > self.num_clients - 1:
            raise ValueError(
                "the construction needs 2 dim <= num_clients - 1: every coordinate takes two "
                "Hadamard columns of its own beside the constant one"
            )

    def _check_values(self) -> None:
        """Each parameter in its range, and only the member's own set."""

        if self.kappa < 1.0:
            raise ValueError("problem.kappa is a condition number, at least 1")
        if self.zeta_star < 0.0 or self.sigma < 0.0:
            raise ValueError("problem.zeta_star and problem.sigma must be non-negative")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("problem.epsilon must be in [0, 1)")
        if not -1.0 <= self.omega <= 1.0:
            raise ValueError("problem.omega is a correlation, in [-1, 1]")
        if self.member == LASSO and self.lam <= 0.0:
            raise ValueError("the lasso member needs problem.lam > 0")
        if self.member != LASSO and self.lam != 0.0:
            raise ValueError(f"problem.lam is the lasso's; the {self.member} member takes none")
        if self.member == DOUBLE_WELL and self.theta <= 0.5:
            raise ValueError("the double well needs problem.theta > 1/2")

    # -- the clients ---------------------------------------------------------

    @property
    def rows_per_client(self) -> int:
        return 2 * self.dim

    def mean_curvature(self) -> Tensor:
        """ā_j = κ^{j/(d−1)}: from 1 to κ."""

        return self.kappa ** (torch.arange(self.dim, dtype=DTYPE) / (self.dim - 1))

    def columns(self) -> tuple[Tensor, Tensor]:
        """(s, r): Hadamard columns 1..d and d+1..2d, each (N, d)."""

        matrix = hadamard(self.num_clients)
        return matrix[:, 1 : 1 + self.dim], matrix[:, 1 + self.dim : 1 + 2 * self.dim]

    def curvature(self) -> Tensor:
        """a_ij = ā_j (1 + ε s_ij), (N, d)."""

        s, _ = self.columns()
        return self.mean_curvature() * (1.0 + self.epsilon * s)

    def shifts(self) -> Tensor:
        """ζ_ij = (ζ*/√d)(ω s_ij + √(1 − ω²) r_ij), (N, d)."""

        s, r = self.columns()
        return (self.zeta_star / math.sqrt(self.dim)) * (
            self.omega * s + math.sqrt(1.0 - self.omega**2) * r
        )

    def centre(self) -> Tensor:
        """x̂: ±1 alternating on {0, 3, 6, ...}, 0 elsewhere."""

        centre = torch.zeros(self.dim, dtype=DTYPE)
        support = torch.arange(0, self.dim, 3)
        centre[support] = (-1.0) ** torch.arange(len(support), dtype=DTYPE)
        return centre

    def support(self) -> Tensor:
        return self.centre() != 0.0

    def subgradient_at_optimum(self) -> Tensor:
        """s* ∈ ∂‖x̂‖₁: sign(x̂_j) on the support, ±0.5 alternating off it."""

        off = 0.5 * (-1.0) ** torch.arange(self.dim, dtype=DTYPE)
        return torch.where(self.support(), torch.sign(self.centre()), off)

    def linear(self) -> Tensor:
        """g: −λ s* for the lasso, 0 otherwise."""

        if self.member == LASSO:
            return -self.lam * self.subgradient_at_optimum()
        return torch.zeros(self.dim, dtype=DTYPE)

    def noise(self) -> Tensor:
        """z_r, (2d, d): the order-d Hadamard rows over √d, then their negatives."""

        rows = hadamard(self.dim) / math.sqrt(self.dim)
        return torch.cat([rows, -rows])

    def client_rows(self) -> Tensor:
        """Every client's rows, (N, 2d, 2d): [a_i | g + ζ_i + σ z_r] per row r."""

        a = self.curvature()
        linear = self.linear() + self.shifts()
        rows = linear[:, None, :] + self.sigma * self.noise()[None, :, :]
        return torch.cat([a[:, None, :].expand(-1, self.rows_per_client, -1), rows], dim=2)

    # -- the federated objective --------------------------------------------

    def objective(self, x: Tensor) -> Tensor:
        """F(x), the mean of the clients' objectives, from the clients' stored terms."""

        return objective_value(
            self.member,
            x,
            self.centre().to(x.device),
            self.curvature().mean(dim=0).to(x.device),
            (self.linear() + self.shifts().mean(dim=0)).to(x.device),
            self.lam,
            self.theta,
        )

    def optimum(self) -> Tensor:
        """x*: x̂, or for the double well its minimum x̂ + u₀ in the start's well."""

        if self.member == DOUBLE_WELL:
            return self.centre() + well(self.theta)
        return self.centre()

    def optimal_value(self) -> float:
        """F*: 0, λ‖x̂‖₁, or φ(u₀) Σ ā_j, in closed form."""

        if self.member == LASSO:
            return self.lam * float(self.centre().abs().sum())
        if self.member == DOUBLE_WELL:
            u0 = torch.tensor(well(self.theta), dtype=DTYPE)
            return float(profile(DOUBLE_WELL, u0, self.theta) * self.mean_curvature().sum())
        return 0.0

    def start(self) -> Tensor:
        """x₀: x̂ + 1/√ā for the convex members, x̂ + u₀/4 for the double well."""

        if self.member == DOUBLE_WELL:
            return self.centre() + well(self.theta) / 4.0
        return self.centre() + 1.0 / self.mean_curvature().sqrt()

    def reference_point_curvature(self) -> float:
        """φ''(u) at x*'s offset: 1, or φ''(u₀) for the double well."""

        if self.member == DOUBLE_WELL:
            u0 = torch.tensor(well(self.theta), dtype=DTYPE)
            return float(profile_curvature(DOUBLE_WELL, u0, self.theta))
        return 1.0

    # -- heterogeneity at the optimum ----------------------------------------

    def zeta_star_squared(self) -> float:
        """(1/N) Σ_i ‖ζ_i‖²: ζ*² up to rounding."""

        return float((self.shifts() ** 2).sum(dim=1).mean())

    def delta_star(self) -> Tensor:
        """δ*_j = (1/N) Σ_i H_i(x*)_jj ζ_ij."""

        return self.reference_point_curvature() * (self.curvature() * self.shifts()).mean(dim=0)

    def delta_hbar_delta(self) -> float:
        """δ*ᵀ H̄(x*)⁻¹ δ*, with H̄(x*) = φ''(u*) diag(ā)."""

        hbar = self.reference_point_curvature() * self.curvature().mean(dim=0)
        return float((self.delta_star() ** 2 / hbar).sum())

    def correlation(self) -> list[float | None]:
        """Per coordinate, the Pearson correlation across clients of s_·j and ζ_·j: ω."""

        s, _ = self.columns()
        shifts = self.shifts()
        out: list[float | None] = []
        for j in range(self.dim):
            spread = float(shifts[:, j].std(unbiased=False))
            if spread == 0.0:
                out.append(None)
                continue
            covariance = float((s[:, j] * (shifts[:, j] - shifts[:, j].mean())).mean())
            out.append(covariance / (float(s[:, j].std(unbiased=False)) * spread))
        return out

    def client_optima(self) -> Tensor:
        """Each client's minimiser, (N, d), in closed form or by a certified scalar root."""

        a, linear, centre = self.curvature(), self.linear() + self.shifts(), self.centre()
        if self.member == QUADRATIC:
            return centre - linear / a
        if self.member == LASSO:
            return soft_threshold(centre - linear / a, self.lam / a)
        return centre + _double_well_client_minima(a, linear, self.theta)

    def support_disagreement(self) -> int:
        """Client–coordinate pairs whose own lasso minimiser has another support than x̂."""

        if self.member != LASSO:
            return 0
        own = self.client_optima() != 0.0
        return int((own != self.support()[None, :]).sum())

    def both_wells_kept(self) -> bool:
        """Whether every client keeps both wells: max |ζ_ij|/a_ij < max |φ'| on [0, u₀]."""

        inner = torch.linspace(0.0, well(self.theta), 200001, dtype=DTYPE)
        ceiling = float(profile_slope(DOUBLE_WELL, inner, self.theta).abs().max())
        return float((self.shifts().abs() / self.curvature()).max()) < ceiling

    # -- FedAvg's exact floor ----------------------------------------------------

    def fedavg_floor(self, alpha: float, local_steps: int) -> float:
        """F − F* where FedAvg with exact gradients settles, at p = 1.

        The quadratic: its fixed point in closed form. The lasso: the long-run
        mean gap of its exact round map iterated from x̂ (off the support there
        is no fixed point; subgradient steps chatter around 0). The double
        well: its fixed point near x̂ + u₀, a bracketed scalar root per
        coordinate.
        """

        a = self.curvature()
        linear = self.linear() + self.shifts()
        if self.member == QUADRATIC:
            weights = 1.0 - (1.0 - alpha * a) ** local_steps
            fixed = self.centre() + (weights * (-linear / a)).sum(0) / weights.sum(0)
            return float(self.objective(fixed)) - self.optimal_value()
        if self.member == LASSO:
            return _lasso_floor(self, a, linear, alpha, local_steps)
        return _double_well_floor(self, a, linear, alpha, local_steps)


def objective_value(
    member: str,
    x: Tensor,
    centre: Tensor,
    curvature: Tensor,
    linear: Tensor,
    lam: float,
    theta: float,
) -> Tensor:
    """Σ_j A_j φ(x_j − x̂_j) + bᵀ(x − x̂) + λ‖x‖₁ for one (A, b): a client's, or their mean."""

    u = x - centre
    value = (curvature * profile(member, u, theta)).sum(-1) + (linear * u).sum(-1)
    if member == LASSO:
        value = value + lam * x.abs().sum(-1)
    return value


def _double_well_client_minima(a: Tensor, linear: Tensor, theta: float) -> Tensor:
    """Per client and coordinate, the global minimiser of a φ(u) + b u.

    φ' is odd, falls from 0 to its minimum at u_m on (0, u₀) and rises from
    there, so ``a φ'(u) + b`` is increasing on [u_m, ∞) and on (−∞, −u_m]: each
    holds at most one root, the minimum of that well, which bisection finds for
    every pair at once. The lower of the two is the client's minimiser.
    """

    grid = torch.linspace(0.0, well(theta), 200001, dtype=DTYPE)
    turn = float(grid[profile_slope(DOUBLE_WELL, grid, theta).argmin()])

    def root(low: float, high: float) -> tuple[Tensor, Tensor]:
        lo = torch.full_like(a, low)
        hi = torch.full_like(a, high)
        exists = (a * profile_slope(DOUBLE_WELL, lo, theta) + linear < 0) & (
            a * profile_slope(DOUBLE_WELL, hi, theta) + linear > 0
        )
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = a * profile_slope(DOUBLE_WELL, mid, theta) + linear < 0
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        return 0.5 * (lo + hi), exists

    right, has_right = root(turn, 50.0)
    left, has_left = root(-50.0, -turn)
    if not bool((has_right | has_left).all()):
        raise AssertionError("a client objective with no minimum in either well")

    def value(u: Tensor, exists: Tensor) -> Tensor:
        return torch.where(exists, a * profile(DOUBLE_WELL, u, theta) + linear * u, torch.inf)

    return torch.where(value(right, has_right) <= value(left, has_left), right, left)


def _lasso_floor(
    spec: ProblemSpec, a: Tensor, linear: Tensor, alpha: float, local_steps: int
) -> float:
    """FedAvg's exact round map iterated from x̂: the mean gap of its last 2,000 rounds."""

    centre = spec.centre()
    tail, cap = 2000, 40000
    rounds = int(min(cap, 20.0 / (alpha * local_steps * float(a.min())))) + tail
    x = centre.clone()
    abar, glinear = a.mean(0), linear.mean(0)
    gaps = []
    # In place, in the order the step is written: (a (u − x̂) + g + λ sign u), times α, from u.
    u, step, sign = torch.empty_like(a), torch.empty_like(a), torch.empty_like(a)
    for index in range(rounds):
        u.copy_(x.expand_as(a))
        for _ in range(local_steps):
            torch.sub(u, centre, out=step)
            step.mul_(a).add_(linear)
            torch.sign(u, out=sign)
            step.add_(sign.mul_(spec.lam)).mul_(alpha)
            u.sub_(step)
        x = u.mean(0)
        if index >= rounds - tail:
            value = (0.5 * abar * (x - centre) ** 2 + glinear * (x - centre)).sum()
            gaps.append(float(value + spec.lam * x.abs().sum()) - spec.optimal_value())
    return math.fsum(gaps) / len(gaps)


def _double_well_floor(
    spec: ProblemSpec, a: Tensor, linear: Tensor, alpha: float, local_steps: int
) -> float:
    """FedAvg's double-well fixed point near x̂ + u₀, every coordinate bisected at once; its gap."""

    theta, u0 = spec.theta, well(spec.theta)

    def excess(u: Tensor) -> Tensor:
        # The round's displacement, accumulated as such: u − α(...) − u would
        # lose it to cancellation where it is smallest, at the fixed point.
        moved = torch.zeros(a.shape, dtype=DTYPE)
        for _ in range(local_steps):
            local = u + moved
            moved -= alpha * (a * local * (1.0 - 2.0 * theta / (1.0 + local * local) ** 2) + linear)
        return moved.mean(0)

    lo = torch.full((spec.dim,), u0 - 0.2, dtype=DTYPE)
    hi = torch.full((spec.dim,), u0 + 0.2, dtype=DTYPE)
    flo = excess(lo)
    if not bool(torch.all(flo * excess(hi) < 0)):
        raise AssertionError("FedAvg's double-well fixed point is not bracketed near u0")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = excess(mid)
        left = flo * fmid < 0
        hi = torch.where(left, mid, hi)
        lo, flo = torch.where(left, lo, mid), torch.where(left, flo, fmid)
    x = spec.centre() + 0.5 * (lo + hi)
    return float(spec.objective(x)) - spec.optimal_value()


# ---------------------------------------------------------------------------
# The construction's identities, checked on every generated dataset
# ---------------------------------------------------------------------------


def check_identities(spec: ProblemSpec) -> dict[str, float]:
    """Assert the construction's identities on ``spec``; return the residuals measured.

    Each is load-bearing: without them ζ*, ω and δ* do not mean what the
    dials say, and x*, F* are not the optimum every gap is measured from.
    """

    residuals: dict[str, float] = {}
    shifts, a, abar = spec.shifts(), spec.curvature(), spec.mean_curvature()

    def hold(name: str, value: float, bound: float) -> None:
        residuals[name] = value
        if not value <= bound:
            raise AssertionError(f"{name} = {value:.3e}, above {bound:.1e}")

    hold("sum_i zeta_i", float(shifts.sum(dim=0).abs().max()), 1e-13)
    hold(
        "sum_i a_ij - N abar_j", float((a.sum(dim=0) - spec.num_clients * abar).abs().max()), 1e-11
    )
    hold(
        "zeta*^2 relative",
        abs(spec.zeta_star_squared() - spec.zeta_star**2) / max(spec.zeta_star**2, 1.0),
        1e-13,
    )
    if spec.zeta_star > 0.0:
        hold(
            "Pearson(s, zeta) - omega",
            max(abs(value - spec.omega) for value in spec.correlation() if value is not None),
            1e-12,
        )
    noise = spec.noise()
    hold("sum_r z_r", float(noise.sum(dim=0).abs().max()), 1e-15)
    hold("||z_r||^2 - 1", float(((noise**2).sum(dim=1) - 1.0).abs().max()), 1e-14)
    hold(
        "Cov(z) - I/d",
        float(
            (noise.T @ noise / len(noise) - torch.eye(spec.dim, dtype=DTYPE) / spec.dim).abs().max()
        ),
        1e-15,
    )
    rows = spec.client_rows()
    stored_mean = rows[:, :, spec.dim :].mean(dim=1) - (spec.linear() + shifts)
    hold("mean of a client's rows - its linear term", float(stored_mean.abs().max()), 1e-13)

    closed = (spec.epsilon * spec.omega * spec.zeta_star) ** 2 * float(abar.mean())
    closed *= spec.reference_point_curvature()
    hold(
        "delta*^T Hbar^-1 delta* - closed form",
        abs(spec.delta_hbar_delta() - closed) / max(closed, 1e-300)
        if closed
        else spec.delta_hbar_delta(),
        1e-10,
    )
    if spec.epsilon == 0.0 or spec.omega == 0.0 or spec.zeta_star == 0.0:
        hold("delta* in a control", float(spec.delta_star().abs().max()), 1e-15)

    # F* and x*: stationarity (the lasso's subdifferential condition).
    optimum = spec.optimum().clone().requires_grad_(True)
    smooth = objective_value(
        spec.member if spec.member != LASSO else QUADRATIC,
        optimum,
        spec.centre(),
        a.mean(dim=0),
        spec.linear() + shifts.mean(dim=0),
        0.0,
        spec.theta,
    )
    (gradient,) = torch.autograd.grad(smooth, optimum)
    if spec.member == LASSO:
        on = spec.support()
        kkt = torch.where(
            on,
            (gradient + spec.lam * torch.sign(spec.optimum())).abs(),
            torch.clamp(gradient.abs() - spec.lam, min=0.0),
        )
        hold("lasso KKT residual at x-hat", float(kkt.max()), 1e-12)
    else:
        hold("||grad F(x*)||", float(gradient.abs().max()), 1e-12)
    hold(
        "F(x*) - F* closed form",
        abs(float(spec.objective(spec.optimum())) - spec.optimal_value()),
        1e-12,
    )

    if spec.member == QUADRATIC:
        # The fixed point in closed form against the composed round map.
        alpha, steps = 0.01, 10
        slope = torch.ones_like(a)
        offset = torch.zeros_like(a)
        linear = shifts + spec.linear()
        for _ in range(steps):
            slope, offset = (1.0 - alpha * a) * slope, (1.0 - alpha * a) * offset - alpha * linear
        composed = spec.centre() + offset.mean(dim=0) / (1.0 - slope.mean(dim=0))
        weights = 1.0 - (1.0 - alpha * a) ** steps
        closed_point = spec.centre() + (weights * (-linear / a)).sum(0) / weights.sum(0)
        hold(
            "FedAvg fixed point: closed form - composed map",
            float((composed - closed_point).abs().max()),
            1e-13,
        )
    return residuals


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

#: The generator's name, and the task's.
DATASET_NAME = "heterogeneous_quadratic"
TASK_NAME = "heterogeneous_quadratic"
MODEL_NAME = "heterogeneous_quadratic_vector"

#: The problem section's keys.
PROBLEM_KEYS = frozenset(
    {"member", "dim", "kappa", "zeta_star", "epsilon", "omega", "lam", "theta", "sigma"}
)


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """What ``fedbrew generate`` prints when this generator finishes."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


def _spec_from_config(config: Mapping[str, Any]) -> ProblemSpec:
    problem = dict(config.get("problem", {}))
    partition = dict(config.get("partition", {}))
    if "num_clients" not in partition:
        raise ValueError("heterogeneous_quadratic needs partition.num_clients: it is N")
    member = str(problem.get("member", QUADRATIC))
    return ProblemSpec(
        member=member,
        num_clients=int(partition["num_clients"]),
        dim=int(problem.get("dim", 16)),
        kappa=float(problem.get("kappa", 10.0)),
        zeta_star=float(problem.get("zeta_star", 1.0)),
        epsilon=float(problem.get("epsilon", 0.5)),
        omega=float(problem.get("omega", 1.0)),
        lam=float(problem.get("lam", 1.0 if member == LASSO else 0.0)),
        theta=float(problem.get("theta", 1.0)),
        sigma=float(problem.get("sigma", 0.0)),
    )


def _spec_from_reference(reference: Mapping[str, Any]) -> ProblemSpec:
    problem = reference["problem"]
    return ProblemSpec(
        member=str(problem["member"]),
        num_clients=int(problem["num_clients"]),
        dim=int(problem["dim"]),
        kappa=float(problem["kappa"]),
        zeta_star=float(problem["zeta_star"]),
        epsilon=float(problem["epsilon"]),
        omega=float(problem["omega"]),
        lam=float(problem["lam"]),
        theta=float(problem["theta"]),
        sigma=float(problem["sigma"]),
    )


def reference_of(
    spec: ProblemSpec, identities: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Everything a run on this data is scored against, in closed form."""

    floors = [
        {"alpha": alpha, "local_steps": steps, "gap": spec.fedavg_floor(alpha, steps)}
        for alpha in FLOOR_STEPS
        for steps in FLOOR_LOCAL_STEPS
    ]
    abar = spec.mean_curvature()
    return {
        "problem": {
            "member": spec.member,
            "num_clients": spec.num_clients,
            "dim": spec.dim,
            "kappa": spec.kappa,
            "zeta_star": spec.zeta_star,
            "epsilon": spec.epsilon,
            "omega": spec.omega,
            "lam": spec.lam,
            "theta": spec.theta,
            "sigma": spec.sigma,
        },
        "rows_per_client": spec.rows_per_client,
        "x_hat": spec.centre().tolist(),
        "x_star": spec.optimum().tolist(),
        "f_star": spec.optimal_value(),
        "x_start": spec.start().tolist(),
        "start_gap": float(spec.objective(spec.start())) - spec.optimal_value(),
        "client_optima": spec.client_optima().tolist(),
        "mean_curvature": abar.tolist(),
        # The smooth part's local strong convexity at x*: 1, or φ''(u₀).
        "mu": spec.reference_point_curvature(),
        "smoothness": spec.kappa,
        "client_smoothness": (1.0 + spec.epsilon) * spec.kappa,
        "zeta_star_squared": spec.zeta_star_squared(),
        "omega_measured": spec.correlation(),
        "delta_star": spec.delta_star().tolist(),
        "delta_hbar_inverse_delta": spec.delta_hbar_delta(),
        "hessian_dissimilarity": spec.epsilon * spec.kappa,
        "second_order_heterogeneity": 2.0 * spec.epsilon * spec.kappa,
        "support": [int(j) for j in torch.nonzero(spec.support()).flatten()],
        "support_disagreement": spec.support_disagreement(),
        "both_wells_kept": spec.both_wells_kept() if spec.member == DOUBLE_WELL else None,
        "well": well(spec.theta) if spec.member == DOUBLE_WELL else None,
        "noise_per_row": spec.sigma**2,
        "fedavg_exact_floor": floors,
        "identities": dict(identities or {}),
    }


def generate_heterogeneous_quadratic_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, the pooled global shard, and the manifest.

    ``seed`` draws nothing: the rows are a deterministic function of the
    dials. ``client_splits`` cuts nothing: f_i is the mean over all 2d rows,
    so every split holds them all (``client_test_source: identical_to_train``).
    """

    del client_splits
    spec = _spec_from_config(config)
    identities = check_identities(spec)
    rows = spec.client_rows()
    output_dir = Path(output_dir)
    shards = output_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    targets = torch.zeros(spec.rows_per_client, dtype=DTYPE)
    clients: list[dict[str, Any]] = []
    for index in range(spec.num_clients):
        client_id = f"client_{index}"
        x = rows[index].clone()
        save_split_client_shard(shards / f"{client_id}.pt", x, targets, x, targets, x, targets)
        clients.append(
            {
                "client_id": client_id,
                "shard": f"shards/{client_id}.pt",
                "num_examples": 3 * spec.rows_per_client,
                "num_train_examples": spec.rows_per_client,
                "num_eval_examples": spec.rows_per_client,
                "num_test_examples": spec.rows_per_client,
            }
        )
    pooled = rows.reshape(-1, rows.shape[-1]).clone()
    save_client_shard(shards / "global_test.pt", pooled, torch.zeros(len(pooled), dtype=DTYPE))
    reference = reference_of(spec, identities)
    (output_dir / "partition_stats.json").write_text(
        json.dumps(
            {
                "dataset_name": DATASET_NAME,
                "partition_strategy": "analytic",
                "num_clients": spec.num_clients,
                "total_examples": 3 * spec.rows_per_client * spec.num_clients,
                "clients": clients,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_name": DATASET_NAME,
        "format": "torch_shards",
        "client_shard_format": "split_v2",
        "client_test_source": IDENTICAL_TO_TRAIN,
        "num_clients": spec.num_clients,
        "input_dim": spec.dim,
        "clients_file": "clients.jsonl",
        "shards_dir": "shards",
        "global_test": "shards/global_test.pt",
        "partition_stats_file": "partition_stats.json",
        "partition_strategy": "analytic",
        "partition_key": "hadamard_columns",
        "seed": seed,
        "reference": reference,
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients)
    return GenerationSummary(
        manifest_path=manifest_path,
        num_clients=spec.num_clients,
        num_examples=spec.num_clients * spec.rows_per_client,
        num_test_examples=len(pooled),
    )


# ---------------------------------------------------------------------------
# The model and the loaders
# ---------------------------------------------------------------------------


class IterateModel(nn.Module):  # type: ignore[misc]
    """The iterate x, a d-element float64 parameter; the task places it at the start."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.x = nn.Parameter(torch.zeros(dim, dtype=DTYPE))

    def forward(self, features: Tensor) -> Tensor:
        """The iterate, once per row: the task's loss reads x itself."""

        return self.x.expand(len(features), -1)


def build_iterate(config: Mapping[str, Any] | None = None) -> IterateModel:
    """Registry builder for ``model.name: heterogeneous_quadratic_vector``."""

    values = dict(config or {})
    reject_unknown_model_keys(values, (), MODEL_NAME)
    dim = values.get("input_dim")
    if dim is None:
        raise ValueError(f"model.input_dim is required for {MODEL_NAME}: it is the problem's d")
    return IterateModel(int(dim))


class RowBatches:
    """What a loader over ``rows`` rows yields, as index tensors; re-iterable.

    Every row, in batches of ``batch_size``, the last dropped when short under
    ``drop_last`` if there is more than one; shuffled, in one permutation drawn
    when the loader is built, from ``seed`` (``listed_loader_order``).
    """

    def __init__(
        self, rows: int, batch_size: int, shuffle: bool, seed: int | None, drop_last: bool = False
    ) -> None:
        self.rows, self.batch_size, self.drop_last = rows, max(1, batch_size), drop_last
        self.order: Tensor | None = None
        if shuffle:
            generator = None
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(seed))
            self.order = torch.randperm(rows, generator=generator)

    def __iter__(self) -> Iterator[Tensor]:
        for index in range(len(self)):
            first = index * self.batch_size
            batch = torch.arange(first, min(first + self.batch_size, self.rows))
            yield batch if self.order is None else self.order[batch]

    def __len__(self) -> int:
        count = math.ceil(self.rows / self.batch_size)
        return self.rows // self.batch_size if self.drop_last and count > 1 else count


class _RowLoader:
    """A loader: the batches of ``RowBatches`` as (features, targets) on the device."""

    def __init__(self, features: Tensor, targets: Tensor, batches: RowBatches) -> None:
        self.features, self.targets, self.batches = features, targets, batches

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for index in self.batches:
            index = index.to(self.features.device)
            yield self.features[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.batches)


def _rows_of(data: Any) -> tuple[Tensor, Tensor]:
    if isinstance(data, Mapping):
        features, targets = data.get("x"), data.get("y")
        if isinstance(features, Tensor) and isinstance(targets, Tensor):
            return features.to(DTYPE), targets.to(DTYPE)
    raise ValueError("heterogeneous_quadratic data must be a shard mapping with 'x' and 'y'")


def _loader_values(config: Mapping[str, Any] | bool | None) -> dict[str, Any]:
    if isinstance(config, bool):
        return {"shuffle": config}
    return dict(config or {})


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


class HeterogeneousQuadraticTask(TaskAdapter):
    """The three members' task: the rows' loss, and the gap to the closed-form optimum."""

    #: What compute_metrics reports, and which side of each is better (TaskAdapter.METRICS).
    METRICS = {
        "loss": "min",
        "optimality_gap": "min",
        "distance_to_optimum": "min",
        "off_support_small": "none",
    }

    #: What each metric measures (TaskAdapter.METRIC_GLOSSES).
    METRIC_GLOSSES = {
        "loss": "client objective Σ_j a_ij φ(x_j − x̂_j) + (g + ζ_i)ᵀ(x − x̂) + λ‖x‖₁",
        "optimality_gap": "optimality gap F(x) − F* against the closed-form optimum",
        "distance_to_optimum": "distance ‖x − x*‖₂ to the optimum, or the nearest minimum",
        "off_support_small": "fraction of the coordinates off x̂'s support with |x_j| ≤ 1e-3",
    }

    #: What its grad_norm_sq measures (TaskAdapter.GRAD_NORM_GLOSS).
    GRAD_NORM_GLOSS = (
        "squared norm of the gradient of F(x) = mean_i f_i(x), the noise rows averaging "
        "out, at the global model; for the lasso member of F's minimum-norm "
        "subgradient, whose coordinates at x_j = 0 are the smooth gradient's "
        "soft-thresholded at λ"
    )

    #: One backward through the per-client losses' sum, as the other examples.
    batched_gradient = "summed"

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
        device: str = "cpu",
        **unused: Any,
    ) -> None:
        del unused, model_config
        self.device = torch.device(device)
        reference = dict((dataset_metadata or {}).get("reference") or {})
        if "problem" not in reference:
            raise ValueError(
                "heterogeneous_quadratic needs a manifest written by its own generator: "
                "the problem and its optimum are in the manifest's `reference`"
            )
        self.spec = _spec_from_reference(reference)
        spec = self.spec
        self.member, self.lam, self.theta = spec.member, spec.lam, spec.theta
        self._centre = spec.centre().to(self.device)
        self._optimum = spec.optimum().to(self.device)
        self._curvature = spec.curvature().mean(dim=0).to(self.device)
        self._linear = (spec.linear() + spec.shifts().mean(dim=0)).to(self.device)
        self._off_support = (~spec.support()).to(self.device)
        self._well = well(spec.theta) if spec.member == DOUBLE_WELL else None
        self._f_star = spec.optimal_value()
        self._scaler: Any = None

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> IterateModel:
        """The iterate at the member's start, x₀."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        model = models.get(str(values.get("name", MODEL_NAME)))(values)
        with torch.no_grad():
            model.x.copy_(self.spec.start())
        return model.to(self.device)

    def build_dataloader(self, data: Any, config: Mapping[str, Any] | bool | None = None) -> Any:
        """Every row, in batches, permuted once when shuffled (``RowBatches``)."""

        features, targets = _rows_of(data)
        return _RowLoader(
            features.to(self.device), targets.to(self.device), self._batches(len(targets), config)
        )

    def row_batches(self, data: Any, config: Mapping[str, Any] | bool | None = None) -> RowBatches:
        """``build_dataloader``'s batches as row indices, from the same draws."""

        return self._batches(len(_rows_of(data)[1]), config)

    def _batches(self, rows: int, config: Mapping[str, Any] | bool | None) -> RowBatches:
        values = _loader_values(config)
        seed = values.get("seed")
        return RowBatches(
            rows,
            int(values.get("batch_size", rows) or rows),
            bool(values.get("shuffle", False)),
            None if seed is None else int(seed),
            bool(values.get("drop_last", False)),
        )

    def loader_order(
        self, data: Any, config: Mapping[str, Any] | bool | None = None
    ) -> LoaderOrder:
        """What ``build_dataloader(data, config)`` yields, declared (``LoaderOrder``)."""

        return listed_loader_order(len(_rows_of(data)[1]), _loader_values(config))

    def train_step(
        self, model: IterateModel, batch: Any, optimizer: optim.Optimizer | None = None
    ) -> dict[str, float]:
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = self.functional_loss(model, None, None, self._move_batch(batch))
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def evaluation_total(self, batch: Any) -> float | None:
        return float(len(self._move_batch(batch)[1]))

    def eval_step(self, model: IterateModel, batch: Any) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            outputs = self.functional_eval(model, None, None, self._move_batch(batch))
        return {name: float(value) for name, value in outputs.items()}

    def objective_loss(self, model: Any, batch: Any) -> tuple[Tensor, float]:
        """The batch's objective, as ``train_step`` takes it, and its rows (TaskAdapter)."""

        loss, _ = self.functional_loss(model, None, None, self._move_batch(batch))
        return loss, float(self.evaluation_total(batch) or 0.0)

    def objective_l1(self, model: Any) -> dict[str, float]:
        """``λ ||x||_1`` on ``x`` for the lasso member; nothing for the smooth ones."""

        del model
        return {"x": self.lam} if self.member == LASSO else {}

    # -- the batched executor ------------------------------------------------

    def split_rows(self, data: Any) -> tuple[Tensor, Tensor]:
        features, targets = _rows_of(data)
        return features.to(self.device), targets.to(self.device)

    def functional_loss(
        self,
        model: IterateModel,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The rows' mean loss at ``params`` (the model's own when None), λ‖x‖₁ once."""

        del buffers
        features = batch[0]
        x = model.x if params is None else params["x"]
        dim = x.shape[-1]
        u = x - self._centre
        per_row = (features[:, :dim] * profile(self.member, u, self.theta)).sum(-1) + (
            features[:, dim:] * u
        ).sum(-1)
        loss = row_mean(per_row, mask)
        if self.member == LASSO:
            loss = loss + self.lam * x.abs().sum()
        return loss, {"loss": loss.detach()}

    def closed_form_gradient(
        self,
        model: Any,
        params: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
        outputs: bool = True,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """``functional_loss``'s gradient for a stack of clients, in closed form (BatchableTask).

        A row's loss is `sum_j a_j phi(u_j) + g.u` with `u = x - x_hat`, so its
        gradient is `a phi'(u) + g`, and the lasso's `lam ||x||_1` adds
        `lam sign(x)`, 0 at exactly 0, as autograd takes it.
        """

        del model, buffers
        features = batch[0]
        x = params["x"]
        dim = x.shape[-1]
        u = x - self._centre
        curvature, linear = features[..., :dim], features[..., dim:]
        # The rows' weights read only the stack's shape and dtype.
        weights = stacked_row_weights(features, mask).unsqueeze(2)
        gradient = (weights * curvature).sum(1) * profile_slope(self.member, u, self.theta) + (
            weights * linear
        ).sum(1)
        if self.member == LASSO:
            gradient = gradient + self.lam * torch.sign(x)
        if not outputs:
            return {"x": gradient}, {}
        per_row = (curvature * profile(self.member, u, self.theta).unsqueeze(1)).sum(-1) + (
            linear * u.unsqueeze(1)
        ).sum(-1)
        loss = stacked_row_mean(per_row, mask)
        if self.member == LASSO:
            loss = loss + self.lam * x.abs().sum(-1)
        return {"x": gradient}, {"loss": loss}

    def functional_eval(
        self,
        model: IterateModel,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        loss, _ = self.functional_loss(model, params, buffers, batch, mask)
        x = (model.x if params is None else params["x"]).detach()
        return {
            "loss": loss.detach(),
            "total": row_count(batch[1], mask),
            "optimality_gap": self._gap(x),
            "distance_to_optimum": self._distance(x),
            "off_support_small": (
                ((x.abs() <= SUPPORT_TOLERANCE) & self._off_support).sum().to(DTYPE)
                / self._off_support.sum().to(DTYPE)
            ),
        }

    def _gap(self, x: Tensor) -> Tensor:
        value = objective_value(
            self.member, x, self._centre, self._curvature, self._linear, self.lam, self.theta
        )
        return value - self._f_star

    def _distance(self, x: Tensor) -> Tensor:
        if self._well is None:
            return torch.linalg.vector_norm(x - self._optimum)
        u = x - self._centre
        nearest = torch.minimum((u - self._well).abs(), (u + self._well).abs())
        return torch.linalg.vector_norm(nearest)

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Every metric pooled over the batches by their rows; all but loss are the iterate's."""

        names = tuple(self.METRICS)
        records = [record for record in outputs if isinstance(record, Mapping)]
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

    def evaluate_model(self, model: IterateModel, data: Any) -> dict[str, float]:
        """Every client's rows at once: loss is F(x), since each client's noise sums to zero."""

        return self.compute_metrics(
            [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        )

    def count_examples(self, data: Any) -> int:
        return len(_rows_of(data)[1])

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        features, targets = batch
        return features.to(self.device, DTYPE), targets.to(self.device, DTYPE)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the generator, the task and the model; nothing at import."""

    from fedbrew.core import registry

    registry.generators.register(
        DATASET_NAME,
        generate_heterogeneous_quadratic_from_config,
        sections={"problem": set(PROBLEM_KEYS)},
    )
    registry.tasks.register(
        TASK_NAME,
        lambda **kwargs: HeterogeneousQuadraticTask(**kwargs),
        metrics=HeterogeneousQuadraticTask.METRICS,
        glosses=HeterogeneousQuadraticTask.METRIC_GLOSSES,
        grad_norm=HeterogeneousQuadraticTask.GRAD_NORM_GLOSS,
    )
    registry.models.register(MODEL_NAME, build_iterate, task=TASK_NAME, shape_keys=("input_dim",))


def _self_check() -> None:
    """The quadratic member's identities at the design's centre, cheap enough for import."""

    spec = ProblemSpec()
    check_identities(spec)
    for member in (LASSO, DOUBLE_WELL):
        candidate = ProblemSpec(
            member=member,
            lam=1.0 if member == LASSO else 0.0,
            zeta_star=0.25 if member == DOUBLE_WELL else 1.0,
        )
        if _spec_from_reference({"problem": _problem_of(candidate)}) != candidate:
            raise AssertionError("the spec does not survive the round trip through the manifest")


def _problem_of(spec: ProblemSpec) -> dict[str, Any]:
    return {
        "member": spec.member,
        "num_clients": spec.num_clients,
        "dim": spec.dim,
        "kappa": spec.kappa,
        "zeta_star": spec.zeta_star,
        "epsilon": spec.epsilon,
        "omega": spec.omega,
        "lam": spec.lam,
        "theta": spec.theta,
        "sigma": spec.sigma,
    }


_self_check()
