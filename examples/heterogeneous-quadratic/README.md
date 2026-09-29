# heterogeneous-quadratic

Three federated problems on one construction, whose heterogeneity dials act one
at a time: the **heterogeneous quadratic** (smooth convex), the **lasso**
(nonsmooth convex, a lasso with a diagonal design, the planted version) and the
**Geman–McClure double well** (smooth nonconvex). Every closed form a run is
scored against is in the manifest's `reference`, and every identity below is
asserted on every dataset the generator writes (`check_identities`).

```bash
fedbrew generate --config data/configs/examples/heterogeneous-quadratic.yaml
fedbrew run --config configs/examples/heterogeneous-quadratic/fedavg_k10.yaml
python examples/heterogeneous-quadratic/run.py      # all eight arms, then a table
```

## The construction

`d` coordinates, `N` clients (a power of two, `2d ≤ N − 1`), a planted centre
`x̂`, one scalar profile `φ`:

```
f_i(x) = Σ_j a_ij φ(x_j − x̂_j) + (g + ζ_i)ᵀ(x − x̂) + λ‖x‖₁,     F = (1/N) Σ_i f_i
a_ij   = ā_j (1 + ε s_ij),   ā_j = κ^{j/(d−1)}
ζ_ij   = (ζ*/√d)(ω s_ij + √(1−ω²) r_ij)
```

`s_·j` and `r_·j` are columns `1 + j` and `1 + d + j` of the Sylvester–Hadamard
matrix of order N: ±1, summing to zero, orthogonal to each other. So
`(1/N) Σ‖ζ_i‖² = ζ*²` exactly, and ω is exactly the Pearson correlation across
clients between a client's curvature perturbation `s_ij` and its shift `ζ_ij`,
in every coordinate. `x̂` is ±1 alternating on `{0, 3, 6, 9, 12, 15}`, 0
elsewhere. The shipped size is `d = 16`, `N = 64`.

| member | φ | λ, g | x\*, F\* | start x₀ |
|---|---|---|---|---|
| `heterogeneous_quadratic` | `u²/2` | 0, 0 | `x̂`, 0 | `x̂ + 1/√ā`: gap 8.000 |
| `lasso` | `u²/2` | λ > 0, `g = −λ s*` | `x̂` (soft-thresholding of the smooth centre), `λ‖x̂‖₁` | `x̂ + 1/√ā`: gap 13.956 at λ = 1 |
| `geman_mcclure` | `u²/2 − θ u²/(1+u²)` | 0, 0 | `x̂ + u₀` (all `2^d` minima `x̂ + {±u₀}^d` are global), `φ(u₀) Σ ā_j` | `x̂ + u₀/4`, in negative curvature: gap 4.722 at θ = 1, ζ\* = 0.25 |

- **The lasso's `g`.** `s*_j = sign(x̂_j)` on the support and `±0.5` (alternating)
  off it. The linear term keeps the minimiser at `x̂` and the clients' gradient
  dissimilarity at `x*` equal to `ζ_i`, so ζ\*, ω and the controls mean what they
  mean in the quadratic. Without it the ℓ1 term itself would couple curvature and
  shift on the support. Every client's own minimiser has `x̂`'s support exactly
  when `max |ζ_ij| < λ(1 − 0.5)`: at λ = 1 none of the 1,024 client–coordinate
  pairs differ, at λ = 0.25 320 do (`support_disagreement`).
- **The double well.** `u₀ = √(√(2θ) − 1)` = 0.643594 at θ = 1, `φ''(0) = −1`,
  `φ''(u₀)` = 1.171573. Every client keeps both wells in every coordinate only
  while `max |ζ_ij|/a_ij < max |φ'|` on `[0, u₀]` = 0.20675: ζ\* ≤ 0.29 at
  ε = 0.5, so the member's centre is ζ\* = 0.25 (`both_wells_kept`).

## The dials and the controls

| dial | meaning |
|---|---|
| κ | condition number: μ = 1, L = κ for the quadratic part |
| ζ\* | first-order heterogeneity at the optimum, `(1/N) Σ‖∇f_i(x*)‖²` = ζ\*² |
| ε | Hessian dissimilarity εκ; second-order heterogeneity τ = 2εκ |
| ω | the curvature–shift correlation |
| λ | the lasso member's ℓ1 weight |
| θ | the double well's depth, θ > ½ |
| σ | per-row noise (below) |

The curvature-weighted gradient dissimilarity at the optimum,
`δ*_j = (1/N) Σ_i H_i(x*)_jj ζ_ij`, is `εωζ*ā_j/√d` for the quadratic members,
so `δ*ᵀH̄⁻¹δ* = (εωζ*)² mean(ā)` (1.003826 at the centre). It is zero in three
controls, and there the quadratic's FedAvg has no drift at all:

| data config | cell | dials |
|---|---|---|
| `heterogeneous-quadratic.yaml` | coupled (the centre) | κ 10, ζ\* 1, ε 0.5, ω 1, σ 20 |
| `…-iid.yaml` | iid: every client is F | ζ\* = ε = 0 |
| `…-shift.yaml` | shift-only | ε = 0 |
| `…-decoupled.yaml` | decoupled | ω = 0: the coupled cell's ζ\*, τ and κ, uncorrelated |
| `…-lasso*.yaml`, `…-double-well*.yaml` | the same four cells of the other members | λ = 1; θ = 1, ζ\* = 0.25 |
| `…-lasso-support.yaml` | shift-only at λ = 0.25 | support disagreement at δ\* = 0 |

## FedAvg's exact floor

With exact gradients (σ = 0) and every client every round, FedAvg at step α and
K local steps settles where the manifest's `fedavg_exact_floor` says, for
α ∈ {0.1, 0.01, 0.001} and K ∈ {1, 10, 100}:

- the quadratic: its fixed point in closed form, per coordinate
  `x̂_j + Σ_i C_ij c_ij / Σ_i C_ij` with `C = 1 − (1 − αa)^K`, `c = −ζ/a`
  (checked against the round map composed and iterated);
- the lasso: off the support there is no fixed point (constant-step subgradient
  steps chatter around 0), so the floor is the mean gap of the exact round map
  iterated from `x̂`, over its last 2,000 rounds;
- the double well: its fixed point near `x̂ + u₀`, a bracketed root per
  coordinate.

At the centre, the gap at the floor:

| α | K = 10 | K = 100 |
|---|---|---|
| quadratic, 0.1 | 3.560e-2 | 8.898e-2 |
| quadratic, 0.01 | 1.014e-3 | 3.472e-2 |
| quadratic, 0.001 | 1.019e-5 | 1.140e-3 |
| lasso (λ 1), 0.01 | 3.365e-2 (3.883e-2 at K = 1: the subgradient step floor) | 4.674e-2 |
| double well, 0.01 | 7.286e-5 | 1.995e-3 |

`tests/test_heterogeneous_quadratic.py` runs FedAvg with exact
gradients at α = 0.1, K = 10 and finds the closed form's floor to 1e-9.

## The stochastic oracle

Each client holds `2d` rows. Row r's loss gradient is `∇f_i(x) + σ z_r`, with
`z_r` the rows of the order-d Sylvester–Hadamard matrix over `√d` and their
negatives: `Σ_r z_r = 0`, `‖z_r‖² = 1`, `Cov(z) = I/d`, exactly. So the rows'
mean is `f_i`, and a minibatch of b rows drawn **iid with replacement** has
`E‖ξ‖² = σ²/b` whatever x, the client or the member.

The arms sample it with fedbrew's `client.sampling: with_replacement`
(`fedbrew/clients/sampling.py`, docs/04 §5.1), which any task that gives its
rows can use: every training pass is **one** minibatch of `batch_size` rows,
drawn with `torch.randint` from a generator seeded once, and the batched
executor plans every client's draws together from the same seed. Hence:

- `update_mode: single_batch` at `local_iterations: K` takes K iid minibatches;
- SCAFFOLD's own loop, `sequential_epoch`, does too: one pass is one batch;
- minibatch SGD at batch K·b is `local_iterations: 1` at `batch_size: K·b`.

Evaluation passes are not shuffled and read every row in order, through the
task's own loader, which is the other examples': every row in batches, permuted
once when shuffled. `update_mode: full_gradient` is refused beside
`with_replacement`; with the default sampling it is the exact gradient of every
row, which at σ > 0 averages the noise rows out exactly.

## The arms

Eight, on the coupled centre: `fedavg_k{1,10,100}` (fixed step, K iid minibatches
of 16), `minibatch_sgd_k{10,100}` (one step on a minibatch of K·16, the
equal-sample baseline) and `scaffold_k{1,10,100}`. Point `data.path` at another
data config for another member or control; the model block carries no dial.

fedbrew's SCAFFOLD is option II (`c_i⁺ = c_i − c + (x − y_i)/(Kη)`, K the steps
taken), with global step 1: the new model is the example-weighted mean of the
sampled clients' models, equal weights here, and `c` moves by
`(1/N) Σ_{i∈S} Δc_i` over the whole roster. Each `c_i` is kept on its client
between the rounds it is sampled in, so it goes stale while it is not.

Every metric: `central_test_optimality_gap` (`F(x) − F*`),
`central_test_distance_to_optimum` (to the nearest minimum for the double well),
`central_test_off_support_small` (the fraction of off-support coordinates with
`|x_j| ≤ 1e-3`, descriptive: subgradient iterates rarely reach exact zero) and
`grad_norm_sq`, which for the lasso is the squared norm of F's minimum-norm
subgradient.
