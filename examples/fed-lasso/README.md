# examples/fed-lasso — least squares plus an L1 term, over a planted sparse signal

*One of the problems in [`examples/`](../README.md); that index says what each one is for.*

**No shipped algorithm can solve this one.** There is no proximal operator
anywhere in fedbrew, so all nine arms below run subgradient descent on a
non-smooth objective, and not one of them produces a single exact zero on a
problem whose answer is thirteen-sixteenths zeros. The problem is here anyway,
because the gap is worth being able to see and measure.

The example ships one dial, `problem.penalty`, which replaces the L1 term with
an L2 one at the same λ. Every arm converges on that setting and still produces
no zeros — [the smooth control](#the-smooth-control-ridge-instead-of-lasso)
below.

Client `i` holds an `m x d` design and its own targets, and optimises

```
f_i(x) = (1/2m)‖H x − y_i‖² + λ‖x‖₁
```

`H` is the `d × d` Sylvester-Hadamard matrix with `m = d` rows, so `HᵀH = mI`
**exactly**: the design is orthonormal, every client shares it, and the smooth
part reduces exactly to `½‖x − θ_i‖²` with `θ_i = Hᵀy_i/m` the client's OLS
solution. The targets are built from `θ_i = x_true + w + u_i`, with the `u_i` in
exact ± pairs, so they sum to zero and

```
F(x) = ½‖x − x_true‖² + λ‖x‖₁ + ζ²/2        x* = S_λ(x_true)        F* = 0.16375
```

where `S_λ` is soft thresholding. An orthonormal design is what makes the lasso
solution closed-form, so `x*`, its support, and `F*` are all known before the
run starts — and so is the wall every arm below stops at.

## Two ground truths, and why the difference is the point

| | what it is | value at the defaults |
| --- | --- | --- |
| `x_true` | the planted signal: 3 non-zeros in 16 coordinates, magnitudes 1.0, −0.5, 0.25 | support `{0, 5, 10}` |
| `x*` | the minimiser of `F`: `x_true` shrunk by λ on its support, exactly zero off it | support `{0, 5, 10}`, values 0.95, −0.45, 0.20 |

`‖x* − x_true‖ = λ√s = 0.0866`, in closed form. **That is a floor, not an
error.** An arm that drove `‖x − x_true‖` below it would be further from the
minimiser of the objective it was actually given. The `‖x − x_true‖` column in
the table below sits between 0.0850 and 0.0871 on every arm — the floor, plus
noise — and it is reported *with* the floor beside it, because on its own it
reads like a recovery score and is not one.

`F*` itself is not zero, and 0.08 of it — 49% — is `ζ²/2`, the heterogeneity
residual that no `x` removes. `central_test_loss` is `F(x)`, so unlike
[`pl-1d`](../pl-1d/) and [`drift-quad`](../drift-quad/), where `F* = 0` made the
loss and the gap the same number, here they differ by a constant and only
`central_test_optimality_gap` says how well the run did.

Both floors travel with the data: the generator writes `x_true`, `x*`, `F*`,
`truth_distance_floor` and `heterogeneity_residual` into the manifest under
`reference`, so `run.json` records what the run was scored against.

## Running it

Generate the data, then run an arm. Both are the ordinary commands; nothing
here is specific to this example except the two config paths.

```bash
fedbrew generate --config data/configs/examples/fed-lasso.yaml
fedbrew inspect-data data/generated/examples/fed-lasso/manifest.json

fedbrew run --config configs/examples/fed-lasso/fedavg.yaml
fedbrew run --config configs/examples/fed-lasso/fedavg_decay.yaml --validate-only
```

The problem is defined outside the package, in [`problem.py`](problem.py), and
reaches the CLI because each config names it: `dataset.extensions` in the
generator config, `experiment.extensions` in every arm config. Chapter 12 is the
general form and [`drift-quad`](../drift-quad/) is its worked case.

Each control has its own generator config, because λ and the penalty form are
both part of the objective and so part of what the data is scored against:

```bash
fedbrew generate --config data/configs/examples/fed-lasso-l2.yaml      # ridge
fedbrew run --config configs/examples/fed-lasso-l2/fedavg.yaml

fedbrew generate --config data/configs/examples/fed-lasso-smooth.yaml  # λ = 0
fedbrew run --config configs/examples/fed-lasso-smooth/fedavg.yaml
```

`run.py` is a convenience over those commands and nothing more — it runs a
directory of arm configs in order through the `fedbrew` CLI, then builds the
comparison table by reading `outputs/`:

```bash
python examples/fed-lasso/run.py                        # 9 arms, CPU
python examples/fed-lasso/run.py --setting fed-lasso-l2 # 9 arms
python examples/fed-lasso/run.py --setting fed-lasso-smooth
python examples/fed-lasso/run.py --table-only           # re-table what is on disk
```

Each arm writes `outputs/examples/fed-lasso/<arm>/` with the usual four
artifacts, and each one starts a fresh interpreter that imports torch — the
honest cost of every arm being a real `fedbrew run`. The configs also pin
`torch_num_threads: 1`, which is **not** what makes a run reproducible — see
[`simplex-lsq`](../simplex-lsq/)'s README for the claim that was retracted.

## Which shipped algorithms can solve it

None, because none of them applies a proximal operator. Reaching the exact zeros
of `g(x) + λ‖x‖₁` takes a proximal step such as `x ← S_ηλ(x − η∇g(x))`, and
nothing under `fedbrew/clients/` or `fedbrew/servers/` applies a proximal
operator to anything. What every arm does instead is take the
subgradient `∇g(x) + λ·sign(x)`, which is a legitimate method with two
consequences the table has to be read through:

* **It never produces a zero.** `sign(x)` pushes a coordinate toward 0 and past
  it; only a prox can land on it and stay. The `exact zeros` column is **0 for
  every arm at every round after the first**, from a start (`x_init: 0.0`) where
  it was 16. The sparsest point the run ever visits is the one it begins at.
* **A constant step leaves an `O(ηλ)` floor**, which is what eight of the nine
  arms are sitting on at round 150.

`fedprox` is not the exception. It adds `μ/2‖w − w_global‖²`, a *smooth* penalty
pulling the local iterate toward the broadcast model; it has nothing to do with
`prox_{ηλ}(·)` and produces no zeros. Its entire `proximal_mu` grid lands within
0.1% of the FedAvg arm. The name is the trap this example exists to name: on a
lasso, a row labelled `fedprox` reads like the right tool.

There is one shipped thing that genuinely helps, and it is not an algorithm: a
**decaying step size**. `client.learning_rate_schedule: cosine` anneals η over
the run, which is precisely the textbook fix for a subgradient method's floor,
and the `fedavg_decay` arm is 390× better on the gap than the same client rule
with a constant step. It still produces no zeros.

So a run demonstrates: what subgradient descent does to a composite objective —
a gap floor set by η and λ, no sparsity at any round, and a support that exists
only at a threshold — measured against an optimum that is known exactly and that
no arm reaches.

## What the shipped strategies do on it

150 rounds, d = 16, 8 clients, all participating, m = 16 rows each, batch 4, 3
local iterations of one pass each (12 local steps per round), λ = 0.05, ζ = 0.4,
seed 42.
`F(x₀) − F* = 0.5725`. Every arm is the best of the grid in **How the arms were
tuned**.

| arm | gap @ 150 | median gap, 136–150 | best gap | first < 1e-3 | `‖x − x*‖` | `‖x − x_true‖` | support size | support F1 | **exact zeros** | params/round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fedavg | 4.08e-05 | 5.55e-05 | 4.04e-05 | 67 | 5.15e-04 | 0.0868 | 3 | 1.00 | **0** | 16 |
| **fedavg_decay** | **1.05e-07** | 1.32e-06 | 1.05e-07 | **22** | 4.10e-04 | 0.0862 | 3 | 1.00 | **0** | 16 |
| fedprox | 4.08e-05 | 5.55e-05 | 4.05e-05 | 67 | 5.16e-04 | 0.0868 | 3 | 1.00 | **0** | 16 |
| fedavgm | 2.65e-05 | 3.45e-05 | 1.63e-05 | 50 | 1.21e-03 | 0.0854 | 3 | 1.00 | **0** | 16 |
| fedadam | 3.87e-04 | 3.72e-04 | 1.64e-04 | 78 | 2.94e-03 | 0.0858 | 4 | 0.86 | **0** | 16 |
| fedyogi | 2.91e-04 | 4.82e-04 | 1.77e-04 | 51 | 2.06e-03 | 0.0861 | 4 | 0.86 | **0** | 16 |
| fedadagrad | 3.17e-04 | 3.31e-04 | 1.85e-04 | 51 | 2.60e-03 | 0.0862 | 5 | 0.75 | **0** | 16 |
| scaffold | 1.20e-04 | 1.02e-04 | 4.10e-05 | 67 | 8.01e-04 | 0.0868 | 3 | 1.00 | **0** | 32 |
| fedlalr | 1.56e-04 | 1.00e-04 | 5.27e-05 | 67 | 1.50e-03 | 0.0871 | 3 | 1.00 | **0** | 48 |
| **`x*`** *(not an arm)* | **0** | 0 | 0 | 1 | **0** | **0.0866** | **3** | **1.00** | **13** | — |

The last row is the reference, and it is the only row in the table that has a
zero in it. Everything above it is within a factor of 400 of everything else and
within a factor of ~10⁵ of the bottom row on the one column that would matter if
sparsity were the point.

Two arms are worth separating from the noise. `fedavg_decay` is a real
improvement with a stated mechanism — annealing η removes the `O(ηλ)` term, so
it ends 390× lower than the constant-step arm and keeps falling where every
other arm has flattened — and the three per-coordinate
adaptive arms (`fedadam`, `fedyogi`, `fedadagrad`) end 7 to 10 times above
FedAvg on the gap, with the densest thresholded supports. It is not the answer
being mostly zeros: in the ridge control below, `x*` has the same thirteen zeros
and the same three arms are among the best four. What differs is the gradient
at those zeros. The subgradient `λ·sign(x)` keeps its magnitude on the order of
`λ` however close a coordinate gets to 0, and a step divided by the coordinate's
own gradient scale does not shrink with it. That is a hypothesis both tables
fit, not a measured cause. Nothing else in the table is separated by more than
the round-to-round oscillation the `median` column shows.

`central_test_loss` and `test_loss_sample_weighted_avg` agree to 1.1e-16 on
every arm, which checks the per-client weighting.

## The smooth control: ridge instead of lasso

`problem.penalty: l2` swaps `λ‖x‖₁` for `λ‖x‖²/2m` and changes nothing else —
same planted signal, same λ, same design, same offsets, same eight clients, same
grid, same 150 rounds:

```
f_i(x) = (1/2m)(‖H x − y_i‖² + λ‖x‖²)
F(x)   = ½‖x − x_true‖² + λ‖x‖²/2m + ζ²/2
x*     = (HᵀH + λI)⁻¹Hᵀȳ = (m + λ)⁻¹Hᵀȳ = m·x_true/(m + λ)      F* = 0.08204439
```

Its own generator config — `data/configs/examples/fed-lasso-l2.yaml` — for the
same reason the null control has one: the penalty is part of the objective and
so part of what the data is scored against.

**λ is the same number and not the same scale.** The L2 term carries the
residual's `1/m` and the L1 term does not, which is what makes the closed form
the textbook ridge estimator rather than something with a loose `m` in it. So
λ = 0.05 shrinks by a *factor* of `m/(m+λ) = 0.996885` here, against L1's
*absolute* `λ√s = 0.0866`. Two penalties at the same λ is the honest comparison;
a λ picked to match the shrinkage would make the two tables look alike by
construction, which is the opposite of what a control is for.

| arm | gap @ 150 | median gap, 136–150 | best gap | first < 1e-3 | `‖x − x*‖` | `‖x − x_true‖` | support size | support F1 | **exact zeros** | params/round | *L1 gap @ 150* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fedavg | 8.67e-10 | 8.64e-10 | 5.47e-12 | 34 | 4.16e-05 | 0.003527 | 3 | 1.00 | **0** | 16 | *4.08e-05* |
| fedavg_decay | 2.80e-11 | 2.81e-11 | 8.38e-13 | 22 | 7.48e-06 | 0.003562 | 3 | 1.00 | **0** | 16 | *1.05e-07* |
| fedprox | 8.60e-10 | 8.87e-10 | 1.19e-11 | 34 | 4.14e-05 | 0.003528 | 3 | 1.00 | **0** | 16 | *4.08e-05* |
| **fedavgm** | **2.29e-13** | 1.33e-11 | 2.29e-13 | 41 | 6.76e-07 | 0.003570 | 3 | 1.00 | **0** | 16 | *2.65e-05* |
| fedadam | 1.73e-11 | 1.79e-11 | 1.38e-11 | 12 | 5.87e-06 | 0.003564 | 3 | 1.00 | **0** | 16 | *3.87e-04* |
| fedyogi | 1.34e-11 | 1.54e-11 | 1.20e-11 | 11 | 5.17e-06 | 0.003564 | 3 | 1.00 | **0** | 16 | *2.91e-04* |
| fedadagrad | 1.35e-11 | 1.55e-11 | 1.23e-11 | 11 | 5.20e-06 | 0.003564 | 3 | 1.00 | **0** | 16 | *3.17e-04* |
| scaffold | 1.81e-08 | 2.67e-08 | 5.88e-09 | 34 | 1.90e-04 | 0.003549 | 3 | 1.00 | **0** | 32 | *1.20e-04* |
| fedlalr | 1.58e-06 | 1.58e-06 | 2.12e-08 | 11 | 1.78e-03 | 0.003343 | 3 | 1.00 | **0** | 48 | *1.56e-04* |
| **`x*`** *(not an arm)* | **0** | 0 | 0 | 1 | **0** | **0.003569** | **3** | **1.00** | **13** | — | *0* |

The last column is the same arm's gap on the L1 data, so the contrast is on one
line rather than one page. Every arm was re-tuned on the identical grid — 907
runs, the same selection rule — because a rate tuned against one objective is
not a rate tuned against the other.

**Every arm converges.** The column sits between 2.29e-13 and 1.58e-06, against
L1's 1.05e-07 to 3.87e-04: the *worst* arm here is 245× better than the worst
arm there, and tuned FedAvg is 47 000× better than the same rule on the same
planted signal with the other penalty. That is the `O(ηλ)` floor not existing.
`λx/m` is differentiable, so a constant step has no non-smooth term to oscillate
across — and `fedavg_decay`'s annealing, which was worth 390× under L1 and was
the one shipped thing that genuinely helped, is worth 31× here and no longer
wins the table.

**The three adaptive arms invert.** `fedadam`, `fedyogi` and `fedadagrad` are
the worst three under L1 by a factor of 7 to 10, and three of the best four
here; they are also among the fastest to 1e-3, at rounds 11–12 against FedAvg's
34. The answer is not what changed: `x*` has the same thirteen zeros under both
penalties. The gradient at those zeros is. Under L1 the subgradient keeps its
magnitude on the order of `λ` however close a coordinate gets to 0; under ridge
it goes to zero with the coordinate, and a normalised step shrinks with it once
it falls below `tau`. That is the hypothesis in the L1 section, and this table
is consistent with it rather than a measurement of it.

`fedavgm`'s 2.29e-13 is the best number in the table and it sits at the **top
edge of the server-rate grid** (`server_learning_rate: 3.0`). It is a bound on
the grid rather than an interior optimum, and it should be read as "at least
this good" and not as a tuned result.

**Support recovery is meaningless here, and the table says so by looking
better.** All nine arms report `support size` 3 and `support F1` 1.00, where
under L1 only five of nine did. Nothing was recovered. Ridge multiplies every
coordinate of `θ̄` by one factor and sets none of them to zero, so `x*` is `θ̄`
scaled; at `noise: 0.0`, `θ̄` *is* `x_true`, and `x*` **inherits** its thirteen
zeros rather than finding them. The reference row's `support F1` of 1.00 is the
same number under both penalties and an achievement under only one. Turn `noise`
up by any amount and they part — soft thresholding still returns three non-zeros
where ridge returns sixteen — and `_self_check` asserts exactly that rather than
asserting the flag, which reads `support_recoverable: true` here and means
nothing.

`exact zeros` is **0 on every arm**, as under L1, and for a stronger reason.
There it was a missing proximal operator: the answer has thirteen zeros and no
shipped rule can land on one. Here no method would produce a zero, because the
answer has none that were not already in the data.

**`‖x − x_true‖` has a different floor, and eight of nine arms are below it.**
`‖x* − x_true‖ = (1 − m/(m+λ))‖x_true‖ = 0.003569`, against L1's `λ√s = 0.0866`
— 24× smaller, because ridge at this λ barely moves. And the column is the
sharpest form of the warning the L1 table carries. Only `fedavgm`, the arm
nearest `x*`, is *at* the floor, at 0.003570; `fedlalr`, the worst arm on the
gap, is furthest *below* it, at 0.003343. Being closer to the planted signal
tracks being further from the minimiser, monotonically, down the column. It is
not a recovery score under either penalty.

**What the control does not carry over.** `F*` is 0.0820 against L1's 0.1638,
and 97.5% of it is `ζ²/2`, the heterogeneity residual no `x` removes, against
49% under L1 — so the two `central_test_loss` columns are not comparable and
only the gap is. `central_test_loss` and `test_loss_sample_weighted_avg` agree
to 1.4e-17 on every arm, which checks the per-client weighting the same way the
L1 table does.

## Support recovery is a threshold, not a result

`support_size` and `support_f1` are measured at `model.support_tolerance`, a
number that is not part of the objective and does not change what any arm
optimises. It changes what the table says. Here is the final iterate of every
arm, scored at seven thresholds — F1, with the support size in brackets:

| arm | τ = 0 | τ = 1e-6 | τ = 1e-4 | τ = 1e-3 | τ = 1e-2 | τ = λ/2 |
| --- | --- | --- | --- | --- | --- | --- |
| fedavg | 0.32 (16) | 0.32 (16) | 0.75 (5) | 1.00 (3) | 1.00 (3) | 1.00 (3) |
| fedavg_decay | 0.32 (16) | 1.00 (3) | 1.00 (3) | 1.00 (3) | 1.00 (3) | 1.00 (3) |
| fedprox | 0.32 (16) | 0.32 (16) | 0.75 (5) | 1.00 (3) | 1.00 (3) | 1.00 (3) |
| fedavgm | 0.32 (16) | 0.32 (16) | 1.00 (3) | 1.00 (3) | 1.00 (3) | 1.00 (3) |
| fedadam | 0.32 (16) | 0.32 (16) | 0.32 (16) | 0.86 (4) | 1.00 (3) | 1.00 (3) |
| fedyogi | 0.32 (16) | 0.32 (16) | 0.35 (14) | 0.86 (4) | 1.00 (3) | 1.00 (3) |
| fedadagrad | 0.32 (16) | 0.32 (16) | 0.38 (13) | 0.75 (5) | 1.00 (3) | 1.00 (3) |
| scaffold | 0.32 (16) | 0.32 (16) | 0.32 (16) | 1.00 (3) | 1.00 (3) | 1.00 (3) |
| fedlalr | 0.32 (16) | 0.32 (16) | 0.32 (16) | 1.00 (3) | 1.00 (3) | 1.00 (3) |
| **`x*`** | **1.00 (3)** | 1.00 (3) | 1.00 (3) | 1.00 (3) | 1.00 (3) | 1.00 (3) |

Every arm scores 0.32 at τ = 0 and 1.00 at τ = 0.01. The column the shipped
config picks — τ = 1e-3 — puts five of nine arms at a perfect score. None of
that is a property of an algorithm; it is a property of a number in the config,
and the reference row is the only one whose score does not move, because `x*`
has actual zeros. Setting `model.support_tolerance: 0.0` in an arm config and
re-running reproduces the first column, and every arm scores 0.32.

Changing that key changes nothing about the run — it is read only when the
support metrics are measured, never in `train_step` — so the two columns come
from the same iterates. That is also why it is the one key here a reader can
change without invalidating a comparison.

This is why `exact_zeros` is a metric. It is the one support number with no free
parameter in it.

## Three controls, because a floor has to be attributed

Each removes one candidate cause, and the three are not interchangeable: the
null control removes the penalty, the smooth control keeps it and removes the
non-smoothness, and the full-batch control removes the sampling.

**The null control: λ = 0, remove the penalty entirely.** Its own generator
config, because λ is part of the objective and so part of what the data is
scored against: `data/configs/examples/fed-lasso-smooth.yaml`. The problem
becomes a perfectly
conditioned least squares, and FedAvg reaches `central_test_optimality_gap` of
**exactly 0.0**, with `‖x − x*‖` at 1.7e-16, at every learning rate from 0.0125
to 0.4. So the entire floor at λ = 0.05 is the L1 term's, and none of it is the
design, the heterogeneity, the participation or the loop.

The shipped control arm runs at η = 0.05 rather than the base arm's tuned 0.004,
and the reason is worth stating: 0.004 was tuned against the λ = 0.05 objective,
and at λ = 0 it reaches 3.3e-07 in 150 rounds — not a floor, just a fifth of the
step. A control has to be run where its claim was made, and the claim is stated
over the whole 0.0125–0.4 band, where the gap is exactly 0.0 throughout.

That run is also the sharpest form of the sparsity point. `x*` at λ = 0 is
`x_true`, which has 13 zeros; FedAvg gets to within 1.7e-16 of it and still
reports **0 exact zeros**. Reaching the optimum and producing zeros are
different things, and only one of them is what a prox is for.

**The smooth control: keep λ, make the term differentiable.**
`data/configs/examples/fed-lasso-l2.yaml`, nine arms, its own section
[above](#the-smooth-control-ridge-instead-of-lasso). It separates the two
things the null control conflates — that there is a penalty, and that the
penalty is non-smooth. The floor is the second one's: at the same λ, with the
term made differentiable, every arm converges and the worst of them is 245×
better than the worst L1 arm.

**Full batch, step-matched — remove the minibatch noise.** CLI flags on the
shipped arm, because batch size is a run knob rather than a property of the
data: `--batch-size 16 --local-iterations 12` gives one batch per pass and the
same 12 local steps per round, so the only thing that changes is the sampling. Tuned
FedAvg reaches 6.41e-05 against batch 4's 4.08e-05: the same floor, within a
factor of 1.6. The floor is not sampling noise.

## The learning rate is squeezed from both sides

Because the floor is `O(ηλ)` and the rate is `O(η)`, the constant-step arms have
a genuine interior optimum rather than a grid edge. FedAvg's whole curve, gap at
round 150, from `--lr` on the shipped arm:

| η | 0.001 | 0.002 | **0.004** | 0.008 | 0.0125 | 0.025 | 0.05 | 0.1 | 0.2 | 0.4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap | 1.55e-02 | 4.33e-04 | **4.08e-05** | 7.69e-05 | 1.34e-04 | 2.51e-04 | 5.13e-04 | 1.05e-03 | 2.45e-03 | 4.43e-03 |

The η = 0.008 cell read `7.70e-05` until this table was re-measured; the value
is `7.694928…e-05`, which rounds to `7.69e-05`. That was a rounding slip in the
original table and not a change the migration caused — every other cell here,
and every cell of every other table in this README, is identical.

Left of the minimum the run has not arrived; right of it, it has arrived
somewhere worse. `fedavg_decay` escapes the trade by starting at η = 0.0125 — three
times the constant-step optimum — and annealing to zero, and its own curve has
no interior optimum worth speaking of: every rate from 0.008 to 0.4 lands
between 1.1e-07 and 2.2e-06.

## How the arms were tuned

One grid, scored on `central_test_optimality_gap` at the final round; the winner
is what the table reports and what each arm config ships as its
`client.learning_rate`. Single seed, and the selection metric is the reported
metric, so these are best-of-grid numbers and not estimates. The tuning is there
so that each arm's floor is its own rather than an artifact of a bad step size —
not so that the arms can be ranked, which on this problem they cannot be.

The grid is here, because a grid is neither a run config nor a generator config
and there is nowhere in the tree it belongs:

```yaml
client.learning_rate:            # log-uniform. The upper end is the minibatch
  scale: log                     # stability bound: the batch Hessian is
  range: [1.0e-3, 5.0e-1]        # m/batch_size = 4 times a projector, so local
  grid: [0.001, 0.002, 0.004, 0.008, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]
  applies_to: [fedavg, fedavg_decay, fedprox, fedavgm, fedadam, fedyogi, fedadagrad, scaffold]
client.learning_rate@fedlalr:    # local AMSGrad's alpha is not on the same
  scale: log                     # scale, so it gets its own band
  range: [1.0e-5, 1.0e-1]
  grid: [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
  applies_to: [fedlalr]
server.server_learning_rate:
  scale: log
  range: [1.0e-2, 3.0]
  grid: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
  applies_to: [fedavgm, fedadam, fedyogi, fedadagrad]
server.beta1:
  scale: linear
  range: [0.0, 0.99]
  grid: [0.0, 0.5, 0.9]
  applies_to: [fedavgm, fedadam, fedyogi, fedadagrad]
client.proximal_mu:
  scale: log
  range: [1.0e-3, 1.0e1]
  grid: [0.01, 0.1, 1.0]
  applies_to: [fedprox]
```

Neither `problem.penalty_strength` nor `problem.penalty` is in the grid, and
neither is tunable: both change the objective, so two values of either are two
different problems with two different optima, and a search over them scored on
the gap would be comparing runs against their own moving targets. That is why
each control is a second generator config rather than a flag.
`model.support_tolerance` is not in it either, for the opposite reason — it
changes nothing about the run and everything about what the run may claim.

The `fed-lasso-l2` arms were tuned on this grid unchanged, 907 runs, scored the
same way. Their winners are in the smooth control's own section; they are not
the L1 winners, which is why re-tuning was necessary rather than tidy.

## Where the abstraction fit, and where it did not

[`pl-1d`](../pl-1d/)'s README and [`drift-quad`](../drift-quad/)'s cover what a
third out-of-tree problem meets again unchanged. Three things are this
example's own, and one of the three is now closed.

### A composite objective does not fit `criterion(outputs, targets)`

`TorchClassificationTask` sets `self._criterion = nn.CrossEntropyLoss()` and
calls it as `self._criterion(model(features), targets)`. A regulariser on the
*parameters* has no way through that signature: `λ‖x‖₁` is not a function of the
outputs or of the targets. `FedLassoTask._criterion` therefore takes three
arguments — `(model, outputs, targets)` — and `LassoModel.penalty()` supplies
the term.

Nothing breaks, because no client rule calls `task._criterion` any more (pl-1d
§3), and every rule reaches the objective through `task.train_step`, which is
free to compute whatever it likes. But the two-argument shape is what a task
author copying `TorchClassificationTask` will assume, and it silently excludes
every objective with a parameter-dependent term: L1, elastic net, a trust-region
penalty, any explicit weight decay that is meant to be part of the reported
loss rather than folded into the optimizer.

### The reference optimum had no config channel at all

**Closed.** This example used to be the third kind of thing that had nowhere to
live: `x*`, `x_true` and `F*` are neither data nor model configuration — they
are properties of the problem, needed by `compute_metrics` and by nothing else —
and `factory._build_task` called a third task's factory with no arguments, so
`FedLassoTask` read them off a registration closure and raised if it was built
before `register()`. There was no way to state them in a run config, and so no
way for `run.json` to record what the run was scored against.

They are now properties of the *data*, because the data is generated. The
manifest's `reference` carries `x_true`, `x*`, `F*`, both floors, the planted
and the realised support, and every client's own support size;
`run_metadata.build_dataset_provenance` copies it into `run.json`, and the task
reads it from the `dataset_metadata` it is handed. The task rebuilds its
`ProblemSpec` from the dials recorded there rather than carrying `n × m` floats
through JSON, and `_self_check` asserts that the round trip is exact.

λ is the one dial that lives in two places, because it is genuinely two things:
part of the objective the client descends (so `model.extra`, the channel a run
config carries it in) and part of what the data is scored against (so the
manifest's `reference`). `FedLassoTask` is the one object handed both, so it is
where they are cross-checked — a λ mismatch is refused rather than run, because
it would fail on nothing at all and score the run against the wrong optimum.

### `cosine` to zero used to kill the final round

Annealing to zero is the natural thing to ask for on this problem, and it was
the first thing tried, and it did not work.
`torch_sgd_client._round_learning_rate` returns exactly `min_learning_rate` when
`round_id == total_rounds`, and `local_update_modes.run_sgd_update_mode` refuses
a non-positive learning rate — so `learning_rate_schedule: cosine` with
`min_learning_rate: 0.0`, a pair `validate_config` accepts without comment, ran
every round but the last and then raised `ValueError: learning_rate must be
positive`.

The exception was not the cost. What it left behind was: `global_rounds − 1`
rows in `round_metrics.csv`, `run.json` still saying `status: running` with
`num_rounds` equal to the rounds that did finish, the second-to-last round's
checkpoint, and nothing on disk saying the run failed. A collector reads a
finished run of the wrong length.

Fixed and recorded as `POST-F01` in `FINDINGS.csv`:
`_round_learning_rate` clamps the scheduled rate to
`local_update_modes.MIN_POSITIVE_LEARNING_RATE`, the smallest positive float,
placed beside the guard that requires positivity. The final round becomes the
no-op the schedule asked for, and `client_learning_rate` records `2.2e-308`, so
the bottoming-out is visible rather than silent. The `constant` branch is not
clamped, so a configured zero still fails loudly.

The `fedavg_decay` arm uses `min_learning_rate: 0.0` for that reason — it is the
honest request, and annealing the whole way is what makes it the best arm in the
table: `1.05e-07` against `9.07e-07` when the schedule was floored at `1e-4` to
work around the crash.

## Files

| File | What it is |
| --- | --- |
| `problem.py` | the whole extension: the objective under either penalty, its analytic (sub)gradient, the soft-threshold and ridge-shrinkage maps that define `x*` (and that no arm runs), the planted signal, the closed-form optimum and every floor it implies, the generator that writes the shards and the manifest, the model, the task adapter, and `register()` |
| `run.py` | a convenience over the CLI: runs a directory of arm configs, then tables `outputs/`. Registers nothing, composes no config |
| `README.md` | this file |

And outside the example, where every other dataset and arm keeps theirs:

| Path | What it is |
| --- | --- |
| `data/configs/examples/fed-lasso.yaml` | the generator config: the dials, and the extension that reads them. `fed-lasso-l2.yaml` beside it is the smooth control and `fed-lasso-smooth.yaml` the λ = 0 null control |
| `configs/examples/fed-lasso/*.yaml` | one run config per arm, tuned. `configs/examples/fed-lasso-l2/` holds the smooth control's nine, tuned on the same grid; `configs/examples/fed-lasso-smooth/` holds the null control's one arm |
| `data/generated/examples/fed-lasso/` | the shards, `clients.jsonl` and the manifest — including `reference`, the closed-form optimum and both floors this run is scored against |
| `outputs/examples/fed-lasso/<arm>/` | what a run wrote |
