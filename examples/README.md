# examples/

Small, exactly-specified optimisation problems, each run through the real
federated loop with the shipped strategies. They exist to make a claim
checkable: on a problem whose optimum is known in closed form, an arm either
reaches it or does not, and the reason is available rather than inferred.

Every example is an **extension**: one `problem.py` holding the objective, its
analytic gradient, the reference optimum, the ground-truth structure, the
generator that writes its data, and a `register()` that names all of it. A
config names the file — `dataset.extensions` in a generator config,
`experiment.extensions` in a run config — and from there the example is an
ordinary dataset and an ordinary set of arms:

```bash
fedbrew generate --config data/configs/examples/<name>.yaml
fedbrew run --config configs/examples/<name>/<algorithm>.yaml
```

Chapter 12 is the general form and [`drift-quad`](drift-quad/) is its worked
case. Each example also ships a `run.py`, which is a convenience over those
commands — it runs a directory of arm configs and tables `outputs/` afterwards
— and a `README.md`. The constraints are unchanged: seconds on a laptop, CPU
only, no downloads, and nothing that reimplements a round.

All five are on that mechanism. None of them registers anything at import,
composes a config in Python, or reaches the loop except through `fedbrew run`;
every table in every README below was re-measured through the CLI, and every
number matched what the private runners produced except two, both wrong before
the migration (`FINDINGS.md`, *Two corrections that are not rows*). Where an
example turns a dial, the dial is a second generator config rather than a flag,
because a dial that changes the answer changes what the data is scored
against.

## The examples

| Example | The problem | What a run demonstrates | Which shipped algorithms can solve it |
| --- | --- | --- | --- |
| [`pl-1d`](pl-1d/) | `F(x) = x² + 3sin²(x)` per client, tilted by a per-client shift. Scalar, non-convex, satisfies PL with `μ = 1/32`. `x* = 0`, `F* = 0` | That an analytic per-client objective fits the task / client / server abstraction additively, and that SCAFFOLD's control variates cancel a sampled-subset bias exactly | Seven arms ship -- `fedavg`, `fedavgm`, `fedadam`, `fedyogi`, `fedadagrad`, `scaffold`, `fedlalr`; there is no `fedprox` arm here, and there never has been. SCAFFOLD reaches machine zero; the rest stall on a floor set by the sampled shift |
| [`drift-quad`](drift-quad/) | `f_i(x) = ½xᵀAx − b_iᵀx`, shared diagonal `A`, per-client offsets. Dials for condition number `κ` and gradient dissimilarity `ζ`. `x* = 0`, `F* = 0` | That κ sets the rate and ζ sets the floor, that each is fixed by a different family, and that neither family fixes the other's dial — with both dials turnable to zero as controls | All eight arms run. Only SCAFFOLD reaches `x*` under partial participation; every other arm converges to a ζ-dependent floor |
| [`fed-lasso`](fed-lasso/) | `f_i(x) = (1/2m)‖Hx − y_i‖² + λ‖x‖₁` over a planted 3-sparse signal, orthonormal design. `x* = S_λ(x_true)` in closed form. A `penalty` dial swaps the L1 term for `λ‖x‖²/2m` at the same λ, where `x* = (m+λ)⁻¹Hᵀȳ` | What subgradient descent does to a composite objective: a floor set by `ηλ`, no sparsity at any round, and a support that exists only at a threshold — with the smooth control removing the non-smoothness and not the penalty, so every arm converges and still produces no zeros | **None.** There is no proximal operator in fedbrew, so no arm produces a single exact zero. `fedprox` is not the exception — its penalty is smooth. A decaying step size helps the gap and not the sparsity. All nine solve the L2 setting, which is a different problem and not a rescue |
| [`simplex-lsq`](simplex-lsq/) | Least squares over the probability simplex `Δ`, orthonormal design, with the unconstrained optimum placed outside `Δ`. `x* = Π_Δ(θ̄)` in closed form | That the objective column inverts: every arm converges to an infeasible point and ends with a **negative** optimality gap, beating `F*` by leaving the feasible set | **None.** No projection, no Frank-Wolfe step and no mirror map, and no hook one could attach to. Every arm converges — to the wrong set |
| [`nonconvex-simplex`](nonconvex-simplex/) | Motzkin-Straus: `−½xᵀAx` over `Δ` for a `K₅` disjoint from a star `K₁,₂₅`, so the clique number is the clique's and the spectral radius is the star's. `F* = −½(1 − 1/ω) = −0.4` | That every arm diverges, no divergence detector fires, all eight report `status: completed`, and projecting the result afterwards is worse than never having run | **None**, and the unconstrained problem is unbounded below. `blowup_factor` never arms because the loss is never positive |

## What to read them for

The tables in these READMEs are written to be un-mistakable for rankings. Each
one says which arms are separated by a mechanism and which are sitting on the
same floor at different phases of the same oscillation, because a table that
reads like a fair comparison when the arms are not comparable is the failure
these examples exist to make visible.

Where an example poses a problem no shipped algorithm can solve, its README says
so in its own section, names what is missing, and says what the run therefore
measures. The problem is worth having before its algorithm; a table that hides
the gap is not.
