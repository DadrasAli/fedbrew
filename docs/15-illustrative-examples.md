# 15 — Illustrative examples

Five small optimisation problems with known optima, each run through the real
federated loop with the shipped strategies. What they are for, what each one
shows, and the discipline a new one has to follow.

The examples live in [`examples/`](../examples/). This chapter is the
discipline; chapter 12 is the mechanism.

## 1. What an illustrative example is

**A controlled problem with ground truth, run through the production path.**
Four properties, and all four have to hold at once or the thing stops being
useful:

| Property | Why it is required |
| --- | --- |
| The optimum is known in closed form | An arm either reaches `x*` or does not. Without that, "converged" is a judgement about a curve's shape |
| The gradient is analytic | A wrong answer is the algorithm's, not the differentiation's — and `problem.py` checks autograd against the analytic form rather than trusting it |
| It runs in minutes on a laptop CPU | Anyone can re-run the table under a claim. A number nobody can reproduce is a citation, not evidence |
| It reaches the loop through `fedbrew run` | The thing being demonstrated is the shipped code path, including its guards, its artifacts and its config validation |

## 2. What an illustrative example is not

**Not a second experiment suite.** The tables in these READMEs are not results.
They exist to make one mechanism visible on a problem where the mechanism is
the only thing that could have produced the number.

**Not a benchmark.** No arm ranking here is transferable. Several tables put
arms in an order and then say, in the same section, that the order is an
artefact of one seed and a floor the arms share — because a table that reads
like a fair comparison when the arms are not comparable is exactly the failure
these examples exist to expose.

**Not a test.** The suite guards behaviour; an example demonstrates it. An
example that is only ever checked by a test is a slow test, and one that is
only ever read is a blog post. They ship as configs plus a `README.md` so both
uses stay open.

**Not a private code path.** None of the five registers anything at import,
composes a config in Python, or reaches the round loop except through the CLI.
Every table was re-measured through `fedbrew run`, and every number matched
what the earlier private runners produced except two, both wrong before the
migration (`FINDINGS.md`, *Two corrections that are not rows*).

## 3. The five that ship

One `problem.py` per example holds the objective, its analytic gradient, the
reference optimum, the generator that writes the data, and a `register()` that
names all of it. `examples/README.md` is the index; this table is derived from
it and guarded against it in both directions.

| Example | The problem | What a run demonstrates | Which shipped algorithms can solve it |
| --- | --- | --- | --- |
| [`pl-1d`](../examples/pl-1d/) | `F(x) = x² + 3sin²(x)` per client, tilted by a per-client shift. Scalar, non-convex, satisfies PL with `μ = 1/32`. `x* = 0`, `F* = 0` | That an analytic per-client objective fits the task / client / server abstraction additively, and that SCAFFOLD's control variates cancel a sampled-subset bias exactly | All seven arms run. SCAFFOLD reaches machine zero; the rest stall on a floor set by the sampled shift |
| [`drift-quad`](../examples/drift-quad/) | `f_i(x) = ½xᵀAx − b_iᵀx`, shared diagonal `A`, per-client offsets. Dials for condition number `κ` and gradient dissimilarity `ζ`. `x* = 0`, `F* = 0` | That κ sets the rate and ζ sets the floor, that each is fixed by a different family, and that neither family fixes the other's dial — with both dials turnable to zero as controls | All eight arms run. Only SCAFFOLD reaches `x*` under partial participation; every other arm converges to a ζ-dependent floor |
| [`fed-lasso`](../examples/fed-lasso/) | `f_i(x) = (1/2m)‖Hx − y_i‖² + λ‖x‖₁` over a planted 3-sparse signal, orthonormal design. `x* = S_λ(x_true)` in closed form. A `penalty` dial swaps the L1 term for `λ‖x‖²/2m` at the same λ, where `x* = (m+λ)⁻¹Hᵀȳ` | What subgradient descent does to a composite objective: a floor set by `ηλ`, no sparsity at any round, and a support that exists only at a threshold — with the smooth control removing the non-smoothness and not the penalty, so every arm converges and still produces no zeros | **None.** There is no proximal operator in fedbrew, so no arm produces a single exact zero. `fedprox` is not the exception — its penalty is smooth. A decaying step size helps the gap and not the sparsity. All nine solve the L2 setting, which is a different problem and not a rescue |
| [`simplex-lsq`](../examples/simplex-lsq/) | Least squares over the probability simplex `Δ`, orthonormal design, with the unconstrained optimum placed outside `Δ`. `x* = Π_Δ(θ̄)` in closed form | That the objective column inverts: every arm converges to an infeasible point and ends with a **negative** optimality gap, beating `F*` by leaving the feasible set | **None.** No projection, no Frank-Wolfe step and no mirror map, and no hook one could attach to. Every arm converges — to the wrong set |
| [`nonconvex-simplex`](../examples/nonconvex-simplex/) | Motzkin-Straus: `−½xᵀAx` over `Δ` for a `K₅` disjoint from a star `K₁,₂₅`, so the clique number is the clique's and the spectral radius is the star's. `F* = −½(1 − 1/ω) = −0.4` | That every arm diverges, no divergence detector fires, all eight report `status: completed`, and projecting the result afterwards is worse than never having run | **None**, and the unconstrained problem is unbounded below. `blowup_factor` never arms because the loss is never positive |

**Three of the five pose problems no shipped algorithm can solve.** That is the
point of having them, not a gap in the set. A problem is worth stating before
its algorithm exists, and each of those three names what is missing —
a proximal operator, a projection, a bounded feasible set — in a section of its
own README rather than in a footnote under a table.

`nonconvex-simplex` is the sharpest: eight arms diverge, no detector fires, and
every run records `status: completed`. A framework that reports success on a
problem it cannot represent is worth being able to point at.

## 4. The rules a new example must follow

Each rule exists because its absence produces a specific wrong claim.

| Rule | Without it |
| --- | --- |
| **State the selection criterion before the numbers.** Say which metric picked the configuration and over which rounds | Best-of-grid numbers read as estimates. They are not: the selection metric and the reported metric are the same quantity, and the README has to say so |
| **Equal tuning budget per arm.** One grid, run identically for every arm, re-tuned per dial setting | An arm that got a wider grid looks better than an arm that got a narrower one, and nothing in the table shows which is which |
| **Multiple seeds, reported as median and IQR — where seeds apply** | A single seed on a floor-limited arm reports one phase of an oscillation. Where an example ships one seed, it says so in the same section as the table |
| **Assert the ground truth in `problem.py`, not in a plot.** A `_self_check()` that runs at import | A plot showing convergence to the right place cannot distinguish a correct problem from a correct-looking one. `drift-quad`'s check is the pattern: offsets sum to exactly zero, the realised ζ equals the dial, `A⁻¹bᵢ` really is client *i*'s minimiser, the manifest's reference optimum is the module's, and autograd agrees with the analytic gradient to 1e-15 |
| **No cluster, no network, no downloads.** CPU, minutes, one process | An example nobody can run is a claim nobody can check |
| **A README that says plainly what a run demonstrates when no arm can converge** | The table becomes a ranking of failures. Naming what is missing turns the same run into a statement about the framework's reach |

The last rule is the one most easily skipped, because a table of numbers looks
like a result whether or not any of the numbers mean what a reader assumes.

### 4.1 Where a dial goes

**A dial is a second generator config, never a run-config flag.** `drift-quad`
ships `drift-quad`, `drift-quad-rate` and `drift-quad-floor`; `fed-lasso` ships
a `-smooth` null control and a `-l2` smooth one; `simplex-lsq` ships a
`-feasible` one. A dial that changes the answer changes what the data is scored
against, so it belongs on the side of the line that writes the data.

It is also where a second *example* is not the answer. `fed-lasso-l2` replaces
the L1 term with an L2 one and every arm converges, which is a different result
on the same problem rather than a different problem — the design, the planted
signal, the offsets and the client count are untouched. It gets a generator
config, nine arm configs and a section of the existing README, and the index
above carries it in `fed-lasso`'s row.

## 5. How to write one

1. **Chapter 12 is the mechanism.** The extension protocol, what `register()`
   may do, which registries take `config_keys`, and what the factory hands an
   out-of-tree component.
2. **This chapter is the discipline.** §4, every rule with its reason.
3. **`examples/drift-quad/` is the file to copy.** It is chapter 12's worked
   case, it is the only one with both dials and both controls, and its
   `_self_check()` is the fullest of the five.

Then: a generator config under `data/configs/examples/`, one run config per arm
under `configs/examples/<name>/`, a `run.py` that runs that directory and
tables `outputs/`, and a `README.md`.

## 6. What is not built, and why

**A tune/run split with a committed `best.yaml`.** Today each arm config ships
the winning hyperparameters inline, and the grid that chose them lives in the
README as a fenced block. Nothing machine-readable connects the two: you cannot
re-run the tuning, and you cannot diff what shipped against what won. The fix
is a committed `best.yaml` per example plus a tuning subcommand that writes
it — a real feature, deliberately not faked by a table. (Written that way
because there is no such subcommand: `tests/test_cli_commands_exist.py` refuses
a `fedbrew <name>` written anywhere in the tree that `COMMANDS` does not know,
and it refused this paragraph's first draft.)

**A `status` field marking comparison versus failure-demo.** Three of the five
are failure demonstrations, and a reader learns which from prose. A machine
reader — a docs guard, a summary table, anything that walks `examples/` —
cannot tell a table meant as a comparison from a table meant as evidence that
nothing works. `tests/test_docs_illustrative_examples.py` currently asserts the
absence of both, so this section stays true or fails.

## For agents

### Paths

| Path | What it is |
| --- | --- |
| `examples/README.md` | the index, and the authority on the five-example table in §3 |
| `examples/drift-quad/problem.py` | the worked case: objective, gradient, generator, `register()`, `_self_check()` |
| `examples/drift-quad/README.md` | the fullest README — two dials, two controls, and the tuning protocol with its own limitations section |
| `configs/examples/<name>/` | one run config per arm |
| `data/configs/examples/` | one generator config per example, plus one per dial setting |
| `docs/12-extending.md` | the extension mechanism these are built on |

### Commands

```bash
# One example, end to end.
fedbrew generate --config data/configs/examples/drift-quad.yaml
fedbrew run --config configs/examples/drift-quad/scaffold.yaml

# Every arm of one example, tabled afterwards.
python examples/drift-quad/run.py

# The chapter's guard.
python -m pytest tests/test_docs_illustrative_examples.py -v
```

### Invariants

1. **The ground truth is asserted in `problem.py` and runs at import.** Every
   one of the five defines `_self_check()` and calls it at module scope. A
   plotted claim is not a checked one.
2. **A dial is a generator config.** Never a run-config flag — §4.1.
3. **Nothing registers at import except the self-check.** `register()` is
   called by the extension loader, from a config that names the file.
4. **An example that no arm can solve says so in its own README**, in a
   section, naming what is missing.
5. **The §3 table is `examples/README.md`'s table.** Both directions: an
   example the chapter omits and a chapter row with no example both fail.
6. **§6 is a claim about the tree, not a wish list.** Building either item
   fails the guard until this chapter is rewritten.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_illustrative_examples.py` | §3's table is exactly the directories under `examples/`, and each row's verdict agrees with `examples/README.md`; every example that no arm solves has its own section saying so; every `problem.py` defines and calls `_self_check()`; §6's two unbuilt items are still unbuilt; every path this chapter names exists. |
| `tests/test_docs_references_resolve.py` | The `docs/` paths named here resolve. |
| `tests/test_docs_extending.py` | The mechanism chapter 12 describes, which §5 points at. |

### Known failure modes

- **Adding an example and not the chapter row.** The guard fails in the
  direction that names the missing row.
- **Turning a dial with a run-config key.** The run then scores against a
  manifest describing different data, and the optimality gap is measured
  against the wrong `F*`.
- **Reading a table as a ranking.** Every table that could be misread says in
  its own section which arms are separated by a mechanism and which are sitting
  on one floor at different phases of one oscillation.
- **Writing a plot instead of an assertion.** A convergence plot cannot
  distinguish a correct problem from a correct-looking one.
