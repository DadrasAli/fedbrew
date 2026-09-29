# examples/fed-logistic-l1 — linear classifiers with a penalty, and the datasets they are posed on

The example keeps the name it was first written under, for l1-regularized
logistic regression; it now poses four problems: l1-regularized and ridge
logistic regression, logistic regression with a nonconvex regularizer, and the
sigmoid loss with a ridge term.

A problem here is a loss of the margin, a penalty on the iterate and its weight
`λ`, posed on a dataset of rows `(a_i, b_i)` with `b_i ∈ {-1, +1}`:

```
min_x  F(x) = (1/n) Σ_{i=1}^n loss(b_i xᵀa_i) + λ r(x)
```

over `x ∈ ℝ^d`, with no constraint set. A **corpus** is one generator config
and one generated dataset, `fed-logistic-l1-<corpus>`; every problem posed on
it is one arm config in its directory, `<loss>-<penalty>-lambda<λ>.yaml`, and
runs on the same shards:

```bash
fedbrew generate --config data/configs/examples/fed-logistic-l1-synthetic.yaml
fedbrew run --config configs/examples/fed-logistic-l1-synthetic/logistic-l1-lambda0.03.yaml
python examples/fed-logistic-l1/run.py --corpus fed-logistic-l1-synthetic
```

The corpus is the data's, and the problem -- the loss, the penalty and `λ` --
is the arm's `model` block. A convex problem's certified optimum is in one
table, [`optima.json`](optima.json), keyed by the corpus's content digest, the
loss, the penalty and `λ` (below).

## The problems

### l1-regularized logistic regression

```
F(x) = (1/n) Σ_i log(1 + exp(-b_i xᵀa_i)) + λ‖x‖₁          (loss: logistic, penalty: l1)
∇ℓ(x) = -(1/n) Σ_i b_i σ(-b_i xᵀa_i) a_i                   ∇²ℓ(x) ⪯ AᵀA / 4n
```

Koh, Kim & Boyd, "An interior-point method for large-scale ℓ1-regularized
logistic regression", JMLR 8, 2007. Convex and non-smooth at every point with
a zero coordinate. `∇ℓ` is Lipschitz
with `L = ‖A‖₂²/4n`, which the manifest records as `reference.lipschitz`. `F`
is coercive, so a minimiser exists without a box. `F*` is certified once per
corpus and `λ`, into the optima table (below).

### Ridge logistic regression

```
F(x) = (1/n) Σ_i log(1 + exp(-b_i xᵀa_i)) + (λ/2)‖x‖²      (loss: logistic, penalty: l2sq)
```

ℓ2-regularized (ridge) logistic regression: le Cessie & van Houwelingen,
"Ridge estimators in logistic regression", JRSS-C 41(1), 1992. Convex, smooth
and `λ`-strongly convex: `∇F = ∇ℓ + λx` is Lipschitz with
`L + λ`, and the minimiser is unique. `F*` is certified into the optima table.
The arms, `logistic-l2sq-lambda<λ>` in each corpus's directory:

| Corpus | `λ` | `F*` |
| --- | --- | --- |
| `fed-logistic-l1-synthetic-1000` | 0.001 | 0.42893336113321356 |
| `fed-logistic-l1-synthetic-kappa1` | 0.01 | 0.4409334313437286 |
| `fed-logistic-l1-synthetic-kappa10` | 0.01 | 0.4593573490876512 |
| `fed-logistic-l1-synthetic-kappa100` | 0.01 | 0.47750977596974997 |
| `fed-logistic-l1-a9a` | 0.001 | 0.3330952806480848 |
| `fed-logistic-l1-ijcnn1-32` | 0.01 | 0.4154526395889211 |
| `fed-logistic-l1-gisette` | 0.001 | 0.4580582604022483 |

### Logistic regression with a nonconvex regularizer

```
F(x) = (1/n) Σ_i log(1 + exp(-b_i xᵀa_i)) + λ Σ_j x_j²/(1 + x_j²)      (loss: logistic, penalty: nonconvex)
r'(u) = 2λu/(1 + u²)²        |r''(u)| ≤ 2λ
```

The benchmark of stochastic nonconvex optimization under that name (Wang,
Ji, Zhou, Liang & Tarokh, SpiderBoost, NeurIPS 2019, experiments). Per
coordinate the regularizer is the Geman–McClure function `u²/(1 + u²)`. A
smooth penalty that behaves like `λu²` near 0 and saturates at `λ` far from
it, so it shrinks small coordinates and leaves large ones alone; it is
nonconvex (`r''` turns negative past `|u| = 1/√3`), so `F` is. There is **no
certified `F*`**: the table holds no entry for it, and a run reports no
`optimality_gap` or `distance_to_optimum`. `F` is smooth with `∇F` Lipschitz
at `L + 2λ`. The arms, `logistic-nonconvex-lambda<λ>`: on synthetic-1000
(`λ = 0.001`), the three kappa corpora (0.01), a9a (0.001), ijcnn1-32 (0.01)
and gisette (0.001).

### The sigmoid loss with a ridge term

```
F(x) = 1 + (1/n) Σ_i tanh(-b_i xᵀa_i) + (λ/2)‖x‖²      (loss: tanh, penalty: l2sq)
loss'(z) = 1 − tanh(z)²      |loss''(z)| ≤ 4/(3√3)
```

The loss `1 − tanh(b aᵀx)` is the sigmoid loss, Mason, Baxter, Bartlett &
Frean's margin cost `1 − tanh(λz)` at `λ = 1` ("Boosting algorithms as
gradient descent", NIPS 12, 1999, eq. 4), which other code calls a "tanh-loss
SVM"; `loss: tanh` names it here. A smooth, bounded surrogate of the 0-1 loss:
nonconvex in `x`, so `F` is, even with the ridge term. `∇F` is Lipschitz at
`(4/(3√3))‖A‖²/n + λ`. There is **no certified `F*`**, as for the nonconvex
penalty. The arms, `tanh-l2sq-lambda<λ>`: on synthetic-1000 (`λ = 0.001`), the
three kappa corpora (0.01), a9a (0.001), ijcnn1-32 (0.01) and gisette (0.001).

## How it is federated

Clients `c = 1..N` hold disjoint row sets `I_c`, all of the same size `m`, and
client `c` optimises `F_c(x) = (1/m) Σ_{i ∈ I_c} loss(b_i xᵀa_i) + λ r(x)`, with
the **whole** penalty in every client, as `examples/fed-lasso` does. At equal
`m`, uniform and example-weighted aggregation are the same average and both are
`F` exactly; `problem.py`'s `_self_check` asserts both spellings on stored rows.

Every client's rows are its train, eval and test split at once
(`client_test_source: identical_to_train`): `F_c` is defined over all `m`
rows, so holding some out would change the objective. The central pass
evaluates `F` on the global shard, every client's rows stacked.

**The deal.** Rows are sorted by a margin (a stable sort) and dealt in blocks
of `partition_block`, round robin: client `c` holds blocks `c, c + N, c + 2N,
…`. A block of 1 is a stratified deal; a block of `m` gives each client one
contiguous band of margins. The block is the heterogeneity dial.

## The corpora

### synthetic — the planted signal on the Halton design

`fed-logistic-l1-synthetic`: 32 clients × 64 rows, `d = 32`, and one arm,
`logistic-l1-lambda0.03`. No draw anywhere:

- **design** `a_ij = Φ⁻¹(vdc(i, p_j))`, the Halton sequence through the normal
  quantile, `p_j` the `j`-th prime;
- **truth** `x_true`: 3 non-zeros at coordinates 0, 10 and 21, values `2.0`,
  `-1.0`, `0.5` (fed-lasso's rule, scaled by `signal_scale = 2`);
- **labels** `b_i = +1` if `σ(x_trueᵀa_i) > vdc(i, p_d)`, else `-1`, with `p_d`
  the first prime the design does not use: a threshold in a base a column
  already uses correlates with that column, and corrupts `x*` while the
  support still matches;
- **clients** dealt by the margin `x_trueᵀa_i` in blocks of 32 — a client sees
  between 22% and 80% positives against a pooled 50%.

At `λ = 0.03`: `x* = (1.456, -0.645, 0.257)` on the planted support, so support
recovery holds (it does over `λ ∈ [0.02, 0.07]`), and
`F* = 0.5121519380638806`, certified to a KKT residual of `1.7e-17`.
`‖x* − x_true‖` is a floor that `distance_to_truth` cannot go below, measured
by solving rather than given by a formula.

### synthetic-1000 — the same generator at 1,000 clients

`fed-logistic-l1-synthetic-1000`: the generator above at 1,000 clients × 32
rows, `d = 32`, dealt in blocks of 16 — many clients, each holding too few
rows to see the planted support. Its L1 `F*` at `λ = 0.001` and `0.03`:
`0.42993436284235004` and `0.5121661415511938`.

### synthetic-kappa1, -kappa10, -kappa100 — a condition-number dial

`fed-logistic-l1-synthetic-kappa<κ>`, `κ ∈ {1, 10, 100}`: the
synthetic generator (32 × 64, `d = 32`, blocks of 32) with its design
reconditioned by `problem.condition_number`. The centred Halton design is
orthonormalised by a reduced QR and its columns rescaled so that the pooled
Gram `AᵀA/n` has eigenvalues `κ^(−j/(d−1))`, `j = 0..d−1`: from 1 down to
`1/κ`. Its condition number is `κ` and `λ_max = 1` for all three, so
`L = 1/4` on each; the truth, the labels and the deal follow from the
reconditioned rows as above. The L1 `F*` at `λ = 0.01`: `0.45512445647395333`,
`0.4738248007229769` and `0.48743008220058726`.

The QR makes the rows depend on the LAPACK they are computed with, to about
`1e-14`, and on how many threads it runs on: one and many give different last
bits, so it runs on one thread, whatever the machine. The manifest records the Gram's spectrum as built (`gram_condition`,
`gram_lambda_max`, `gram_lambda_min`) and what built it (`built_with`: torch,
its BLAS and LAPACK, the CPU); two builds on different machines are the same
problem to that precision and not bit for bit -- and so not the same digest,
which is what the optima table is keyed by.

### LIBSVM corpora: a9a, ijcnn1-32, gisette

Three public binary-classification files from the LIBSVM collection. **The
generator never downloads**: each file is fetched once by hand into
`data/raw/datasets/libsvm/`, and its SHA-256, pinned in the generator config,
is checked before a byte is parsed. A missing file is refused with the `curl`
command that fetches it, and a file with another digest is refused rather than
generating a different problem under the same name.

| Dataset | File | Rows kept | Rows | Clients × rows | `d` | Deal |
| --- | --- | --- | --- | --- | --- | --- |
| `a9a` | `a9a` | the first 32,000 | 0/1 features as they are | 1,000 × 32 | 123 | blocks of 16, by the L1 `x*` at `λ = 0.03`, exact key |
| `ijcnn1-32` | `ijcnn1.bz2` | the first 32,000 | as they are | 32 × 1,000 | 22 | blocks of 50, by the L1 `x*` at `λ = 0.01`, exact key |
| `gisette` | `gisette_scale.bz2` | the first 6,000 | each scaled to unit L2 norm | 100 × 60 | 5,000 | blocks of 30, by the L1 `x*` at `λ = 5e-4` |

The digests are in the generator configs. There is no planted signal on real rows,
so the deal sorts by the margin against a solved `x*` — the certified L1
solution at `problem.partition_reference_lambda`, a dial of the corpus.
`problem.partition_key: exact` takes that margin as each row's products with
the solution rounded to ten significant digits, summed exactly
(`math.fsum`): the same key on any machine, whatever order its BLAS sums in.
`float` takes the design's product with the solution as it is.

On real data the corpus's reference records `lambda_max = ‖Aᵀb‖_∞/2n`, the
scale an L1 `λ` is read against, and `support_f1` is measured against `x*`'s
own support, from the table, since nothing was planted. The L1 arms:

| Corpus | Arm | `F*` |
| --- | --- | --- |
| `fed-logistic-l1-a9a` | `logistic-l1-lambda0.001` | 0.3468377180429282 |
| `fed-logistic-l1-a9a` | `logistic-l1-lambda0.03` | 0.5291036251024969 |
| `fed-logistic-l1-a9a` | `logistic-l1-lambda0.05` | 0.5765317125957747 |
| `fed-logistic-l1-ijcnn1-32` | `logistic-l1-lambda0.01` | 0.42788364502097165 |
| `fed-logistic-l1-gisette` | `logistic-l1-lambda5e-5` | 0.16230244614601907 |
| `fed-logistic-l1-gisette` | `logistic-l1-lambda5e-4` | 0.4220170868705644 |
| `fed-logistic-l1-gisette` | `logistic-l1-lambda0.001` | 0.521996719845196 |

Gisette carries duplicate columns, so its support Hessian can be singular:
the certified solve's Newton step is taken in the Hessian's range, and a
coordinate the step pushes across its own sign leaves the support.

## The certified optima

A convex problem's `x*` has no closed form. It is solved once per corpus and
problem and kept in [`optima.json`](optima.json): the corpus's name and content
digest, the loss, the penalty, `λ`, `x*` by its non-zeros, `F*`, the KKT
residual and `x*`'s support. The generator writes the corpus's digest into its
manifest -- the SHA-256 of the global shard's rows and labels, as float64 bytes
in the order it stacks them, after a header naming the client count and the
shape -- and the task, at the start of a run:

- recomputes the digest of the rows it loaded, and refuses shards that are not
  the digest their manifest records;
- on a convex problem, looks up the entry for (digest, loss, penalty, `λ`),
  reads `x*` and `F*` from it, and reports
  `central_test_optimality_gap = F(x) − F*`; it refuses a convex problem with
  no entry, and names the cause when the table certifies the same corpus and
  problem at another digest (rows built on another LAPACK, say) -- the rows
  are then not the ones `F*` was solved on;
- on a nonconvex problem, looks nothing up and reports no gap.

`python examples/fed-logistic-l1/certify.py --config <corpus config> --problem
<loss> <penalty> <λ>` makes an entry: it rebuilds the corpus from its
generator config, bit for bit what `fedbrew generate` writes on the same
machine, solves, and writes. A run config names another table with
`model.optima`. The solves:

- **L1.** FISTA at the step `1/L` identifies the support; Newton on the
  support, with its signs fixed, takes the residual to machine precision.
  The certificate is the residual on the full vector,
  `max_k |∇ℓ(x)_k + λ sign(x_k)|` on the support and `max(|∇ℓ(x)_k| − λ, 0)` off
  it, and `certify.py` refuses an entry above `1e-12`.

- **Squared L2.** Damped Newton from 0 on the Hessian
  `Aᵀdiag(s(1 − s))A/n + λI`, `s = σ(−b aᵀx)`, with a backtracking line search,
  until `max_k |∇F(x)_k|` is under `1e-12`.

`F*` is `F` at the stored `x*`, summed in the order the global shard stacks the
rows, which is the order the gap is computed in.

## The batched executor

The task is batchable (chapter 11 §9): `split_rows`, `row_batches`,
`functional_loss` and `functional_eval` are the arithmetic `train_step` and
`eval_step` run, so `runtime.performance.executor: batched` trains every
client of a round together. `tests/test_fed_logistic_l1.py` holds one FedAvg
round of each problem, batched, to the sequential run within the executor's
`1e-12`.

## Which shipped algorithms can solve it

**The squared-L2 problem: gradient arms**, at a small enough step: `F` is
smooth and strongly convex. The shipped FedAvg arm, one full-split step per
client a round at full participation, is gradient descent on `F` at step
`1/(L + λ)` and converges to `x*`; with more local steps, client drift moves
its fixed point, and the gap measures by how much.

**Logistic regression with a nonconvex regularizer, and the sigmoid loss: gradient arms reach a stationary point**,
not a certified minimum; with no `F*` there is no gap to read, and `F(x)` and
the distance to `x_true` are what a run reports.

**The L1 problem: none**, for fed-lasso's reason. There is no proximal
operator anywhere in fedbrew, so every arm runs subgradient descent on a
non-smooth objective: `exact_zeros` reads `d` at round 0 and 0 from round 1
on, and the support exists only at `support_tolerance`.

## Files

| Path | What it is |
| --- | --- |
| `problem.py` | the losses, penalties and their gradients, the generator, the reference solves, the task and the model, `register()` and `_self_check()` |
| `optima.json` | the certified optima of the convex problems, keyed by corpus digest, loss, penalty and `λ` |
| `certify.py` | certifies problems on a corpus and writes their entries into the table |
| `run.py` | runs one corpus's arms through `fedbrew run` and tables their final round |
| `../../data/configs/examples/fed-logistic-l1-*.yaml` | one generator config per corpus |
| `../../configs/examples/fed-logistic-l1-*/<loss>-<penalty>-lambda<λ>.yaml` | one FedAvg arm per problem and `λ` on the corpus, its step `1/L` and untuned |
