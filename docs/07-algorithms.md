# 07 — Algorithms

The nine server strategies, the nine client update rules, which pairings are
legal, what each communicates per round, and the config keys each requires.

A federated algorithm here is a **pair**: a server strategy and a client update
rule, chosen independently by `server.strategy` and `client.update_rule`. Some
pairings are meaningless and are refused at config load rather than run.

## 1. The two registries

`Registry` (`fedbrew/core/registry.py`).

```
server_strategies  fedavg fedavgm fedadam fedyogi fedadagrad fedopt
                   scaffold fedlalr centralized
client_updates     local_sgd fedavg centralized local_adamw fedprox
                   scaffold delta_sgd fedlalr fedavg_ft
```

Four names appear in both. They are separate objects: `server.strategy:
scaffold` selects the SCAFFOLD server, `client.update_rule: scaffold` selects
the SCAFFOLD client, and the config sets each.

## 2. Which pairings are enforced

Three algorithms need both halves, **in both directions**, and naming one half
without the other fails at config load rather than letting a half-configured
arm run.

| Pair | What the two halves exchange |
| --- | --- |
| `centralized` ⟷ `centralized` | the strategy pools the dataset into a single client; the rule supplies the local update applied to it |
| `scaffold` ⟷ `scaffold` | the server broadcasts a control variate the client corrects its step with; the client returns the control delta the server updates it from |
| `fedlalr` ⟷ `fedlalr` | the server synchronizes the momentum and second moment the client's local AMSGrad reads |

One table, `PAIRED_STRATEGIES` in `fedbrew/core/config.py`, read by
`_validate_paired_strategies` — which `validate_config` calls before it judges
any algorithm-specific option, because a half-paired config is wrong about
which algorithm is running and every later message would answer the wrong
question. `fedbrew/core/validation.py` reports the same three earlier, under
`--validate-only`; it does not gate, so it is not where the refusal lives.

Either half alone would silently change what runs, or fail in round 1. The
centralized baseline is the clearest of the first kind: configure only the
strategy and you get a pooled dataset trained by a federated rule; configure
only the rule and you get federated data trained as if pooled. SCAFFOLD is the
clearest of the second: the server reads a fit payload that has no
`control_delta` in it, after the data is staged and the first local iterations
are spent.

`fedprox`, `delta_sgd`, `local_adamw` and `fedavg_ft` are **client-side only**
— they pair with the plain `fedavg` server, which is what makes them
comparable to a FedAvg baseline.

**None of this reaches a component from outside the package.** Every rule here
names strategies and update rules the package ships, and a component loaded
through `experiment.extensions` (chapter 12) can appear in none of them — so a
pair including one is left unjudged rather than refused for missing from a list
it cannot be on. Preflight says so, as an info issue
(`algorithm.extension_pairing_unchecked`), and the extension is responsible for
refusing a partner it cannot work with. The cost is real and is the deliberate
one: an incompatible pair involving an extension fails when it is built, or in
round 1, instead of at preflight.

## 3. Server strategies

### 3.1 `fedavg`

The baseline. Weighted mean of client model states
(`fedbrew/servers/fedavg.py`):

```
w_{t+1} = sum_c (n_c * w_c) / sum_c n_c
```

`server.aggregation_weighting` chooses the weight *n_c*: `examples` (default)
uses the weight each client reports, `uniform` uses 1. **Metrics stay
example-weighted in both modes** — that key governs the parameter average
only, because a metric's weighting is what makes the reported number a
population mean. Chapter 08 §4.1.

**What a client reports.** *n_c* is `FitResult.num_examples`. Every rule but
`fedprox` and `scaffold` takes it from `TaskAdapter.federated_aggregation_weight`
(`fedbrew/tasks/base.py`); those two report the count of their post-fit
evaluation pass directly (`AGGREGATION_WEIGHT_HOOK_BYPASS_RULES`,
`fedbrew/core/federated_state.py`). That pass is `TorchSGDClient._evaluate_model` over
the client's whole train split, at `client.eval_batch_size`, with a loader that
never drops a batch. So *n_c* is a size of the train split, not a count of what
the round's local update touched:

| Task | *n_c* |
| --- | --- |
| `classification` | the rows in the client's train split, counted by the post-fit evaluation pass |
| `causal_lm`, `model.active_target_weighting: false` — the default unless the manifest's task is `causal_lm_sft` | the active target tokens (chapter 06 §3.1) in the client's train split, counted by the same pass |
| `causal_lm`, `model.active_target_weighting: true` — the default for a `causal_lm_sft` manifest | the active target tokens of every batch the round's training pass ran, summed over the round (`TorchCausalLMTask.federated_aggregation_weight`). Refused under `fedprox` and `scaffold`, which never ask the task: the key at config load, the `causal_lm_sft` default where the manifest is read (`factory._model_config`, and preflight's `data` check). The message names the rules that honour it. Set it to `false` to weight those two by the train split's tokens, which is what they do (FINDINGS.csv `POST-F30`) |
| an extension task | what its `federated_aggregation_weight` override returns; without one, the evaluation pass's count as for `classification`: the sum of `eval_step`'s `total`, or the split's size when not every step returns one (chapter 12 §6) |

On every row but the third, nothing about the local update moves *n_c*:

| What the round does | Rows the update reads | *n_c* |
| --- | --- | --- |
| `single_batch` | `local_iterations` batches | the train split |
| `sequential_epoch`, or an unset `update_mode` | every row, `local_iterations` times | the train split |
| `drop_last: true` | the split less its last partial batch, per pass | the train split |
| `max_local_steps` (`local_adamw`) | at most that many batches | the train split |
| `frozen_batch_gradients`, `full_gradient` | every row once per iteration | the train split |

On the third row the count follows the training pass instead: a batch read in
two iterations counts twice, a batch dropped by `drop_last` or never reached
under `max_local_steps` counts zero, `single_batch` counts `local_iterations`
batches, and `full_gradient` counts the whole split once per iteration
(`frozen_batch_gradients` is refused on `causal_lm`). Those are token
exposures, not distinct tokens.

This is FedAvg's `n_k`, a dataset size, and it is the convention FedBrew
keeps: no built-in task weights by examples trained, and classification
arms under different update modes weight their clients identically.
`tests/test_aggregation_weight_is_the_train_split_size.py` pins it through a
real run: under `single_batch`, a two-pass `sequential_epoch` with
`drop_last`, and `local_adamw` capped at one step, the weight the server
receives is the 12-row train split while the update read 5 or 20 rows. How much
work stands behind a weight is what `optimizer_steps` and
`active_target_tokens` report (chapter 08 §4.2), and why arms that differ in
update shape are not comparable at equal `local_iterations` (chapter 04 §2.1).

FedAvg is where example weighting comes from, so the default is the paper here.
SCAFFOLD, FedLALR and Δ-SGD are published with a uniform client average, and so
is the FedOpt family's Algorithm 2, although that paper's experiments weight by
example count; on those four preflight reports the difference at `info`:

```
info  algorithm.scaffold_aggregation_weighting
      SCAFFOLD as published averages clients uniformly, but this run weights
      the model by example count while the control variate stays uniform, so
      the drift correction is not the one the paper analyses
```

It is `info` and not a warning because example weighting is a comparability
choice, not a misconfiguration — but it is a choice, and the notice is the only
thing that says a run made it. SCAFFOLD's wording is the sharp one: its server
scales the summed control delta by `1/num_clients` whatever this key says, so
under `examples` the two halves of its update estimate different means.
`configs/femnist/scaffold.yaml` records why that arm keeps the default anyway;
`fedlalr.yaml` and `delta_sgd.yaml` set `uniform` and match their papers, so
the FEMNIST sweep is not uniform in either direction and the choice is made per
arm.

Streaming: each result is folded into a running sum and dropped, so peak memory
is a constant number of model states regardless of participation — **two model
states**, measured at 4, 16 and 64 clients by
`tests/test_aggregation_peak_memory.py`: the running sum and one reference
copy.

### 3.2 The FedOpt family

`fedopt`, plus four named aliases that set `server_optimizer` themselves.
FedAdagrad, FedAdam and FedYogi are Reddi et al.,
[arXiv:2003.00295](https://arxiv.org/abs/2003.00295). `fedavgm` is server
momentum, which that paper runs as a baseline and credits to Hsu et al. (2019).

All four compute the same pseudo-gradient, then apply a different server
optimizer to it:

```
delta_t = mean_c(w_c) - w_t            the aggregate client update
w_{t+1} = w_t + server_lr * update(delta_t)
```

| `server.strategy` | `update(delta)` |
| --- | --- |
| `fedavgm` | `m_t = beta1 * m_{t-1} + delta_t`, update is `m_t` — server momentum |
| `fedadagrad` | `v_t = v_{t-1} + delta_t^2`, update is `m_t / (sqrt(v_t) + tau)` |
| `fedadam` | `v_t = beta2 * v_{t-1} + (1-beta2) * delta_t^2`, update is `m_t / (sqrt(v_t) + tau)` |
| `fedyogi` | `v_t = v_{t-1} - (1-beta2) * delta_t^2 * sign(v_{t-1} - delta_t^2)` |

with `m_t = beta1 * m_{t-1} + (1 - beta1) * delta_t` for the latter three.

**`v_{-1}` is initialised to `tau^2`, not zero** — `FedOptServer._initial_v`
(`fedbrew/servers/fedopt.py`) — the smallest value Algorithm 2 of the paper
allows, since it requires `v_{-1} >= tau^2`: it keeps the `sqrt(v) + tau`
denominator away from pure `tau` on the first rounds. The effect is bounded:
from a zero start, the
first server step would differ by less than a factor of two, for all three.

Yogi differs from Adam only in how `v` moves toward `delta_t^2`. Adam moves it
by `(1-beta2)` of the gap between them; Yogi by `(1-beta2) * delta_t^2`, in the
direction the sign term gives. When `delta_t^2` is far below `v`, Adam's `v`
therefore shrinks by nearly `(1-beta2) * v` a round and Yogi's only by
`(1-beta2) * delta_t^2`, so Yogi's effective step grows more slowly as the
updates shrink.

Required keys: `server.server_learning_rate` and `beta1` for all four, plus
whichever of `beta2` and `tau` the chosen optimizer reads — and **only** those.
A hyperparameter the optimizer never reads is refused rather than accepted and
dropped, so a config cannot carry a placeholder that `run.json` then records as
if it had shaped the result:

| `server.strategy` | reads | refuses |
| --- | --- | --- |
| `fedavgm` | `server_learning_rate`, `beta1` | `beta2`, `tau` |
| `fedadagrad` | `server_learning_rate`, `beta1`, `tau` | `beta2` |
| `fedadam`, `fedyogi` | all four | — |

`fedavgm` keeps no second moment at all, so neither the decay nor the
denominator floor has anywhere to enter; `fedadagrad` keeps one and never
decays it, which is what makes it Adagrad. The table lives in
`UNREAD_FEDOPT_HYPERPARAMETERS` (`fedbrew/servers/fedopt.py`), beside the
updates that decide it, and `validate_config` reads it from there.

Each is bounded, and the bound is refused on the run path — `FEDOPT_BOUNDS` in
the same module, read by `validate_config`, by `FedOptServer.__init__`, and by
`--validate-only`:

| Key | Bound | Why that end is closed |
| --- | --- | --- |
| `server_learning_rate` | `> 0` | at zero the server never applies its update |
| `beta1`, `beta2` | `in [0, 1)` | at 1 the moment never takes up the delta, so the server never moves |
| `tau` | `> 0` | at zero `v_{-1} = tau²` is zero, so a coordinate whose delta is exactly zero computes `0 / (sqrt(0) + 0)` |

`tau: 0` is the one worth spelling out. An exactly-zero delta is what every
client returns for a parameter that receives no gradient, so one round produced
a NaN in the server's model state, which is permanent. The bounds used to live
only in the `--validate-only` preflight, which is not on the run path.

`server.server_optimizer` is required **only** for the bare `strategy: fedopt`
spelling and refused under the four aliases, each of which names its own
optimizer in the builder that registers it. No shipped config uses the bare
form, so that key is never exercised by a shipped arm. Chapter 04 §4 covers why
it is refused rather than ignored.

### 3.3 `scaffold`

Karimireddy et al. Server keeps a control variate `c` alongside the model:

```
c <- c + (1/N) * sum_c (c_i_new - c_i_old)
```

where `N` is the total client count, not the participating count. Emits
`server_control_norm` and `mean_client_control_delta_norm` — and those two are
added **after** `filter_metrics`, so they appear regardless of
`server.metrics`, as every server's diagnostics do. Chapter 08 §7.2.

The control deltas are summed beside the model, not through
`WeightedStateAccumulator`, so they are checked on their own: each client's
`control_delta`, their sum and the updated `c` go through the same
`refuse_non_finite_state` (`fedbrew/core/torch_utils.py`) the accumulator
applies to model states, and the model and `c` are assigned together only
after all three pass. A NaN or Inf anywhere refuses the round with both
unchanged, and the loop records it as the `non_finite_client_state`
divergence a non-finite model produces, so the last checkpoint is the last
healthy one. Before this, a finite model beside a NaN delta aggregated and
left NaN in `c` for the rest of the run (FINDINGS.csv `POST-F27`).

### 3.4 `fedlalr`

Locally adaptive rates with synchronised optimizer state,
[arXiv:2309.09719](https://arxiv.org/abs/2309.09719). The server aggregates
the model, the momentum and the second moment, and broadcasts all three.

That is why it moves **3× a FedAvg arm** per round. Preflight says so —
`fedbrew run --validate-only` prints `algorithm.fedlalr_communication_cost` —
and `tests/test_client_communication_cost.py` guards the multiplier.

**The server reads `client.epsilon`** to seed `v̂₋₁ = ε²`
in `_build_server` (`fedbrew/core/factory.py`) — the same key the client uses as
its Adam epsilon, with
the same implicit default `1e-8`. One key, two consumers.

The paper aggregates uniformly rather than by example count; preflight warns
when the config does otherwise.

### 3.5 `centralized`

Runs the **FedAvg server and client unchanged** over a single pooled client
(`fedbrew/data/centralized_dataset.py`). Averaging one result is the identity,
so the baseline differs from a FedAvg run only in how the data is partitioned —
the local-update modes cannot drift apart, which is the point.
`tests/test_centralized_equivalence.py` guards it.

## 4. Client update rules

### 4.1 `local_sgd`, `fedavg`, `centralized`

The shared SGD engine. `fedavg` and `centralized` additionally require
`update_mode` and `frozen_gradient_weighting`.

| `client.update_mode` | One local iteration |
| --- | --- |
| `sequential_epoch` | one pass of ordinary SGD over the client's batches, one step per batch |
| `single_batch` | one mini-batch step |
| `frozen_batch_gradients` | one pass whose batch gradients are all computed at the iteration-start model, then combined into one update |
| `full_gradient` | one step on the exact gradient of the task's training loss over the whole train split, computed batch by batch at the iteration-start model |

`defaults.local_iterations` counts those iterations, so a round is
`local_iterations` steps under `single_batch`, `frozen_batch_gradients` and
`full_gradient`, and `local_iterations` × the client's batch count under
`sequential_epoch` (chapter 04 §2.1).

`frozen_gradient_weighting` (`examples`, `uniform`, `sum`) decides how those
frozen gradients combine. `full_gradient` reads no weighting: each batch counts
by its share of what the task's loss averages over, so the step is the same at
any `batch_size` and in any order, and it equals `frozen_batch_gradients` under
`examples` exactly where the loss is a mean over examples. Everywhere else the
frozen mode is refused: at load on `causal_lm`, and in the engine for any task
that overrides `train_loss_denominator` (chapter 04 §2.1, FINDINGS.csv
`POST-F19`). `delta_sgd` takes `full_gradient` too, and its step-size rule then
sees the exact gradient. Its step size resets to `eta_0` every round, so a round
of one local step is FedAvg at learning rate `eta_0`; it adapts from a round's
second step.

**`max_grad_norm` does not mean the same thing in all four.** It bounds the
gradient each applied update uses, and the modes apply different numbers of
updates per epoch:

| `update_mode` | What is clipped | How often | Epoch moves at most |
| --- | --- | --- | --- |
| `single_batch`, `sequential_epoch` | each batch gradient | once per step | `N × lr × max_grad_norm` |
| `frozen_batch_gradients`, `full_gradient` | the combined epoch gradient | once per epoch | `lr × max_grad_norm` |

So two arms carrying the same `max_grad_norm` under different `update_mode`s
are not on a common bound — they differ by the batch count `N`, and under
`frozen_gradient_weighting: sum` the pre-clip quantity is `N` times larger
again, so the clip engages far sooner.
`tests/test_grad_clipping_is_per_mode.py` measures the difference. Per mode the
behaviour is coherent — one clip per applied update either way.

**`frozen_batch_gradients` and `full_gradient` do not run under
`runtime.use_amp: true`**, refused at config load and at preflight
(`algorithm.sgd_engine_amp_unsupported`). This refusal is per setting, not per
rule: `fedavg` under AMP is fine until it is asked to combine its gradients by
hand.

`max_grad_norm` does run. The two settings look alike — each makes the local
step pass `task.train_step` something other than a plain optimizer — and
§4.4's measurement is what separates them:

| Setting | Wrapper the step passes | `param_groups`? | Under AMP |
| --- | --- | --- | --- |
| `frozen_batch_gradients`, `full_gradient` | `_GradientOnlyOptimizer` | no | **Refused.** `GradScaler.unscale_` raises `AttributeError` |
| `max_grad_norm` | `_ClippingOptimizer` | yes | **Runs.** 3.97e-05 over five rounds, against a zero noise floor |

`_GradientOnlyOptimizer` wraps no optimizer — it computes gradients so the
frozen mode can combine them by hand — so it has no groups to expose, and
`GradScaler` has nothing to unscale. The same fact refuses `fedlalr` and
`delta_sgd`. `_ClippingOptimizer` wraps a real one and hands its groups
through, which is all `GradScaler` asks for.

Both were refused until the difference was measured. What is left is a
mechanism, not a policy: the refusal covers exactly the wrapper that cannot
be scaled.

### 4.2 `local_adamw`

Local AdamW instead of SGD. Optimizer state is per-client and does **not**
federate; `tests/test_adamw_and_model_state_compat.py` guards that the model
state stays loadable regardless.

### 4.3 `fedprox`

Adds a proximal term pulling the local model toward the round-start weights:

```
L_total = L_task + (mu/2) * ||w_local - w_round_start||^2
```

Requires `client.proximal_mu`. Emits `fit_proximal_loss` and `fit_total_loss`,
so you can see how much of the objective the proximal term accounts for.

**Communication is unchanged** — 1× a FedAvg arm. FedProx costs compute, not
bandwidth.

The proximal term is added as a closed-form gradient, `mu * (w - w0)`, inside
`_FedProxCorrectingOptimizer.step` (`fedbrew/clients/torch_fedprox_client.py`) --
wrapping a real optimizer and correcting `.grad` right before delegating to
it, so `fit()` calls `task.train_step(model, batch, optimizer)` directly
instead of reimplementing forward/loss/backward against two of the task's
own private attributes. `local_update_modes.py`'s `_ClippingOptimizer` does
the same thing for gradient clipping.

**Composes with `runtime.use_amp: true`**, and did not until the composition
was measured. `GradScaler.step` unscales `.grad` before delegating to a wrapped
optimizer, so the proximal term is added to true-scale gradients. §4.4 carries
the measurement for both this rule and SCAFFOLD.

### 4.4 `scaffold`

Corrects each local gradient by `(c - c_i)` inside
`_ScaffoldCorrectingOptimizer.step` (`fedbrew/clients/torch_scaffold_client.py`) --
wrapping a real optimizer and correcting `.grad` right before delegating to
it, the same pattern §4.3 uses for FedProx's proximal term and
`local_update_modes.py`'s `_ClippingOptimizer` uses for gradient clipping --
then uploads the model **and** the control-variate delta together.

That is **2× a FedAvg arm** per direction per round. The client adds the delta's
bytes into its own `communicated_bytes`, so that metric is already the true
upload volume.

Emits `control_delta_norm`, `client_control_norm`, `local_steps`.

**Composes with `runtime.use_amp: true`**, on the same mechanism as FedProx
in §4.3. Both refused it until now, in three layers, for a reason that read
"standard AMP usage and should work, but nothing in this repository has
measured it".

Measured on an A100, on 2026-09-07. Two things came out of it.

The guard was **stricter than the requirement**. `GradScaler.unscale_` reads
the optimizer's `param_groups`; `train_step` tested the nominal
`torch.optim.Optimizer` type. One AMP training step per wrapper, with that
guard lifted:

| Wrapper | Under AMP |
| --- | --- |
| `torch.optim.SGD` (control) | ok |
| `_ClippingOptimizer` | ok |
| `_ScaffoldCorrectingOptimizer` | ok |
| `_FedProxCorrectingOptimizer` | ok |
| `_GradientOnlyOptimizer` | `AttributeError`: no `param_groups` |

And the **numbers agree**. Five rounds on the synthetic label-skew manifest,
same seed and data, comparing `central_test_loss` round by round. The fp32 arm
was run twice first, so the comparison has a noise floor to clear:

| Arm | fp32 repeat | fp32 vs AMP | relative |
| --- | --- | --- | --- |
| `scaffold` | 0.0 | 6.13e-05 | 5.1e-05 |
| `fedprox` | 0.0 | 6.13e-05 | 5.1e-05 |
| `fedavg` + `max_grad_norm` | 0.0 | 3.97e-05 | 3.3e-05 |

The noise floor is zero — the same config run twice reproduces bit-identically
— so those differences are AMP's, and 5e-05 relative is two orders inside
fp16's own precision. A run that completes was never the claim; a run that
agrees is.

`scaffold` and `fedprox` land on the same delta, to three significant figures.
That coincidence was not investigated: what the table records is that each arm
agrees with its own fp32 run, not why two different corrections move the
trajectory by the same amount on this fixture.

`fedlalr` (§4.6) and `delta_sgd` (§4.5) keep their refusal, and so do
`update_mode: frozen_batch_gradients` and `full_gradient` (§4.1). All four step through
`_GradientOnlyOptimizer`, which wraps no optimizer and exposes no
`param_groups`. That refusal is a measurement now too.

`max_grad_norm` composes — its row is in the table — and its refusal was
lifted with the other two. Every AMP refusal left in the tree names the same
mechanism, and none of them names a decision.

### 4.5 `delta_sgd`

Auto-tuned per-step size, [arXiv:2306.11201](https://arxiv.org/abs/2306.11201).
The server is plain FedAvg, so the whole method is client-side.

| Key | Default | Paper |
| --- | --- | --- |
| `eta_0` | required | initial step |
| `theta_0` | `1.0` | used unchanged across every experiment in the paper |
| `gamma` | `2.0` | same |
| `delta` | `0.1` | same |
| `eta_max` | `null` — no clamp | an unclamped adaptive step is a different algorithm from a clamped one |

It takes all four `update_mode`s (chapter 04 §2.1), and runs `sequential_epoch`
when `update_mode` is unset.

The step-size trace is the algorithm, so it is emitted every round:
`client_step_size_{mean,min,max,final}`, `step_size_clamp_fraction`,
`undefined_curvature_fraction`. A run whose mean step never leaves `eta_0` is
one where the auto-tuner did nothing.

**Incompatible with `runtime.use_amp: true`** and refused at load: the rule
reads the raw gradient off `.grad`, which the AMP path consumes inside
`GradScaler.step` instead of leaving there.

### 4.6 `fedlalr`

The client half of §3.4. Requires `beta1`, `beta2`, `epsilon`. Uploads model,
momentum and second moment.

Emits `effective_learning_rate_coordinate_{mean,min,max}` — statistics of the
per-coordinate rate `alpha / sqrt(v_hat)` **within** one client. The server
emits `effective_learning_rate_across_clients_{mean,std,min,max}`, the spread
**across** clients of each client's coordinate mean. The two families once
shared the names `client_effective_learning_rate_*`, which are now refused.
Chapter 08 §7.3 defines each column.

### 4.7 `fedavg_ft`

FedAvg plus per-client fine-tuning at evaluation time — the personalization
baseline. Requires `client.finetune_epochs`;
`client.finetune_learning_rate` defaults to `client.learning_rate`.

It is the only rule that produces a *personal* model, so it is the only one
that can be evaluated under `evaluation.model_scope: personal` or `both`. A
rule with no personal model **rejects** a non-global scope rather than
silently evaluating the global one under a personalized name.

The fine-tuning happens **after** the global pass, deliberately: with
`reuse_model` on, `build_model` returns one cached module, so fine-tuning
mutates the same object the global pass read. Measuring global first keeps both
numbers honest and allocates no second model.

**The fine-tuning pass draws its own batch order**, under the dataloader phase
`finetune` (chapter 10 §1). It used to run under `fit`, which is the same phase
the fit pass uses, so both took one seed and one order. Both passes also start
from the same weights — the round's global model — so with a `sequential_epoch`
fit pass, where both counts are passes, `finetune_epochs ≤ local_iterations`
and an inherited learning rate, the personal model was the fit pass's first
epochs recomputed rather than an adaptation
drawn independently of them. Only the dropout masks differed, and only because
`evaluate()` runs inside `isolated_evaluation_rng`.

This changes what a `fedavg_ft` run computes.

## 5. Communication cost per round

Model-shaped states moved per direction per round, relative to FedAvg.

| Algorithm | Multiplier | What moves |
| --- | --- | --- |
| `fedavg`, `fedprox`, `delta_sgd`, `local_adamw`, `fedavg_ft` | **1×** | the model |
| the FedOpt family | **1×** | the model; optimizer state stays server-side |
| `scaffold` | **2×** | model + control-variate delta |
| `fedlalr` | **3×** | model + momentum + second moment |
| LoRA, on the rules §5.1 lists | ≪1× | adapter tensors only — chapter 06 §3.4 |

**Compare arms on `communicated_bytes`, not on round count alone.** Preflight
says so for SCAFFOLD and FedLALR. A 100-round SCAFFOLD arm has moved as much as
a 200-round FedAvg arm.

The multipliers are measured rather than asserted, because an asserted one can
be wrong in the direction nothing contradicts. A client that uploads a second
model-sized state while its meter counts one reports half its real volume, in
the right units and at the right order of magnitude. So
`tests/test_client_communication_cost.py` runs every rule's `fit` and checks the
meter against every model-shaped state in the payload it returns — structurally,
so a state added under any name is counted without anyone remembering to add a
row here.

### 5.1 Which rules train adapter (LoRA) state

A rule trains adapter-only state when it loads, extracts and weights the model
through the task's hooks (`TaskAdapter.load_federated_model_state` and the
rest), so an adapter model moves only its adapter. Each combination below
trains a real adapter-only round on a tiny GPT-2 LoRA model in
`tests/test_adapter_state_support.py`, and the adapter moves:

| `client.update_rule` | `server.strategy` |
| --- | --- |
| `fedavg`, `fedavg_ft`, `local_sgd`, `local_adamw`, `delta_sgd` | `fedavg`, `fedopt`, `fedadam`, `fedyogi`, `fedadagrad` |
| `centralized` | `centralized` |

Three rules cannot, and are refused with an adapter-scoped model
(`ADAPTER_SCOPED_MODELS`, `fedbrew/core/config.py`: `hf_causal_lm_lora`) at
config load, by `fedbrew run` and `--validate-only` alike, before anything is
built. The client refuses too, before its first update, when the task reports
adapter scope for a model load could not judge, an extension's included. Both
messages name the rule and the rules that can (`FULL_STATE_ONLY_CLIENT_RULES`,
`fedbrew/core/federated_state.py`):

| Rule | Why not |
| --- | --- |
| `fedprox` | loads and returns the whole model's `state_dict`, not the task's federated state, so an adapter-only broadcast does not load |
| `scaffold` | the same, and keys its control variates by that `state_dict` |
| `fedlalr` | its local AMSGrad looks its moments up by the model's parameter names, which under a PEFT adapter carry the adapter name the federated state's keys do not |

Chapter 07 said "any rule" until all three were measured failing in round 1,
after the model was built (FINDINGS.csv `POST-F29`).

## 6. Options a rule refuses

`_training_client_kwargs` hands each of nine engine keys only to the rules
whose local step reads it (`ENGINE_CLIENT_OPTIONS`):

```
momentum  weight_decay  nesterov  learning_rate_schedule  min_learning_rate
update_mode  frozen_gradient_weighting  max_local_steps  max_grad_norm
```

A key its rule never receives is **rejected** rather than ignored, per rule,
by `validate_config` against `UNHONOURED_CLIENT_OPTIONS`:

| Rule | Local step | Rejected keys |
| --- | --- | --- |
| `fedprox` | plain SGD, proximal term in the backward pass; `update_mode` `sequential_epoch` (unset) or `full_gradient` | `momentum` `weight_decay` `nesterov` `learning_rate_schedule` `min_learning_rate` `frozen_gradient_weighting` `max_local_steps` `max_grad_norm` |
| `scaffold` | plain SGD, `(c - c_i)` written into the step; `update_mode` `sequential_epoch` (unset) or `full_gradient` | `momentum` `weight_decay` `nesterov` `learning_rate_schedule` `min_learning_rate` `frozen_gradient_weighting` `max_local_steps` `max_grad_norm` |
| `fedlalr` | local AMSGrad at a per-coordinate rate; `update_mode` `sequential_epoch` (unset) or `full_gradient` | `momentum` `weight_decay` `nesterov` `learning_rate_schedule` `min_learning_rate` `frozen_gradient_weighting` `max_local_steps` `max_grad_norm` |
| `delta_sgd` | a step size measured from the local smoothness; any `update_mode`, `sequential_epoch` (unset) | `momentum` `weight_decay` `nesterov` `learning_rate_schedule` `min_learning_rate` `max_local_steps` |
| `local_adamw` | `torch.optim.AdamW`; `update_mode` `sequential_epoch` (unset) or `full_gradient` | `momentum` `nesterov` `frozen_gradient_weighting` `max_grad_norm` |
| `local_sgd` | the base SGD client; `update_mode` `sequential_epoch` (unset) or `full_gradient` | `frozen_gradient_weighting` `max_local_steps` `max_grad_norm` |
| `fedavg`, `centralized`, `fedavg_ft` | the shared local-update engine | none — all nine reach the client |

Six of these rules also check the options in `__init__`. That check is
reachable by direct construction only: it tests an attribute the factory sets
only for the rules that honour the option, so for a rule that does not, it is
the base class's default and never fires. `delta_sgd` with `momentum: 0.9` and
`fedlalr` with `max_grad_norm: 10.0` both used to load, train and be recorded
in `run.json` as configured — the second one unclipped.

`tests/test_ignored_client_options.py` guards this, deriving the third column
from the factory's own gates. Chapter 04 §5.3.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/core/registry.py` | `Registry` — which strategies and rules exist |
| `fedbrew/servers/fedavg.py` | the baseline, and `WeightedMetricAccumulator` |
| `fedbrew/servers/fedopt.py` | the four server optimizers and `v_{-1} = tau^2` |
| `fedbrew/servers/scaffold.py` | the server control variate |
| `fedbrew/servers/fedlalr.py` | synchronised optimizer state |
| `fedbrew/clients/torch_sgd_client.py` | the shared SGD engine |
| `fedbrew/clients/local_update_modes.py` | the three `update_mode` variants |
| `fedbrew/clients/torch_fedprox_client.py` | the proximal term |
| `fedbrew/clients/torch_scaffold_client.py` | the `(c - c_i)` correction |
| `fedbrew/clients/torch_delta_sgd_client.py` | the step-size schedule and its trace |
| `fedbrew/clients/torch_fedlalr_client.py` | per-coordinate rates |
| `fedbrew/clients/fedavg_ft_client.py` | fine-tuning, the ordering that keeps both passes honest, and the dataloader phase that keeps their batch orders apart |
| `fedbrew/core/config.py` | `PAIRED_STRATEGIES` and `_validate_paired_strategies` — the enforced pairings |
| `fedbrew/core/validation.py` | `_validate_shipped_algorithm_compatibility` — the SCAFFOLD pairing, and the cost notices |

### Commands

```bash
# Prove the strategy lists, pairings and cost multipliers here are current.
python -m pytest tests/test_docs_algorithms.py -v

# The algorithm-fidelity guards.
python -m pytest tests/test_aggregation_correctness.py \
                 tests/test_aggregation_properties.py \
                 tests/test_aggregation_weighting.py \
                 tests/test_fedopt_server.py \
                 tests/test_scaffold_fedprox_communication_cost.py \
                 tests/test_client_communication_cost.py \
                 tests/test_fedlalr.py \
                 tests/test_delta_sgd.py \
                 tests/test_fedavg_ft.py \
                 tests/test_centralized_equivalence.py

# Check a pairing without training.
fedbrew run --config configs/femnist/scaffold.yaml --validate-only
```

### Invariants

1. **A strategy and a rule are chosen independently.** Three pairings are
   enforced; do not assume a name selects both halves.
2. **Aggregation stays streaming.** Two model states in memory regardless of
   participation, never one per participant.
3. **`aggregation_weighting` governs parameters only.** Metrics stay
   example-weighted in both modes.
4. **`v_{-1} = tau^2`, never zero**, for the three adaptive FedOpt variants.
5. **SCAFFOLD is 2×, FedLALR is 3×.** Both are asserted; a change to what
   either uploads must update the test and the preflight notice.
6. **`communicated_bytes` already includes auxiliary state.** SCAFFOLD and
   FedLALR add theirs at the client, so the metric is the true upload volume.
7. **A rule with no personal model rejects a non-global `model_scope`**
   rather than evaluating the global model under a personalized name.
8. **AMP needs an optimizer exposing `param_groups`, and nothing more.**
   `delta_sgd`, `fedlalr`, `update_mode: frozen_batch_gradients` and
   `update_mode: full_gradient` are refused under `use_amp` because they step
   through `_GradientOnlyOptimizer`,
   which exposes none. `scaffold`, `fedprox` and `max_grad_norm` were refused
   too, on a guard stricter than that requirement, until §4.4's measurement
   lifted all three. No AMP refusal survives that is not this one fact.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_algorithms.py` | The registry lists, enforced pairings and cost multipliers here match the code. |
| `tests/test_aggregation_correctness.py` | The weighted mean is the weighted mean. |
| `tests/test_aggregation_peak_memory.py` | §3.1's two model states, measured, and unchanged by participation. |
| `tests/test_aggregation_properties.py` | Permutation invariance, weight scaling, single-client identity. |
| `tests/test_aggregation_weighting.py` | `uniform` changes parameters, not metrics. |
| `tests/test_active_target_weighting_is_honoured_or_refused.py` | §3.1: which rules ask the task for the weight, measured by running each client's `fit`; `active_target_weighting` on is refused under `fedprox` and `scaffold` through `fedbrew run` and `--validate-only`, and the `causal_lm_sft` default where each path reads the manifest; `false`, unset and an honouring rule still load (`POST-F30`). |
| `tests/test_aggregation_weight_is_the_train_split_size.py` | §3.1: a classification client's weight, as the server receives it, is its train split's size under `single_batch`, a capped `local_adamw` and a two-pass `drop_last` `sequential_epoch`, none of which reads that many rows. |
| `tests/test_fedopt_server.py` | The four update rules and the `tau^2` initialisation. |
| `tests/test_scaffold_fedprox_communication_cost.py` | SCAFFOLD 2×, FedProx 1×. |
| `tests/test_amp_composes_with_wrapped_optimizers.py` | §4.4: scaffold and fedprox accept `runtime.use_amp: true` in all three layers, the three rules that step through `_GradientOnlyOptimizer` still refuse it, and the chapter quotes the numbers behind both. |
| `tests/test_scaffold_fedprox_step_correctness.py` | What the gradient correction and the proximal term actually compute, against a closed-form prediction and a pinned trajectory. |
| `tests/test_client_communication_cost.py` | Every rule's `communicated_bytes` equals every model-shaped state in the payload it returns, and section 5's multipliers are measured. |
| `tests/test_communication_cost_metrics.py` | A client's cost metrics survive `client.metrics` into the CSV. |
| `tests/test_fedlalr.py` | Synchronised optimizer state and the per-client rates. |
| `tests/test_delta_sgd.py` | The step-size schedule and its clamp. |
| `tests/test_fedavg_ft.py` | Fine-tuning, and the global-then-personal ordering. |
| `tests/test_finetune_draws_its_own_order.py` | §4.7: the fine-tuning pass draws a batch order the fit pass did not, in every round, and both stay reproducible. |
| `tests/test_centralized_equivalence.py` | The centralized baseline matches pooled training. |
| `tests/test_auxiliary_state_aggregation.py` | Control variates and moments survive aggregation. |
| `tests/test_scaffold_control_state_is_finite.py` | §3.3: a NaN or Inf control delta, or finite deltas that overflow `c`, refuse the round with the model and `c` bit-for-bit unchanged; the run is recorded as diverged and resumes from its last checkpoint (`POST-F27`). |
| `tests/test_adapter_state_support.py` | §5.1: every listed combination trains an adapter-only round that moves the adapter, and `fedprox`, `scaffold` and `fedlalr` are refused on an adapter model at load, through `fedbrew run` and `--validate-only`, and by the client before any update (`POST-F29`). |
| `tests/test_frozen_gradient_weighting.py` | The three weighting modes, and the mode refused on a task whose loss is not an example mean, at load and in both engines (`POST-F19`); `run_sgd_update_mode` asks for a weighting only under the mode that reads it (`POST-F21`). |
| `tests/test_full_gradient.py` | `full_gradient` is one step on the gradient of one batch holding the whole split, for classification, fed-lasso and the causal LM; the same at every batch size and order; equal to `frozen_batch_gradients` + `examples` where that is exact, and the frozen mode refused on the token-mean loss (`POST-F19`); refused under `drop_last` and under AMP. For every rule with a loop of its own, and for `delta_sgd`, it equals that rule's own pass over one batch holding the whole split. |
| `tests/test_ignored_client_options.py` | Which of the nine engine keys each rule refuses, derived from the factory. |

### Known failure modes

- **Setting only half of a paired algorithm.** Three are enforced; the message
  names the missing half.
- **Comparing SCAFFOLD to FedAvg at equal round counts.** SCAFFOLD has moved
  twice the bytes. Compare on `communicated_bytes`.
- **Expecting `server.metrics` to control a server's own diagnostics.** Both
  servers that emit them — SCAFFOLD and FedLALR — add them after the filter,
  so they always appear. `server.metrics` selects among the aggregated
  *client* metrics only.
- **Setting `runtime.use_amp: true` with `delta_sgd` or `fedlalr`.** Refused,
  per rule, at config load: both step through a wrapper with no `param_groups`
  for `GradScaler` to read. `scaffold` and `fedprox` used to be on this list
  and are not any more — §4.4.
- **Setting an engine key on a rule that does not receive it.** Refused, per
  rule, by the §6 table — `momentum` on `fedprox`, `max_grad_norm` on
  `fedlalr`, a schedule on `delta_sgd`. The rule's own `__init__` check does
  not cover this; the config validator does.
- **Using `model_scope: personal` with a non-personalizing rule.** Rejected;
  only `fedavg_ft` produces a personal model.
- **Reading a round's `effective_learning_rate_coordinate_min` as the smallest
  rate any client used.** It is the example-weighted mean of the clients'
  minima; the smallest client mean is `effective_learning_rate_across_clients_min`.
