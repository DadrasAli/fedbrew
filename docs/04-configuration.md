# 04 — Configuration

Every key a run config may carry: its type, its default, whether that default
is written down anywhere, and what rejects a wrong one.

Derived from `fedbrew/core/config.py` and the factory that reads it. Where a
default is produced somewhere other than the dataclass — a builder, a client
constructor, a `.get()` call — that location is named.

**The schema is strict and not stable.** Every key below is validated, and a
key nothing reads is refused, but this is version 0.0.1: a key can be renamed,
removed or changed in meaning between versions. A config is written against one
release, and `run.json` records the resolved config a run actually used.

## 1. How a config is read

```
configs/<arm>.yaml
  -> yaml.safe_load
  -> _reject_restated_keys        removed keys fail by name, with the reason
  -> dataclass split (_split_extra)  named fields bind; everything else -> extra
  -> _validate_known_keys         a key in extra that no reader consumes fails
  -> validate_config              types, ranges, enums, and cross-key rules
  -> apply_cli_overrides          --lr, --rounds, ... then validate_config again
  -> FullConfig
```

Two properties follow from that shape and are worth holding onto.

**A key that no reader consumes is an error, not a no-op.** Each section
declares its accepted `extra` keys in `_KNOWN_EXTRA_KEYS`
(`fedbrew/core/config.py`), and anything else fails at load:

```
unknown configuration key: client.max_grad_nrom. This section accepts:
accumulator, beta1, ... A key nothing reads is silently dropped and the
reader takes its default, so a misspelling changes the run without
appearing anywhere in run.json.
```

That message is the whole rationale. `max_grad_nrom: 1.0` used to sit in
`extra` unread while the reader returned `None`, and the run trained without
gradient clipping under a config that said otherwise.

**What every run checks, and what only `--validate-only` checks.** They are
two different things:

| | Every `fedbrew run` | `fedbrew run --validate-only` |
| --- | --- | --- |
| What runs | `load_config` → `validate_config`, again after CLI overrides; then `build_components`, whose builders refuse what they cannot build; then the loop's own refusals (resume, non-finite state, state compatibility) | `load_config` → `validate_config`, then the ten-area preflight `validation.run_checks`: `components`, `experiment`, `server`, `client`, `task`, `data`, `model`, `evaluation`, `runtime`, `algorithm compatibility` |
| On a problem | stops at the first one; a refusal prints RUN REFUSED and exits 2 | runs every check, lists every error, warning and note, exits 1 on any error |
| Trains | yes | no |

An ordinary run never calls the preflight, so what only preflight reports —
its warnings and notes (a non-empty output directory, an unchecked extension
pairing, the aggregation-weighting notice), and the full structural manifest
validation of its `data` area (chapter 05; loading a dataset makes narrower
checks of its own) — is not checked before training unless you run
`--validate-only` first. Everything this chapter calls *refused at load*
is refused on both paths. `tests/test_preflight_runs_only_under_validate_only.py`
holds the split: an ordinary run calls `validate_config` and not `run_checks`,
and `--validate-only` calls both and trains nothing. `--validate-only` is the
cheapest way to check a new config, and worth running before any long job.

## 2. The blocks

| Block | Required | What it decides |
| --- | --- | --- |
| `experiment` | yes | identity, seed, output location |
| `defaults` | yes | the round and local-iteration counts, shared by server and clients — §2.1 |
| `server` | yes | strategy, participation, metric filter |
| `client` | yes | update rule, local optimisation, metric filter |
| `data` | yes | which dataset, and where |
| `model` | yes | architecture and its per-builder keys — chapter 06 |
| `runtime` | yes | device, determinism, performance, checkpointing, staging |
| `evaluation` | no | per-split schedules and client scope |
| `client_statistics` | no | which cross-client columns are written |
| `divergence` | no | when to stop a run that is not learning |

The last three have complete dataclass defaults, so a config may omit them
entirely. The first six have required fields and cannot be omitted.

The root is closed like every block. A top-level key that is none of these
ten, and not `server_config` or `client_config` below, is refused at load,
before anything is built: the message names the key, suggests the closest
block, and lists every key the root accepts (`root_config_keys`,
`fedbrew/core/config.py`). Before this, `evaluaton:` loaded and the run took
every evaluation default, and `divergance:` ran with no absolute blow-up
threshold (FINDINGS.csv `POST-F26`). `server_config: <path>` and
`client_config: <path>` are the older spelling of the server and client
blocks: a YAML file holding the block, resolved against the config's own
directory, in place of the block itself.

There is no `task` block and no `task` key. The task is the one the model was
registered with (`models.register(..., task=)`, read through `MODEL_TASKS`);
writing a top-level `task:` or `experiment.task` is refused at load. §10.

### 2.1 `defaults`

Two keys, both required, and the only supported place to set either. The
block is closed like every other: any other key is refused at load, naming
the two it accepts (`DEFAULTS_KEYS`, `fedbrew/core/config.py`).

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `global_rounds` | int | **required** | Number of federated rounds. Written into `server.global_rounds` at load. |
| `local_iterations` | int | **required** | Iterations of the local loop per selected client per round; what one iteration is depends on `update_mode`, below. Written into `client.local_iterations` at load. |

The block exists because both numbers are read by more than one component —
the round count by the server and by every schedule that counts rounds, the
iteration count by every client — and a value with two spellings drifts. The
resolved config carries them as `server.global_rounds` and
`client.local_iterations`; a config that writes either of those names is refused
and told to come here (`_resolve_schedule_defaults`, `fedbrew/core/config.py`).

**What one iteration is.** `local_iterations` counts iterations of the local
loop; the rule's `update_mode` (§5) decides what one iteration does. With
K = `local_iterations` and B = the client's number of training batches:

| `update_mode` | One iteration | Parameter updates per client per round |
| --- | --- | ---: |
| `single_batch` | one optimizer step on the next mini-batch; the loader starts over when it runs out | K |
| `sequential_epoch` | one pass over the client's train split, one optimizer step per batch | K × B |
| `frozen_batch_gradients` | one pass whose batch gradients are all evaluated at the iteration's starting point, then combined (`frozen_gradient_weighting`), clipped once and applied as one update | K |
| `full_gradient` | one optimizer step on the exact gradient of the task's training loss over the whole train split: the gradient of one batch holding every sample, computed batch by batch at the iteration's starting point and clipped once. `batch_size` sets only how much is in memory at once | K |

`fedavg`, `centralized`, `fedavg_ft` and `delta_sgd` take every `update_mode`.
The first three require it; `delta_sgd` runs `sequential_epoch` when it is
unset (`_training_client_kwargs`, `fedbrew/core/factory.py`).

`full_gradient` weights each batch's gradient by its share of the count the
task's loss is a mean over (`TaskAdapter.train_loss_denominator`, chapter 12
§6): examples for classification and the shipped examples, active target
tokens for the causal-LM task. That is what makes it exact on both.
`client.drop_last: true` is refused under it, since it would leave samples out,
and so is `runtime.use_amp: true`.

`frozen_batch_gradients` weights by examples, equally or by sum whatever the
task, so it is refused on a task whose loss averages over anything else: at
load for the built-in tasks in `NON_EXAMPLE_MEAN_TASKS` (`causal_lm`), and for
any task whose class overrides `train_loss_denominator`, an extension task
among them, before its first update. On the causal-LM loss it was 53.5% from
the gradient of the pass (FINDINGS.csv `POST-F19`).
`fedprox`, `scaffold`, `fedlalr`, `local_sgd` and `local_adamw` run their own
loop, and take two modes: `sequential_epoch`, one pass with one step of the
rule's update per batch, which is also what an unset `update_mode` means; and
`full_gradient`, one step of that update per iteration on the exact gradient of
the whole split. `local_adamw`'s steps are further capped by `max_local_steps`
when it is set, under either mode.

So one `local_iterations` value is not one amount of local work: arms that
differ in update shape — `single_batch`, `frozen_batch_gradients` or
`full_gradient` against `sequential_epoch` or an unset `update_mode` —
are not comparable at equal `local_iterations` (FINDINGS.csv `POST-F15`, open
by decision). B also
differs between clients of unequal size, so under `sequential_epoch` the
clients of one run differ from each other too.

The key was called `local_epochs` until it was renamed, because "epochs" was
wrong under `single_batch` and "updates" is wrong under `sequential_epoch`. A
config that still writes `defaults.local_epochs` or `client.local_epochs` is
refused (§10), and so is a checkpoint whose client states carry
`local_epochs` (chapter 09 §5).

## 3. `experiment`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `seed` | int | **required** | Chapter 10 covers what it does and does not fix. |
| `output_dir` | str | **required** | Artifacts are written here. Empty or whitespace-only is refused at load: `Path("")` is `Path(".")`, so the run would write wherever it was started. `"."` is accepted — that is a choice. |
| `name` | str | `""` | Free text. |
| `run_id` | str \| null | `null` | Generated when unset. |
| `use_run_subdir` | bool | `false` | Write under `output_dir/run_id` instead. |
| `tags` | list[str] | `[]` | Echoed into `run.json`. |
| `notes` | str | `""` | Free text. |
| `extensions` | list[str] | `[]` | Components defined outside the package, loaded before any name in the config is checked. Each entry is a path ending in `.py`, resolved like `data.path`, or a dotted module name; the module's `register()` adds its names to the registries. Recorded in `run.json` with each file's SHA-256, and shown in amber in the plan header. Chapter 12. |

`experiment.extra` accepts nothing: every option is a named field, so
`experiment.sead` fails rather than being ignored.

## 4. `server`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `strategy` | enum | **required** | One of the nine below. |
| `participation_rate` | float | — | In `(0, 1]`. A fixed number of clients per round, `ceil(rate × clients)` and at least one. **Exactly one** of this and `participation_probability` is required. |
| `participation_probability` | float | — | In `(0, 1]`. Bernoulli participation, for any strategy: each client joins each round independently with this probability, one value for every client. The count varies by round and can be zero; a round that selects no client is not aggregated, so the model and the server's state carry over unchanged, and it records `num_clients` 0. **Exactly one** of this and `participation_rate` is required. |
| `metrics` | list[str] | **required** | Filters fit-phase and server-diagnostic columns. Empty list keeps everything. Does **not** filter evaluation columns — chapter 08 §4.3. |

Registered strategies — `server_strategies` (`fedbrew/core/registry.py`):

```
fedavg  fedavgm  fedadam  fedyogi  fedadagrad  fedopt
scaffold  fedlalr  centralized
```

**A name outside this list fails at config load**, with the registered names
in the message. The same applies to `client.update_rule`, `task.name`,
`data.name` and `model.name`: one table, `REGISTERED_NAMES` in
`fedbrew/core/config.py`, read by `_validate_registered_names`. The check
stands down if registering the builtins raises at all — an absent optional
extra is a build-time problem and reporting it as an unknown component name
would be the wrong problem in the wrong words.

`server.extra` keys:

| Key | Type | Default | Required by |
| --- | --- | --- | --- |
| `aggregation_weighting` | `examples` \| `uniform` | `"examples"` | any strategy. Weights the **parameter** average only; metrics stay example-weighted in both modes. |
| `server_optimizer` | str | — | `strategy: fedopt` only, and **refused** under the four aliases. |
| `server_learning_rate` | float | — | the FedOpt family. |
| `beta1` | float | — | the FedOpt family. |
| `beta2`, `tau` | float | — | the FedOpt optimizers that read them, and **refused** by the ones that do not. |

`server_optimizer` is **required only for `strategy: fedopt`**, and refused
under the four named aliases (`fedadam`, `fedyogi`, `fedadagrad`, `fedavgm`),
which set the optimizer themselves. No shipped config uses either spelling of
the key, so it is never exercised by a shipped arm. Chapter 07 covers the
difference.

`beta2` and `tau` are required and refused per optimizer, not per family:
`fedavgm` reads neither, `fedadagrad` reads `tau` and not `beta2`, and only
`fedadam` and `fedyogi` read all four. Chapter 07 §3.2 has the table and the
reason each one has nowhere to enter.

Refused rather than warned about. It used to warn only when the value
*differed* from the strategy — a warning about a key nothing read, silent when
it agreed — because the factory decided the optimizer for a named strategy and
the aliases' own default could never fire. Each alias now names its optimizer
in one place, and a key its component never receives is refused, which is the
rule `UNHONOURED_CLIENT_OPTIONS` already applies on the client side.

## 5. `client`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `update_rule` | enum | **required** | One of the nine below. |
| `batch_size` | int | **required** | Training batch size. |
| `metrics` | list[str] | **required** | Filters the `fit_`-prefixed task metrics at the client. Applied **before** the algorithm extras, so it cannot remove them. |
| `learning_rate` | float \| null | `null` | |

Registered update rules — `client_updates` (`fedbrew/core/registry.py`):

```
local_sgd  fedavg  centralized  local_adamw  fedprox
scaffold  delta_sgd  fedlalr  fedavg_ft
```

### 5.1 `client.extra` — the shared SGD engine

Read by `local_sgd`, `fedavg` and `centralized`, and partly by the others.

| Key | Type | Default | Where the default lives |
| --- | --- | --- | --- |
| `train_shuffle` | bool | **`true`** | `_training_client_kwargs` (`fedbrew/core/factory.py`) |
| `eval_shuffle` | bool | **`false`** | `_training_client_kwargs` (`fedbrew/core/factory.py`) |
| `drop_last` | bool | **`false`** | `_training_client_kwargs` (`fedbrew/core/factory.py`) |
| `eval_batch_size` | int \| null | `max(batch_size, 256)` for classification; `batch_size` for causal_lm | `_eval_batch_size` (`fedbrew/core/factory.py`) |
| `momentum` | float | — | required for the rules that read it |
| `weight_decay` | float | — | same |
| `nesterov` | bool | — | requires positive `momentum` |
| `learning_rate_schedule` | `constant` \| … | — | `_training_client_kwargs` (`fedbrew/core/factory.py`) |
| `min_learning_rate` | float | — | floor for the schedule |
| `update_mode` | `sequential_epoch` \| `single_batch` \| `frozen_batch_gradients` \| `full_gradient` | — | required for `fedavg`, `centralized` and `fedavg_ft`; `delta_sgd` takes all four and runs `sequential_epoch` when unset; `fedprox`, `scaffold`, `fedlalr`, `local_sgd` and `local_adamw` take `sequential_epoch` (also when unset) or `full_gradient`; `frozen_batch_gradients` is refused on `causal_lm` (§2.1) |
| `frozen_gradient_weighting` | `examples` \| `uniform` \| `sum` | — | same |
| `max_local_steps` | int \| null | `null` | caps steps per round |
| `max_grad_norm` | float \| null | **`null` — no clipping** | `_training_client_kwargs` (`fedbrew/core/factory.py`); bounds a different quantity per `update_mode` — chapter 07 §4.1 |

Three of those defaults are set in no shipped config and change what the
optimiser does:

- **`train_shuffle: true`.** Turning it off changes the optimisation, not just
  throughput, and nothing in a config would say so.
- **`drop_last: false`.** Setting it `true` drops each client's final partial
  batch. On cross-device FEMNIST many writers hold fewer than `batch_size`
  examples, so those clients would contribute **zero** gradient while still
  being counted in the round's client total.
  `tests/test_empty_training_batches.py` exists for this.
- **`eval_shuffle: false`.** Harmless for the metrics — a mean does not care
  about order — but it is the reason the per-client evaluation state can be
  reused between rounds. Chapter 11 covers that.

### 5.2 `client.extra` — per-rule keys

| Key | Type | Default | Rule |
| --- | --- | --- | --- |
| `proximal_mu` | float | — | `fedprox` |
| `beta1` | float | `0.9` | `fedlalr` |
| `beta2` | float | `0.999` | `fedlalr` |
| `epsilon` | float | `1e-8` | `fedlalr` — **read by the server too**, see below |
| `eta_0` | float | — | `delta_sgd` |
| `theta_0` | float | `1.0` | `delta_sgd` |
| `gamma` | float | `2.0` | `delta_sgd` |
| `delta` | float | `0.1` | `delta_sgd` |
| `eta_max` | float \| null | `null` — no clamp | `delta_sgd` |
| `finetune_epochs` | int | — | `fedavg_ft` |
| `finetune_learning_rate` | float \| null | `null` → inherits `client.learning_rate` | `fedavg_ft` |

**`client.epsilon` has two consumers.** The FedLALR client uses it as the Adam
epsilon; the FedLALR *server* also reads it to seed `v̂₋₁ = ε²`
in `_build_server` (`fedbrew/core/factory.py`), with the same default. One key,
two readers, one
implicit default — changing it moves both.

**`client.finetune_learning_rate` defaults to the training rate.** A
personalization arm that sets `finetune_epochs` and not the rate fine-tunes at
whatever `client.learning_rate` was. Defensible, and invisible in the config.

### 5.3 Options a rule silently ignores

The factory hands each of nine of the engine keys above only to the rules whose
local step reads it (`ENGINE_CLIENT_OPTIONS`):

```
momentum  weight_decay  nesterov  learning_rate_schedule  min_learning_rate
update_mode  frozen_gradient_weighting  max_local_steps  max_grad_norm
```

Rather than ignore a key its rule never receives, config validation rejects it,
per rule (`UNHONOURED_CLIENT_OPTIONS`, guarded by
`tests/test_ignored_client_options.py`). Chapter 07 §6 has the rule-by-rule
table. `fedavg`, `centralized` and `fedavg_ft` receive all nine; every other
rule refuses between four and all of them.

**And validation requires them of all three.** It used to require them of
`fedavg` and `centralized` only: `UPDATE_MODE_CLIENT_RULES` and
`FIXED_LR_SGD_CLIENT_RULES` were defined in both `config.py` and `factory.py`,
with `fedavg_ft` in the factory's copy alone. The factory therefore built that
rule with all nine while `_validate_local_sgd_options` returned before checking
any of them, and a `fedavg_ft` config missing `momentum` was accepted by
validation and caught only by `TorchSGDClient.__init__`, several seconds into
the run. One definition now, in `config.py`, guarded by
`tests/test_client_rule_sets_are_shared.py`.

**Taking a mode is not taking a fixed rate.** `FIXED_LR_SGD_CLIENT_RULES` was
derived from `UPDATE_MODE_CLIENT_RULES`, and the `update_mode` check ran from
inside the fixed-rate one, so a rule could take the engine's modes only by also
stating `momentum`, `weight_decay`, `nesterov` and a schedule. They are two
sets now, each built from `FEDAVG_ENGINE_CLIENT_RULES` (the three rules above),
with `UPDATE_MODES_BY_CLIENT_RULE` saying which modes each rule takes and
`frozen_gradient_weighting` required of the rules that can run
`frozen_batch_gradients`. No rule's settings changed.

## 6. `task`, `data`, `model`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `task.name` | str | derived | The task the model was registered with; a config cannot set it. Chapter 06. |
| `data.name` | str | `""` | Inferred from `data.path` when unset. |
| `data.path` | str \| null | `null` | Required for `manifest_dataset`. |
| `data.num_clients` | int \| null | `null` | `synthetic_classification` only. |
| `data.samples_per_client` | int \| null | `null` | same |
| `data.input_dim` | int \| null | `null` | same |
| `data.num_classes` | int \| null | `null` | same |
| `model.name` | str | **required** | One of eight builders. |
| `model.input_dim` / `hidden_dim` / `num_classes` | int \| null | `null` | Injected into every builder. |

`input_dim` and `num_classes` appear in both blocks, and they are the same
word for two different jobs: `data.*` sizes the in-memory
`synthetic_classification` backend, `model.*` sizes the input and output
layers. They are checked against each other — not block against block, but
each against the **dataset's own** `get_metadata()`, which is `data.*` for the
in-memory backend and the manifest on disk for every other. A model whose
`num_classes` disagrees with its data is refused when the components are built,
and reported by `--validate-only` before the job starts
(`model_data_shape_mismatch`, `fedbrew/core/factory.py`). A key only one side
declares is not a disagreement: `femnist_resnet18` takes no `input_dim`, and a
manifest need not describe a shape its models take as given.

`data.extra` and `model` builder keys are covered where they are read:
`data.extra` accepts nothing from a built-in backend (its every option is a
named field), and the per-builder `model` key tables are chapter 06 — each
builder declares a `_KNOWN_KEYS` frozenset and rejects anything outside it.

A strategy, update rule or dataset backend loaded through
`experiment.extensions` may declare the keys it reads at registration
(`Registry.register(..., config_keys=)`). They are accepted in its block only
while that component is the one the config selects, so a key declared for one
strategy does not load clean under another.

## 7. `runtime`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `device` | `cpu` \| `cuda` \| `auto` | **required** | `auto` resolves to `cuda` only when a CUDA allocation succeeds. A node that *has* a GPU and cannot hand one out warns and says why; one with no CUDA at all falls back in silence, because there is nothing surprising to report. `run.json` records both `requested_device` and `resolved_device` either way. |
| `use_amp` | bool | **required** | Refused as `true` on a task with no autocast path (`causal_lm`, any extension task), and wherever the local step passes a gradient collector with no `param_groups`: `delta_sgd`, `fedlalr`, `update_mode: frozen_batch_gradients` and `update_mode: full_gradient`. Everything else accepts it. Chapters 06 §3 and 07 §4.4. |
| `deterministic` | bool | **`false`** | Now written into every shipped run config. Chapter 10. |
| `deterministic_warn_only` | bool | `false` | `true` warns instead of failing on a nondeterministic op. |
| `resume_from` | path \| null | `null` | Written by `--resume-from`. |
| `resume_latest` | bool | `false` | Written by `--resume-latest`. |
| `quiet`, `verbose`, `no_rich` | bool | `false` | Written by the matching flags. `verbose` expands the plan header to every column and reports every round; `quiet` wins if both are set. |
| `print_every` | int ≥ 1 \| null | `null` | Written by `--print-every`. The rounds that print a block: 1, N, 2N, … and the final round — the rounds an `every: N` schedule pins — in place of the evaluation rounds. What is evaluated and written to `round_metrics.csv` does not change. `verbose` still widens each block to every column; `quiet` wins over it. |

### 7.1 `runtime.performance`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `matmul_precision` | `highest` \| `high` \| `medium` | **absent → torch's `highest`** | **Changes the numbers.** See below. |
| `torch_num_threads` | int ≥ 1 \| null | `null` → **torch decides** | CPU thread count. Undocumented as "unset means torch's own default" until now. `0` is refused: it is not a way to say "let torch decide" — omit the key. |
| `cudnn_benchmark` | bool \| null | `null` → torch's `false` | **Ignored when `deterministic: true`** — `configure_runtime` (`fedbrew/core/runtime_setup.py`). Must be a real bool: it is read through `bool()`, which takes `"false"` as true. |
| `reuse_model` | bool | `true` | One cached model instance per architecture. |
| `fast_batching` | bool | `true` | Classification only. |
| `shard_cache_bytes` | int | `4294967296` (4 GiB) | Forced to `0` for the centralized strategy, whose pooled view would only hold a second copy. |
| `dataloader` | mapping | — | Four keys, below. |

These three are checked by `validate_config` and then **applied**, not
attempted. `configure_runtime` used to wrap the torch import, the CUDA probe
and all three settings in one `try` under a bare `except Exception` that
returned early, so a `torch_num_threads` value `int()` could not read left
`cudnn_benchmark` and `matmul_precision` unapplied and absent from the record —
the run trained at torch's default precision while the config asked for
another, and the only trace was a `run.json` key nothing reads. A value the
runtime cannot apply is now refused at load; past that, a setting that fails is
a fault and stops the run. Only the torch import, which is optional, is still
caught and recorded as `runtime_setup_error`.

**`matmul_precision` is the one key here that changes results.** `high` puts
fp32 matmuls on TensorFloat32 (10 stored mantissa bits) or a bfloat16 pair
(~16), and `medium` on bfloat16 (8 mantissa bits), against 24 for `highest`.
Absent means `highest`, not
"unset" — so a config that omits it and one that sets `high` are not running
the same arithmetic. Every shipped run config states it explicitly, and
`tests/test_shipped_config_explicitness.py` keeps it that way.

A typo is rejected at load rather than passed to torch:
`torch.set_float32_matmul_precision` does not raise on an unknown value — it
emits a `UserWarning` and leaves the setting untouched — so a typo would have
run at `highest` while `run.json` recorded the typo as though it applied.

### 7.2 `runtime.performance.dataloader`

Exactly four keys, in `_KNOWN_EXTRA_KEYS` (`fedbrew/core/config.py`):

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `num_workers` | int | `0` | Also settable with `--num-workers`, which creates the block when absent. |
| `pin_memory` | bool | `false` | Throughput only. Inert while `num_workers == 0` and `fast_batching` is on. |
| `persistent_workers` | bool | torch's default | **Only read when `num_workers > 0`.** |
| `prefetch_factor` | int | torch's default | **Same gate.** |

`num_workers` is `0` in every shipped config, so the last two are unreachable
as shipped. That is not a reason to set them without also raising
`num_workers`.

Four keys were **removed** from this block because the client overwrites them
on every call. Naming one now fails with its reason: `batch_size`, `shuffle`,
`drop_last` and `seed` — see §10.

### 7.3 `runtime.checkpointing`

The defaults depend on whether the **block** is present, and the two sets are
opposites — `checkpoint_config_with_defaults` (`fedbrew/core/checkpointing.py`).

| Key | Default when the block is present | Default when the whole block is absent |
| --- | --- | --- |
| `enabled` | `true` | `true` |
| `interval` | `1` | `1` |
| `save_last` | **`true`** | **`false`** |
| `save_best` | **`true`** | **`false`** |
| `save_every_round` | **`false`** | **`true`** |
| `keep_last` | **`3`** | **`null` — never prune** |
| `best_metric` | `val_accuracy_sample_weighted_avg` | same |

So a config with no `checkpointing` block writes a numbered checkpoint **every
round**, never writes `latest.pt` or `best.pt`, and **never prunes** — the
reverse of every per-key default. A reader who knows the documented defaults
gets the opposite behaviour on such a config. All shipped configs now carry the
block.

`best_mode` is **derived from `best_metric`'s name, never configured**, and
`best_metric` must be a `val_` or `personal_val_` metric. Chapter 08 §11.

**`save_best` has to agree across the arms of a comparison.** It writes a file;
it does not move a number — running `configs/dev/synthetic.yaml` twice under
`deterministic: true`, identical but for this key, returns all three per-round
checkpoints byte-identical, and of `round_metrics.csv`'s 52 columns over 3
rounds only the six wall-clock ones differ. What it changes is what a run can
be *read at*: an arm with it on is read at its best validation round, an arm
with it off only at its final one, so a table quoting a "best" figure across
the two gives one arm a maximum over every validation evaluation and the other
a single draw. Which way a family sets it is free; that its arms agree is not,
and `tests/test_comparison_arms_agree_on_save_best.py` walks every shipped
family for it.

### 7.4 `runtime.data_staging`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Also driven by `--staging` / `--no-staging`. |
| `local_root` | path \| null | `null` | **The only way to point staging at a specific scratch path.** |

`local_root` accepts an environment-variable path. An unset value and one
naming a variable the job did not export are the same answer: there is no
usable scratch root, so nothing is staged rather than a literal
`$FL_LOCAL_SCRATCH` directory being created
— `resolve_staging_root` (`fedbrew/core/data_staging.py`). Chapter 11 covers
when staging is worth it.

## 8. `evaluation`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `train.every` | int \| `final` \| `never` | `10` | |
| `train.clients` | scope | `"participating"` | |
| `val.every` | int \| `final` \| `never` | `5` | |
| `val.clients` | scope | `"all"` | |
| `test.every` | int \| `final` \| `never` | `10` | |
| `test.clients` | scope | `"all"` | `participating` is **rejected** for test. |
| `central_test.every` | int \| `final` \| `never` | `10` | |
| `model_scope` | `global` \| `personal` \| `both` | `"global"` | |

**Schedule grammar** — `parse_evaluation_schedule` (`fedbrew/core/config.py`):

| Value | Meaning |
| --- | --- |
| positive int *n* | rounds where `round % n == 0`, **plus round 1 and the final round** |
| `"final"` | the final round only |
| `"never"` | the split is not evaluated |

An interval pins round 1 as well as its own multiples, so a 500-round run at
`every: 10` both starts from a measured baseline and ends on a measured round.
`0` and negatives are rejected with a message naming `final` and `never`.

**Client scope grammar** — `parse_evaluation_client_scope` (`fedbrew/core/config.py`):

| Value | Meaning |
| --- | --- |
| `all` | every client |
| `participating` | only the clients selected for training this round |
| `sample:<N>` | a fixed sample of *N* clients, the same set each time |
| `resample:<N>` | *N* clients redrawn each evaluation |

The roles are enforced, not conventional: train is an optimisation diagnostic,
val is what model selection may look at, and test is for reporting only — so
`test.clients: participating` is refused.

**Both sampling modes draw per split.** `val: sample:200` and `test:
sample:200` are two independent draws, not one shared set. They used to be one:
the draw key carried the seed, the size and the round and not the split, so two
splits at the same *N* got the identical clients — and the clients whose
validation data selected `best.pt` were exactly the clients whose test data was
reported. Measured on the two shipped configs in that shape, the overlap was
100% where independent draws give 4% (`sample:40` of 1000) and 14.5%
(`sample:2000` of 13,771). `tests/test_evaluation_draw_is_per_split.py` pins it
as a distribution rather than as "the two sets differ", which a weakly mixed
key would also satisfy.

## 9. `client_statistics` and `divergence`

Both are covered in full by chapter 08 — they decide which metric columns exist
and when a run stops. Summarised here because they are config blocks:

| Key | Default |
| --- | --- |
| `client_statistics.per_client_csv` | `false` |
| `client_statistics.std` | `true` |
| `client_statistics.variance` | `false` |
| `client_statistics.min` | `true` |
| `client_statistics.max` | `true` |
| `client_statistics.worst_percent` | `10.0` |
| `divergence.metric` | `"fit_loss"` |
| `divergence.non_finite` | `true` |
| `divergence.blowup_factor` | `10.0` |
| `divergence.blowup_absolute` | `null` |
| `divergence.patience` | `null` |
| `divergence.min_delta` | `0.0` |

**`divergence: null` turns the whole thing off.** There is no `enabled` key —
it was removed because it was a second spelling of a state the detectors
already expressed, so a config could say `enabled: true` with nothing to
detect. The `active` property on `DivergenceConfig` (`fedbrew/core/config.py`) is true when any detector
is on, and every detector off *is* off. Writing `divergence: null` is the short
way to say that; turning off each detector individually is the long way, and
means the same thing.

**`divergence.metric` is cross-checked against `server.metrics`.** A non-empty
`server.metrics` that omits the watched name filters it out before it reaches
the round record, and every detector then watches nothing for the whole run
without failing. `validate_config` refuses it, so the config fails to load and
no run starts. Evaluation columns, `central_test_*` and
the selected strategy's own diagnostics are exempt, because `server.metrics`
does not reach those — chapter 08 §4.3 says why. The empty default keeps
everything and cannot go wrong.

## 10. Keys that were removed

Naming one of these fails at load with the reason, rather than being accepted
and ignored — `_REMOVED_KEYS` (`fedbrew/core/config.py`).

| Removed key | Why |
| --- | --- |
| `server.name` | `server.strategy` already identifies the strategy |
| `client.name` | `client.update_rule` already identifies the rule |
| `client.type` | checked against the registry and then discarded, so a `type` disagreeing with `update_rule` validated clean and ran `update_rule` |
| `client.lazy_clients` | follows from the dataset: a manifest dataset gets the lazy pool, the in-memory synthetic one does not |
| `model.init` | no builder read it |
| `data.split` | nothing read it |
| `divergence.enabled` | see §9 |
| `runtime.num_workers` | nothing read it; two shipped configs asked here for ten workers and ran at zero |
| `runtime.performance.dataloader.batch_size` | the client passes a batch size on every call, so a value here was always overwritten |
| `runtime.performance.dataloader.shuffle` | `client.train_shuffle` / `client.eval_shuffle` decide it per phase, and win the same way |
| `runtime.performance.dataloader.drop_last` | `client.drop_last` is the setting; a value here was ignored while training and **silently truncated every evaluated split** |
| `runtime.performance.dataloader.seed` | loader seeding is derived per client, round and phase from `experiment.seed`; a fixed seed here would give every client the same shuffle |
| `runtime.data_staging.mode` | `copy_tree` was the only supported value; anything else printed a skip line and left staging off |
| `runtime.data_staging.fallback_local_root` | an unset or unexpanded `local_root` already means "stage nothing" |
| `server.global_rounds` | **relocated**, not deleted: set `defaults.global_rounds` — §2.1 |
| `client.local_iterations` | **relocated**, not deleted: set `defaults.local_iterations` — §2.1 |
| `defaults.local_epochs` | **renamed**: set `defaults.local_iterations` — §2.1. A config that writes it predates the rename |
| `client.local_epochs` | **renamed and relocated**: set `defaults.local_iterations` — §2.1. A config that writes it predates the rename |
| `experiment.task` | the task is recorded by the model's registration (`models.register(..., task=)`) and read from `model.name`; a model cannot be registered without it, so there is nothing left to override |
| the top-level `task` block | the task is inferred from `model.name` through the model's registration |

`server.global_rounds` and `client.local_iterations` are relocations rather
than deletions: the value is still read, and only the spelling a config may use
moved. Their fields therefore still exist on the resolved config, which is why
they have to be declared in `_REMOVED_KEYS` — a check asking "is this still a
field?" cannot tell a config that wrote the old name from the loader having
filled the field in, and for a while neither could. `defaults.local_epochs` and
`client.local_epochs` are the renamed key's old spellings. They are declared
too, so the refusal names the key that replaced them rather than calling them
unknown.

## 11. CLI overrides

`fedbrew run` flags that write into the config, then re-validate
— `apply_cli_overrides` (`fedbrew/core/runner.py`). Each writes one named
field of a block the root already has, so no override can add a top-level
key; a misspelled flag is refused by the parser before the config is read.

| Flag | Writes |
| --- | --- |
| `--config` | which file to load |
| `--validate-only` | run preflight and exit without training |
| `--output-dir` | `experiment.output_dir` |
| `--seed` | `experiment.seed` |
| `--rounds` | `server.global_rounds` — the resolved field, i.e. it overrides `defaults.global_rounds` |
| `--participation-rate` | `server.participation_rate` |
| `--local-iterations` | `client.local_iterations` — the resolved field, i.e. it overrides `defaults.local_iterations` |
| `--lr` | `client.learning_rate` |
| `--batch-size` | `client.batch_size` |
| `--device` | `runtime.device` |
| `--num-workers` | `runtime.performance.dataloader.num_workers`, creating the block if absent |
| `--staging` / `--no-staging` | `runtime.extra["data_staging"]` |
| `--resume-from` / `--resume-latest` | the resume path |
| `--run-id`, `--use-run-subdir`, `--tag`, `--notes` | run identity |
| `--quiet`, `--verbose`, `--no-rich`, `--print-every` | terminal output. All four are also `runtime.extra` keys, so a config can set what a flag sets. `--quiet` is refused beside `--verbose` or `--print-every` |

## 12. Defaults no shipped config states

Collected, because a reader of a config cannot see any of them.

| Key | Implicit value | Cost of assuming wrong |
| --- | --- | --- |
| `client.train_shuffle` | `true` | different optimisation |
| `client.drop_last` | `false` | `true` silently zeroes small clients' gradients |
| `client.eval_shuffle` | `false` | breaks client-eval state reuse |
| `client.epsilon` (fedlalr) | `1e-8`, read by client **and** server | two consumers move together |
| `client.finetune_learning_rate` | inherits `client.learning_rate` | the personalization LR is invisible |
| `runtime.performance.torch_num_threads` | torch's own default | CPU throughput |
| `runtime.performance.pin_memory` | `false` | throughput only |
| `runtime.performance.dataloader.persistent_workers` / `prefetch_factor` | torch's defaults, **and unreachable at `num_workers: 0`** | no effect as shipped |
| `runtime.data_staging.local_root` | `null` | staging silently does nothing |
| `server.server_optimizer` | required only for `fedopt` | never exercised by a shipped arm |
| `runtime.checkpointing.*` with the block absent | the **opposite** of the per-key defaults | §7.3 |

Two more were implicit and are now written into every shipped config rather
than documented: `runtime.performance.matmul_precision` and
`runtime.deterministic`. Both change results, and
`tests/test_shipped_config_explicitness.py` fails if a new config omits either.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/core/config.py` | every dataclass, every default, every validator |
| `fedbrew/core/config.py` | `_KNOWN_EXTRA_KEYS` — **the authority on which `extra` keys exist** |
| `fedbrew/core/config.py` | `_REMOVED_KEYS` — removed keys and their reasons |
| `fedbrew/core/config.py` | `parse_evaluation_schedule` |
| `fedbrew/core/config.py` | `parse_evaluation_client_scope` |
| `fedbrew/core/factory.py` | where most `extra` defaults are applied |
| `fedbrew/core/checkpointing.py` | `checkpoint_config_with_defaults` — the two opposing checkpoint default sets |
| `fedbrew/core/runtime_setup.py` | `configure_runtime` — the performance block's effects |
| `fedbrew/core/data_staging.py` | `resolve_staging_root` |
| `fedbrew/core/runner.py` | `apply_cli_overrides` |
| `configs/reference_evaluation.yaml` | a commented config exercising the evaluation surface |

### Commands

```bash
# Prove this chapter's key tables match the code.
python -m pytest tests/test_docs_config_keys.py -v

# Check a config without training it. The whole preflight runs.
fedbrew run --config configs/dev/smoke.yaml --validate-only

# The validators and the rejection rules.
python -m pytest tests/test_unknown_config_keys.py \
                 tests/test_evaluation_config.py \
                 tests/test_evaluation_schedule.py \
                 tests/test_evaluation_draw_is_per_split.py \
                 tests/test_client_rule_sets_are_shared.py \
                 tests/test_no_dead_or_shadowing_paths.py \
                 tests/test_shipped_config_explicitness.py \
                 tests/test_ignored_client_options.py \
                 tests/test_validation_commands.py
```

### Invariants

1. **`_KNOWN_EXTRA_KEYS` is the authority on which `extra` keys exist.** A new
   option must be added there or it fails at load. Adding the reader is not
   enough.
2. **An unread key is an error.** Never widen an allow-list to make a config
   load; either something reads the key or the key should not be written.
3. **A removed key gets an entry in `_REMOVED_KEYS`, not a silent deletion.**
   The reason is shown to whoever still has it in a config.
4. **The absent-block checkpoint defaults are the opposite of the per-key
   ones.** Any change to one set must state what happens to the other.
5. **`matmul_precision` and `deterministic` must be explicit in every shipped
   config.** Guarded; a new config omitting either fails.
6. **`cudnn_benchmark` is ignored under `deterministic: true`.** Do not
   document it as an unconditional switch.
7. **Validation runs twice** — before and after CLI overrides — so an override
   cannot produce a config that would have been rejected as written.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_config_keys.py` | Every key table here matches `_KNOWN_EXTRA_KEYS`, the dataclass fields and their defaults, `_REMOVED_KEYS`, and the registry names. |
| `tests/test_unknown_config_keys.py` | An unread key fails at load, in every block and at the root, through `load_config`, `fedbrew run` and `--validate-only`. |
| `tests/test_preflight_runs_only_under_validate_only.py` | §1: an ordinary `fedbrew run` calls `validate_config` and never the ten-area preflight; `--validate-only` calls both and trains nothing; the ten areas are the ones §1 names. |
| `tests/test_shipped_config_explicitness.py` | `matmul_precision` and `deterministic` are stated by every shipped config. |
| `tests/test_evaluation_schedule.py` | The `every` grammar, including `final` and `never`. |
| `tests/test_evaluation_config.py` | Split roles and client-scope parsing. |
| `tests/test_evaluation_draw_is_per_split.py` | §8: two splits sampled at one *N* draw independently, and every other property of the draw is unchanged. |
| `tests/test_client_rule_sets_are_shared.py` | §5: the client-rule sets have one definition, and `fedavg_ft` is validated like the other two. |
| `tests/test_no_dead_or_shadowing_paths.py` | §4: each FedOpt alias names its own optimizer, an unread `server_optimizer` is refused, and an extension cannot shadow a built-in name. |
| `tests/test_ignored_client_options.py` | Which of the nine engine keys each rule refuses. |
| `tests/test_cli_num_workers_override.py` | `--num-workers` creates the block when absent. |
| `tests/test_validation_commands.py` | `--validate-only` runs the preflight and exits. |
| `tests/test_cli_flags_exist.py` | Every flag in §11 is one argparse defines. |
| `tests/test_comparison_arms_agree_on_save_best.py` | §7.3: the arms of every shipped family agree on `save_best`, and a selecting arm names its metric. |

### Known failure modes

- **Copying a config that omits `checkpointing`.** The absent-block defaults
  are the opposite of the per-key ones: every round checkpointed, no `best.pt`,
  no pruning. This fills a disk quota quietly.
- **Assuming an unknown key is ignored.** It is a load error. That is the
  point — but it means a config from an older revision may not load, and the
  error names the reason.
- **Setting `persistent_workers` or `prefetch_factor` without `num_workers`.**
  Both are unreachable at `num_workers: 0`, which is what every shipped config
  has.
- **Setting `cudnn_benchmark: true` alongside `deterministic: true`.** The
  benchmark setting is skipped entirely; the config reads as though both apply.
- **Expecting `server.metrics` to narrow the evaluation columns.** It filters
  the fit path and server diagnostics only — chapter 08 §4.3.
- **Reading `divergence.enabled` in an old config.** Removed; the load error
  names `divergence: null` as the replacement.
- **Assuming `server_optimizer` is needed.** Only `strategy: fedopt` requires
  it; the four named aliases set it themselves.
