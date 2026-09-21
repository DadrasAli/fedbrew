# 12 — Extending

Adding a server strategy, client update rule, task, dataset, model, partitioner
or metric — from outside the package or inside it — and the registration each
one needs to actually take effect.

Every extension point has the same shape: write the thing, register the name,
declare the config keys, add a validator, add a test, ship a config. **Skipping
the third step is the failure that costs most**, because a key nothing forwards
means the run claims a setting it did not use.

## 1. How a component reaches the CLI

Six registries decide which names a config may use (chapter 01 §3), and there
are exactly two ways into them.

| | In the package | Out of tree |
| --- | --- | --- |
| Where the registration lives | `register_builtin_components()` | your module's `register()` |
| What causes it to run | nothing; it is always loaded | a config naming the module under `extensions` |
| Origin carried by every name | `BUILTIN` | the config entry, verbatim |
| Version recorded in `run.json` | the package's commit | the file's SHA-256 |
| Config keys declared in | `_KNOWN_EXTRA_KEYS` | `config_keys=` at registration |
| Everything after registration | — | identical |

**The last row is the point.** An out-of-tree component is selected by the same
config key, built by the same factory, validated by the same preflight, and
recorded in the same `run.json` as a built-in. There is no second code path and
no `--extension` flag: a config names everything a run is made of, and
`run.json` is that config's record, so an extension has to be a thing the
config names or it breaks the property the artifacts are for.

So `fedbrew run --validate-only`, the plan header, `--rounds` and every config
guard work on an out-of-tree component on the first try, and a private runner
that imports the loop directly is never the answer to "the CLI cannot see my
task".

### 1.1 The protocol

An extension is a module that defines `register()` and registers nothing until
it is called:

```python
def register() -> None:
    from fedbrew.core import registry

    registry.tasks.register("my_task", MyTask)
```

Four rules, and the loader (`fedbrew/core/extensions.py`) enforces them:

1. **Nothing registers at import.** The module can then be imported for its
   other definitions — by its own tests, by a sweep script — without touching
   a registry.
2. **The import is local to `register()`.** Importing `fedbrew.core.registry`
   at module scope is harmless, but keeping every import inside the function is
   the habit that stops an extension pulling torch into a process that only
   wanted to read its constants.
3. **`register()` is called once per resolved location per process.** Two
   configs naming the same file share one `LoadedExtension` record. A
   *different* file registering a name that exists is refused, with a message
   naming both origins.
4. **Registering is all it does.** `register()` runs during config load, before
   any name in the config has been looked up, so work done there is work done
   on every `--validate-only` too.

### 1.2 Where the entry goes

| Config | Key | Read by |
| --- | --- | --- |
| run config | `experiment.extensions` | `fedbrew run` |
| generator config | `dataset.extensions` | `fedbrew generate` |

Two keys rather than one because the two commands read different files, and a
generator config has no `experiment` block to put it in. A problem defined
outside the package that ships both a generator and a task writes the same
entry in both — which is what `examples/drift-quad` does.

Each entry is either a path ending in `.py`, expanded and resolved the way
`data.path` is, or a dotted module name for something installed. One loader,
two import forms; a file is placed in `sys.modules` under a private prefix, so
two extensions with the same basename cannot shadow one another.

An entry that does not exist, does not import, or defines no `register()` fails
at config load, next to the message about a name that is not registered — which
is the pair of errors you get if you write the config before the module.

### 1.3 Declaring the config keys it reads

`Registry.register(..., config_keys=("my_key",))` is the out-of-tree form of
`_KNOWN_EXTRA_KEYS`, with one difference that removes the worst failure mode in
this chapter: **declaring and forwarding are one act.** A declared key is
accepted in the component's config block *and* handed to its factory as a
keyword argument. You cannot declare a key the run then ignores.

They are accepted only while that component is the one the config selects, so a
key declared by one strategy does not load clean under another. Chapter 04 §6.

| Registry | Block its declared keys are read from |
| --- | --- |
| `server_strategies` | `server` |
| `client_updates` | `client` |
| `datasets` | `data` |
| `tasks`, `models`, `generators` | none — declaring keys on these raises |

The last row is not an oversight. A model's keys are its own, declared as
`_KNOWN_KEYS` and rejected by `reject_unknown_model_keys` (§5); a generator's
are declared as the config *sections* it reads (§4); and a task is built from
the run's shape rather than from a block of its own.

### 1.4 What the factory hands an out-of-tree component

A built-in is constructed by a dispatch table that enumerates built-in names,
and an out-of-tree component can appear in none of them. So `is_extension`
(`fedbrew/core/factory.py`) branches on **where the registration came from**,
never on a name list, and the extension is built from a fixed contract:

| Registry | Contract | Given |
| --- | --- | --- |
| `tasks` | `EXTENSION_TASK_KEYS` | `model_config`, `batch_size`, `eval_batch_size`, `device`, `dataloader_config`, `reuse_model`, `dataset_metadata` |
| `datasets` | `EXTENSION_DATASET_FIELDS` | the set `data` fields among `path`, `num_clients`, `samples_per_client`, `input_dim`, `num_classes`, plus `seed` |
| `client_updates` | `EXTENSION_CLIENT_KEYS` | what `FedAvgClient` is given, `total_rounds` included |
| `server_strategies` | — | FedAvg's common keyword arguments |

Plus, in every case, the keys the component declared. `tests/test_extension_components.py`
pins the three tuples, so a contract that changes is a change someone makes on
purpose.

Take `**unused: Any` and ignore what you do not want. The contract is a
superset on purpose: a task that has no throughput question still gets
`batch_size`, because the alternative is a constructor signature that has to be
guessed.

### 1.5 What preflight will not check for you

**Pairing.** Every pairing rule in `fedbrew/core/validation.py` enumerates
names the package ships, so it cannot judge a pair that includes yours.
Preflight reports `algorithm.extension_pairing_unchecked` as an info issue and
stands down. If your strategy needs its own client rule, **refuse the wrong
partner in your own constructor**, where you can see both — an incompatible
pair then fails when it is built rather than at preflight, which is the
deliberate cost of leaving the hook open.

**`client.learning_rate`.** Preflight neither requires nor refuses one for an
out-of-tree rule, because whether a rule derives its own step size is a fact
about the rule.

Everything else — the config guards, the metric-name rules, the checkpoint
selection rules, the determinism checks — applies unchanged, because none of
them names a component.

### 1.6 What the run records

| Where | What |
| --- | --- |
| plan header | an amber `Extensions` row, one line per entry |
| `run.json` | `reproducibility.extensions`: per entry, the resolved path, the file's SHA-256 and every `(registry, name)` it registered |
| `manifest.json` | `extensions`: the same, for the generator that wrote the data |

`code_state` covers the package's commit and nothing outside it, so the SHA-256
is the only thing that ties a run to a version of your file. It is recorded
without being checked: nothing refuses a run because an extension changed since
the last one. The record is what makes the change findable afterwards.

The plan header row is amber, which is the header's scarcest colour — it marks
the things that change what a number means. Chapter 03 §2 lists the five, and
`Extensions` is one of them because a name in this config may be a different
object than the same name in the next.

### 1.7 What a new algorithm in the package must declare

Everything above is the out-of-tree route, where a name reaches the CLI without
the package knowing about it. **Adding a strategy or a rule to
`register_builtin_components()` is the other route, and it is the one with
homework.** Nine guards enumerate the two algorithm registries and diff them
against a hand-written table, so a name registered and nothing else turns the
suite red in nine places at once.

That is deliberate. Each table is a question the author is the only person who
can answer, asked at the moment they are the only person thinking about it.

| Guard | The table to add a row to | What the row declares |
| --- | --- | --- |
| `tests/test_docs_architecture.py` | chapter 01's registry table | that the name exists, and the per-registry count above it |
| `tests/test_docs_config_keys.py` | chapter 04's `Registered strategies` / `Registered update rules` blocks | that a config may name it |
| `tests/test_docs_algorithms.py` | chapter 07's registry listing, its prose, and §6's per-rule table | what the component is, and which `client` options it silently ignores — the §6 row is diffed against `UNHONOURED_CLIENT_OPTIONS` in `fedbrew/core/config.py` |
| `tests/test_aggregation_weighting_notice.py` | `AGGREGATION_WEIGHTING_NOTICE` in `fedbrew/core/validation.py` | the aggregation-weighting notice this strategy emits, or `None` with the reason it emits none |
| `tests/test_ignored_client_options.py` | `RULE_CONFIGS`, in the test | a config that builds the rule, so its ignored-option row is derived from the factory's gates rather than asserted |
| `tests/test_client_communication_cost.py` | `BUILDERS`, in the test | how to build the client, so the communication meter is checked against the payload it actually sends |
| `tests/test_planned_columns_are_written.py` | `RULES`, in the test | a minimal client block for the rule and the strategy it pairs with, so the plan header's column list and its "not written" row are checked against the CSV the rule actually writes |
| `tests/test_adapter_state_support.py` | `ADAPTER_STATE_CLIENT_RULES` or `FULL_STATE_ONLY_CLIENT_RULES` in `fedbrew/core/federated_state.py` | whether the rule trains adapter-only (LoRA) state, or why it cannot and is refused with it — chapter 07 §5.1. Asks of a rule only |
| `tests/test_active_target_weighting_is_honoured_or_refused.py` | `AGGREGATION_WEIGHT_HOOK_CLIENT_RULES` or `AGGREGATION_WEIGHT_HOOK_BYPASS_RULES` in `fedbrew/core/federated_state.py` | whether the rule's client asks the task for its aggregation weight, checked by running its `fit`; a rule that does not is refused with `active_target_weighting` on — chapter 07 §3.1. Asks of a rule only |

Measured rather than assumed: registering one strategy and one rule that reuse
FedAvg's builders, and running the suite, failed exactly the first seven
modules — eleven test methods, four of them subtests. Measured on 2026-09-16,
when the seventh was added; the eighth and ninth, added on 2026-09-21, are
not in that measurement.

Two more fire on what the new name *is* rather than on the name itself:

| Guard | Fires when |
| --- | --- |
| `tests/test_resume_is_all_or_nothing.py` | the strategy's builder constructs a server class the coupled-state sweep does not already cover — confirmed by registering a `FedAvgServer` subclass, which fails it where reusing `FedAvgServer` does not |
| `tests/test_communication_cost_metrics.py` | the client both measures and filters its own metrics, which is the shape at risk of reporting a cost it did not pay |

**The three registries with no row here — `tasks`, `models`, `generators` — are
covered differently**, by `_KNOWN_KEYS`, by the model's `task=`, and by a
generator's declared sections. §1.3.

`tests/test_docs_extending.py` checks this table in both directions: every
guard named exists and enumerates a registry, and a test module that enumerates
one and appears in neither table has to be classified before the suite is
green.

## 2. The worked case: `examples/drift-quad`

A federated problem with a closed-form optimum, defined entirely outside the
package: a generator that writes its data, a task that scores against the known
answer, and a model that is the iterate. `examples/drift-quad/problem.py`
contains all of it, and the whole of its registration is this:

```python examples/drift-quad/problem.py
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
        generate_drift_quad_from_config,
        sections={"problem": {"dim", "condition_number", "dissimilarity"}},
    )
    registry.tasks.register(TASK_NAME, lambda **kwargs: DriftQuadTask(**kwargs))
    registry.models.register(MODEL_NAME, build_quad_vector, task=TASK_NAME)
```

Its generator config names the file, so `fedbrew generate` can find the
generator that `dataset.name` selects:

```yaml data/configs/examples/drift-quad.yaml
  name: drift_quad
  output_dir: data/generated/examples/drift-quad
  seed: 42
  # The problem is defined outside the package, so the generator is too.
  extensions:
    - examples/drift-quad/problem.py
```

Its eight arm configs name the same file, so `fedbrew run` can find the task
and the model:

```yaml configs/examples/drift-quad/fedavg.yaml
  # The task, the model and the generator are defined outside the package.
  # Loaded before any name below is looked up; run.json records the file's
  # SHA-256 beside the commit.
  extensions:
    - examples/drift-quad/problem.py
```

And then it is an ordinary dataset and an ordinary set of runs:

```bash
fedbrew generate --config data/configs/examples/drift-quad.yaml
fedbrew inspect-data data/generated/examples/drift-quad/manifest.json
fedbrew run --config configs/examples/drift-quad/fedavg.yaml
```

**What it does not contain** is the part worth reading twice. There is no edit
to any file under `fedbrew/`, no name of it in any in-tree list, no `run.py`
that imports the loop, and no branch anywhere that asks whether a component is
an example. It is loaded by the same loader that would load yours, from a
directory that could be anywhere on disk.

Two things it does that a smaller extension would not need, both covered below:
it writes its optimum into the manifest under `reference` (§4.1), and it
declares that nothing is held out (§4.1 again), because an analytic objective
has nothing to hold out.

## 3. A new algorithm

1. **Write the client update, the server strategy, or both.** Subclass
   `ClientUpdate` in `fedbrew/clients/base.py` or `ServerStrategy` in
   `fedbrew/servers/base.py`. Implement `aggregate_stream` on a new strategy if
   you can — the base class falls back to buffering every client result, which
   costs one model copy per selected client. Chapter 01 §5.

2. **Register it** under the name configs will use — in your module's
   `register()`, or in `register_builtin_components()`
   (`fedbrew/core/registry.py`) if it belongs in the package. A built-in
   registration goes through a local import inside a builder function so that
   importing the registry does not import torch; an extension's `register()`
   does the same thing for the same reason.

   Until it is registered, a config naming it fails at load with the list of
   names that *are* registered (`_validate_registered_names`,
   `fedbrew/core/config.py`) — which is the first thing you will see if you
   write the config before the registration.

3. **Wire any new config keys.** Out of tree: `config_keys=` at registration,
   which declares and forwards in one act (§1.3). In the package: add them to
   `_KNOWN_EXTRA_KEYS` (`fedbrew/core/config.py`) *and* hand them to the
   constructor in `fedbrew/core/factory.py`. A key not in the set is rejected
   at load. A key in the set that the factory never forwards is **worse**: the
   config loads, the run does not use the value, and nothing says so. Chapter
   04 §1.

4. **Add a validator** in `fedbrew/core/validation.py` if the algorithm has a
   pairing requirement (SCAFFOLD and FedLALR both require a matching client and
   server), a hyperparameter range, or a communication cost other than 1×.
   Chapter 07 §2 and §5. An out-of-tree algorithm validates itself instead;
   §1.5 says what preflight will and will not do for it.

5. **Add a test.** The shipped ones are the template:
   `tests/test_aggregation_correctness.py` for what the server computes,
   `tests/test_fedlalr.py` and `tests/test_delta_sgd.py` for a whole algorithm,
   `tests/test_client_communication_cost.py` for reported volume. An
   out-of-tree component's tests live with it — `examples/drift-quad/` has its
   own, and `tests/test_examples_are_extensions.py` checks that a migrated
   example really does reach the CLI this way.

6. **Ship a config.** `configs/<dataset>/<algorithm>.yaml` for a built-in;
   anywhere you like for an extension. Either way it must state
   `matmul_precision`, `deterministic` and the rest of the settings that change
   numbers, which `tests/test_shipped_config_explicitness.py` enforces for
   everything under `configs/`.

7. **Document it** in chapter 07 if it is in the package, and add any new
   metric to chapter 08. Both have guards that will fail if a *registered*
   built-in name has no prose. The guards read `builtin()` rather than `list()`,
   so an extension loaded by a test does not put the suite into a state where
   the chapters are required to document it.

### 3.1 If the algorithm sends more than the model

Auxiliary state — a control variate, an optimizer moment — travels in
`FitResult.payload`, never as a new field on the protocol dataclasses.

Add its bytes into the client's own `communicated_bytes` so the metric stays
the true upload volume, as SCAFFOLD and FedLALR do in
`TorchScaffoldClient.fit` (`fedbrew/clients/torch_scaffold_client.py`) and
`TorchFedLALRClient.fit` (`fedbrew/clients/torch_fedlalr_client.py`). Then add a
preflight notice stating the multiplier, so a user comparing arms on round
count is told not to.

State the server sums outside `WeightedStateAccumulator` is not checked by
it. Pass each incoming piece, and the persistent state it would produce,
through `refuse_non_finite_state` (`fedbrew/core/torch_utils.py`) before
assigning anything, as `ScaffoldServer.aggregate_stream` does: the loop turns
the `NonFiniteStateError` into a recorded divergence, but only if nothing was
mutated first.

### 3.2 If the algorithm produces a personal model

It must accept `evaluation.model_scope: personal` and report its metrics under
the `personal_` split prefix. A rule with no personal model must **reject** a
non-global scope rather than evaluate the global one under a personalized name.
`fedavg_ft` is the worked example; chapter 07 §4.7.

## 4. A new dataset

1. **Write a generator** exposing
   `generate_<name>_from_config(config, output_dir, seed, client_splits)` and
   returning a summary with `num_clients`, `num_examples`, `num_test_examples`
   and `manifest_path`. In the package it goes under `fedbrew/data/`; out of
   tree it goes wherever your module is. Write shards through
   `fedbrew/data/writers/torch_shards.py` and the manifest through
   `fedbrew/data/writers/manifest.py`, so the output matches what
   `manifest_dataset` and `inspect-data` expect.

   An argument you do not use stays in the signature and says why. Two of
   drift-quad's four do nothing — the offsets are a deterministic function of
   the spec, and there is no cut to split — and its docstring is a paragraph on
   each, because a silently ignored `seed` is how a dataset becomes
   irreproducible without anyone touching it.

2. **Register it** in the `generators` registry:
   `registry.generators.register(name, target, sections={"problem": {...}})`.
   Kind `shards`
   (the default) for a generator that writes its own manifest, kind `tensors`
   for one returning a pooled `(train_x, train_y, test_x, test_y, metadata)`
   pair from `(config, seed)` that the shared partitioners should split. A
   built-in names its target as a `"module:function"` string resolved on first
   use, which is what keeps registering the eight of them from importing
   torchvision or transformers; an extension can pass the function, since its
   module is already imported by the time `register()` runs.

   `GeneratorRegistry.register` builds the `GeneratorSpec`
   (`fedbrew/core/registry.py`) from those parts, so nobody constructs one by
   hand.

3. **Declare its config sections, and the keys inside each**, as
   `sections={"problem": {"dim", "condition_number", "dissimilarity"}}`. A
   section not declared is refused at generate time with a message naming
   what the generator *does* read — which is what stops `sequence_lenght: 512`
   generating 256-token windows silently — and a key not declared inside a
   section is refused the same way, one level down, which is what stops
   `condition_numbr: 100.0` generating κ = 100.

   The package's own eight write the shorter `sections={"mnist"}`, naming the
   section only, because their key lists live in `_GENERATOR_SECTION_KEYS`
   (`fedbrew/data/generate.py`) beside the readers; a built-in adds its keys
   there. A name-only section that `_GENERATOR_SECTION_KEYS` does not list is
   **refused on the first `generate`**, not left unchecked: it was, once, and
   drift-quad's `problem` block silently generated a dataset at the default.

4. **Decide the three client splits.** `train_ratio`, `eval_ratio` and
   `test_ratio`. If the source has no external test set, the test split comes
   out of the clients: a third per-client slice, as FEMNIST cuts, or clients
   held out whole. Chapter 05 §4 says what each one measures.

5. **Add a generator config** under `data/configs/`, and an experiment config
   under `configs/`.

6. **Run `fedbrew inspect-data`** on the result. It validates the manifest
   against the shards and prints the per-client label distribution — which is
   how you check the partition did what you asked.

7. **Add a row to chapter 05's provenance table** if the generator is in the
   package, honestly. A generator with no shipped config and no test is
   documented from its allow-list alone, and the chapter says so.

### 4.1 If the data has a known answer

A generated problem whose optimum is known in closed form writes it into the
manifest under `reference`, as free-form JSON:
`run_metadata.build_dataset_provenance` copies it into `run.json`, so a run
records what it was scored against, and an extension task reads it from the
`dataset_metadata` it is handed. `examples/drift-quad` is the worked case —
its `reference` carries `x*`, `F*`, the dials that produced them and every
floor they imply.

That is also the place to put anything a task must agree with the data about.
`DriftQuadTask` is handed both `model_config` and `dataset_metadata`, and it is
the only object that is, so it cross-checks the curvature in the `model` block
against the dials in `reference` and refuses a run whose two halves describe
different problems.

If nothing is held out — an analytic objective is not estimated from samples,
so there is nothing *to* hold out — say so in the manifest with
`client_test_source: identical_to_train`
(`fedbrew/data/manifest_validation.py`). Preflight then prints a note on every
run against that data saying its `test_*` and `central_test_*` numbers are
training numbers. The alternative is a table of test metrics that only a
README explains.

## 5. A new model

Smaller than the other two.

1. Add a builder — under `fedbrew/models/` in the package, or in your module.
2. Declare `_KNOWN_KEYS` and call `reject_unknown_model_keys` with it. Without
   this a misspelled key is dropped and the default runs — the defect that made
   `lora_alph: 32` train at 16. This is the model-block equivalent of §1.3, and
   it is the same for both origins: `model` keys travel as `model.extra` and
   the builder is what checks them.
3. Register it with `task=` naming the task adapter it needs. `MODEL_TASKS`
   (`fedbrew/core/registry.py`) is a view filled from that, so a model cannot be
   registered without its task.
4. Nothing else pairs them: a config does not restate the task. `experiment.task`
   used to, and is now refused by name with a message pointing at `model.name`.
5. Add its key table to chapter 06 if it is in the package. The guard diffs it
   against `_KNOWN_KEYS` per built-in builder, so a missing or invented key fails.

If it changes **what gets federated** — as `hf_causal_lm_lora` does — set
`model_state_scope` in the metadata. The server validates the scope on every
incoming result, so a mismatch is refused rather than silently averaged. Only
the rules that go through the task's state hooks can train it (chapter 07
§5.1): `fedprox`, `scaffold` and `fedlalr` refuse an adapter-scoped task before
their first update, and a model the package should refuse at load belongs in
`ADAPTER_SCOPED_MODELS` (`fedbrew/core/config.py`).

## 6. A new task or dataset backend

Rarer than the others, and both are one class each.

**A task** — a new modality beside `classification` and `causal_lm` — subclasses
`TaskAdapter` (`fedbrew/tasks/base.py`) and implements `build_model`,
`build_dataloader`, `train_step`, `eval_step` and `compute_metrics`, plus the
four federated-state methods if what leaves the client is not the whole model.

`train_step`'s `optimizer` argument is optional in the signature and is not
optional in either shipped task. Every rule in `fedbrew/clients/` passes one,
and both tasks raise when it is omitted rather than building a default. An
optimizer built per call carries no state and no configuration: an Adam-family
rule resets its moments every batch and runs as sign-SGD at whatever rate the
default named, and even a stateless SGD takes the default's learning rate,
momentum, weight decay and nesterov flag instead of the client's. Refuse it in
your task too; a run that silently trains under a rule nobody configured is the
failure both refusals exist to prevent.
Register it in `tasks`, and register every model that needs it with `task=`
naming it. Chapter 06 §3. Its constructor is handed `EXTENSION_TASK_KEYS`
(§1.4) when it comes from an extension.

Implement `evaluate_model(model, data)` as well unless you want no central test
set. It is not abstract — `eval_step` is per batch and required, this scores a
whole dataset the caller already holds — and it is the optional capability
`SupportsDatasetEvaluation` declares. `FedAvgServer.evaluate_global` and
`ScaffoldServer.evaluate_global` check for it and return no metrics without it,
so a task that omits it produces **no `central_test_*` column at all** and
nothing in the run says why. Both shipped tasks implement it in three lines:
build an eval dataloader, `eval_step` each batch, `compute_metrics` the
outputs.

Override `train_loss_denominator(batch, output)` if your training loss is not
a mean over the batch's examples. It returns the count `train_step`'s loss
averages over, and `update_mode: full_gradient` weights each batch's gradient
by its share of it, which is what makes that mode the exact gradient of the
whole split. The default is the batch's example count, right for
classification's cross-entropy and for every shipped example's objective; the
causal-LM task returns the active target tokens its `train_step` reports as
`total`. A task that averages over tokens and keeps the default gets a
`full_gradient` step that is not the gradient of anything it reports, and
nothing fails: `tests/test_full_gradient.py` measures the difference.
Overriding it also takes `frozen_batch_gradients` away from your task. That
mode weights by examples, so the engine refuses it for any task that overrides
this method, before the first update (FINDINGS.csv `POST-F19`).

Override `federated_aggregation_weight(training_outputs, evaluated_num_examples)`
if your client's weight should be something other than the default: the count
of the post-fit evaluation pass over the whole train split, which is the
split's size in whatever `eval_step`'s `total` counts. `training_outputs` are
what `train_step` returned this round, one per batch the update ran, so a
weight built from them counts exposures and moves with the update mode.
`fedprox` and `scaffold` do not call it, so a task whose weight the hook
decides is refused with them where it can be told: the causal-LM task's
`active_target_weighting` (FINDINGS.csv `POST-F30`). Chapter 07 §3.1 defines
what each built-in task reports.

A task chooses its own metric names, and the columns a run writes are the ones
it returns. What it cannot choose is what the plan header *predicts*: the
header lists the columns a run of this shape usually produces, from the metric
names rather than from the task, so a task that writes no accuracy is still
promised an `accuracy` column before it starts. The run record is right and the
prediction is wrong; drift-quad's README says so where its tables are.

**A dataset backend** — a third beside `synthetic_classification` and
`manifest_dataset` — subclasses `FederatedDataset` (`fedbrew/data/dataset.py`)
and implements `list_clients`, `get_client_data`, `get_client_metadata`,
`get_global_data` and `get_metadata`. Whether it gets the lazy client pool
follows from the dataset rather than from a config key, so consider whether
building a client is expensive enough to want it. Chapter 01 §7.

Prefer a generator (§4) to a backend when the data can be written once. A
backend that generates on the fly puts the data outside `fedbrew generate`,
which means outside the manifest, which means outside `run.json` — and the
one-way rule that `run` only ever reads a manifest is what makes a run's data
identifiable at all. drift-quad started as a backend and became a generator for
exactly this reason.

## 7. A new partitioner

**In the package only.** Partitioning is dispatched by name inside
`fedbrew/data/generate.py`, and there is no registry for it — an out-of-tree
problem that partitions its own data does so inside its generator, and records
what it did as `partition_strategy` in the manifest.

1. Add a module under `fedbrew/data/partitioners/` returning
   `dict[client_id, list[int]]` over **training indices only**.
2. Dispatch it in `_partition_train_indices` (`fedbrew/data/generate.py`).
3. Declare its keys in the `partition` section of `_GENERATOR_SECTION_KEYS`.
4. **Teach `partition_test_indices_like_train` how to mirror it**
   (`fedbrew/data/official_test_partitioning.py`) — otherwise the official test
   set is dealt out on a profile that does not match the training heterogeneity,
   and the per-client statistics measure clients on a distribution they never
   saw. Chapter 05 §4.1.
5. Add disjointness and coverage tests, as
   `tests/test_partition_disjointness.py` and
   `tests/test_label_skew_coverage.py` do.

## 8. A new metric

**In the package only**, and the tightest extension point, because the metric
surface is name-driven throughout.

**A per-client base metric** — a third alongside `loss` and `accuracy` — means
adding it to `CLIENT_METRIC_BASES` (`fedbrew/core/config.py`). That changes
every aggregate column set, the `client_metrics.csv` schema and the checkpoint
validation at once. It is not a local change.

**A new cross-client statistic** — a suffix alongside `_std`, `_min`,
`_worst{P}` — must be added in **two** places that read the same tuple:
`client_metric_names` (`fedbrew/core/config.py`) and
`_client_distribution_statistics` (`fedbrew/core/loop.py`) — two files, which the
single line range this used to cite could not say. Adding it to one only means
checkpoint validation accepts
a metric no run produces, and `best.pt` is missing at the end of a run rather
than at its start.

Make it NaN-hostile in the same way as its neighbours: a non-finite input must
produce `NaN` in the new column without changing the column set, because a
round that dropped columns would change the CSV schema partway through a run.

**An algorithm diagnostic** is easier, and is the one an extension can do: emit
it from your client or server and list it in `client.metrics` or
`server.metrics`. Decide deliberately whether it goes before or after
`filter_metrics` — chapter 08 §7 documents that the two built-in strategies
with diagnostics add theirs after it, so no metrics list can drop them.

**Anything used for checkpoint selection** must be a `val_` or `personal_val_`
metric, and its name must contain a word from one of the two direction
frozensets in `checkpointing.py` — otherwise it is rejected at config load with
a message saying so. This holds for an extension's metrics too, because the
rule reads the name and not the component.

## 9. What a change must not break

Before opening a change, the invariants that cut across every extension point:

| Invariant | Chapter |
| --- | --- |
| Aggregation stays streaming — two model states in memory, never one per participant | 01, 11 |
| Artifacts are flushed every round | 09 |
| An unread config key is an error | 04 |
| A config names every component a run is built from | 04, 09 |
| `fedbrew run` reads a manifest and never writes one | 05 |
| A registered built-in name has documentation with a guard | 07, 06, 05 |
| The column set does not change partway through a run | 08 |
| A performance change must not change the numbers | 11 |
| Every `fedbrew ...` command and flag written anywhere is real | 02 |
| Every `docs/` path written anywhere resolves | 00 |

## For agents

### Paths

| Path | The extension point it serves |
| --- | --- |
| `fedbrew/core/extensions.py` | `load_extensions`, `LoadedExtension`, `REGISTER_HOOK` — the out-of-tree loader |
| `fedbrew/core/registry.py` | the six registries, `register_builtin_components`, `registering_from`, `BUILTIN`, `MODEL_TASKS`, `GeneratorRegistry`, `GeneratorSpec` |
| `fedbrew/core/factory.py` | `is_extension`, `EXTENSION_TASK_KEYS`, `EXTENSION_DATASET_FIELDS`, `EXTENSION_CLIENT_KEYS`, and forwarding config keys to constructors |
| `fedbrew/core/config.py` | `_KNOWN_EXTRA_KEYS`, `CLIENT_METRIC_BASES`, `client_metric_names` |
| `fedbrew/core/validation.py` | pairing rules, ranges, cost notices, extension neutrality |
| `fedbrew/servers/base.py` | new server strategy |
| `fedbrew/clients/base.py` | new client update rule |
| `fedbrew/tasks/base.py` | new task adapter, and the federated-state group |
| `fedbrew/data/dataset.py` | new dataset backend |
| `fedbrew/data/generate.py` | generator dispatch, `_GENERATOR_SECTION_KEYS`, `_partition_train_indices` |
| `fedbrew/data/partitioners/` | new partitioner |
| `fedbrew/data/official_test_partitioning.py` | mirroring a partition onto the official test set |
| `fedbrew/models/config_keys.py` | `reject_unknown_model_keys` |
| `fedbrew/core/loop.py` | `_client_distribution_statistics` |
| `fedbrew/core/checkpointing.py` | selection-metric rules |
| `examples/drift-quad/problem.py` | the worked out-of-tree case: generator, task, model, one `register()` |

### Commands

```bash
# Prove this chapter's extension points and file paths are current, and that
# every excerpt above is still a verbatim quote of the file it names.
python -m pytest tests/test_docs_extending.py -v

# The out-of-tree path, end to end, as chapter 12 §2 writes it.
fedbrew generate --config data/configs/examples/drift-quad.yaml
fedbrew run --config configs/examples/drift-quad/fedavg.yaml --validate-only

# After any extension: the guards that catch a half-registered component.
python -m pytest tests/test_unknown_config_keys.py \
                 tests/test_registry_origins.py \
                 tests/test_extensions_loader.py \
                 tests/test_cli_commands_exist.py \
                 tests/test_docs_references_resolve.py \
                 tests/test_shipped_config_explicitness.py

# Then the whole suite, and the documentation guards it now includes.
python -m pytest
```

### Invariants

1. **Register the name, declare the keys, forward the keys.** All three. In the
   package a key declared but not forwarded is the worst of the three failures;
   out of tree `config_keys=` makes the two one act, which is the reason to
   prefer it.
2. **An extension registers nothing at import.** `register()` is the only
   entry, and it is called once per resolved location.
3. **A branch on origin, never on a name.** `is_extension` exists so that
   nothing in the package needs a list of what is not in the package.
4. **A new suffix goes in `client_metric_names` and
   `_client_distribution_statistics` together.** They read the same tuple for
   this reason.
5. **A new partitioner must teach the test-set mirror how to follow it.**
6. **A new model builder declares `_KNOWN_KEYS` and calls the rejector**, and
   registers with `task=`.
7. **Auxiliary state travels in `payload`**, and its bytes go into
   `communicated_bytes`.
8. **A component with no personal model rejects a non-global `model_scope`.**
9. **Every new registered built-in name needs prose in its chapter.** The
   chapter guards diff against the registries' `builtin()` view, so an
   undocumented built-in fails the suite rather than shipping unfindable — and
   an extension loaded by a test does not.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_extending.py` | Every extension point named here exists, every path resolves, every excerpt is verbatim, and §2's two commands run. |
| `tests/test_registry_origins.py` | Every name carries its origin; a duplicate names both. |
| `tests/test_extensions_loader.py` | Both entry forms load, once per location, and a bad one fails clearly. |
| `tests/test_extensions_config.py` | Both `extensions` keys load before any name is checked. |
| `tests/test_extension_components.py` | The three factory contracts are what this chapter says. |
| `tests/test_examples_are_extensions.py` | The migrated example reaches the CLI through the hook, and importing it registers nothing. |
| `tests/test_unknown_config_keys.py` | An undeclared key is rejected. |
| `tests/test_docs_algorithms.py` | Every registered built-in strategy and rule has prose. |
| `tests/test_docs_model_keys.py` | Every builder's key table matches its `_KNOWN_KEYS`. |
| `tests/test_docs_data_keys.py` | Every generator's sections match its allow-list. |
| `tests/test_docs_metric_names.py` | Every aggregate column matches `client_metric_names`. |
| `tests/test_partition_disjointness.py` | A new partitioner's output does not overlap. |
| `tests/test_client_communication_cost.py` | Reported volume matches what moved. |
| `tests/test_shipped_config_explicitness.py` | A new config states the settings that change numbers. |

### Known failure modes

- **Registering at import instead of in `register()`.** The module then has a
  side effect, and importing it for a constant registers its names.
- **Adding a `.get()` without touching `_KNOWN_EXTRA_KEYS`.** Rejected at load.
- **Adding to `_KNOWN_EXTRA_KEYS` without forwarding in the factory.** Loads
  fine, does nothing, says nothing. Nothing catches this automatically. Out of
  tree, `config_keys=` makes it impossible.
- **Declaring `config_keys` on `tasks`, `models` or `generators`.** Raises:
  those registries have no config section.
- **Declaring a generator section by name alone, from outside the package.**
  Refused on the first `generate`, naming the mapping form; §4 step 3. The
  name-only form means "the keys are in `_GENERATOR_SECTION_KEYS`", and yours
  are not.
- **Adding a metric suffix to `client_metric_names` only.** Checkpoint
  validation accepts a metric no run emits, and `best.pt` is missing at the
  end of the run.
- **Registering a name that already exists.** `register` raises, naming both
  origins.
- **Overriding `aggregate` but not `aggregate_stream`.** It works, and
  silently buffers one model copy per selected client.
- **Adding a partitioner without the test-set mirror.** The per-client test
  statistics then measure clients on a distribution unlike their own.
- **Shipping a config under `configs/` without `matmul_precision` or
  `deterministic`.** Guarded; the suite fails.
- **Writing a private runner because the CLI cannot see your component.** It
  can; §1. A runner that imports the loop skips `--validate-only`, the plan
  header, every config guard and the whole of `run.json`.
