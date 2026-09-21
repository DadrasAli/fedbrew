# 06 — Models and tasks

The eight model builders, the exact key set each accepts, and the two tasks
that define what a model is asked to do.

A model config key that no builder reads is an **error**, not a no-op. This
chapter's key tables are the readable form of the `_KNOWN_KEYS` frozenset each
builder declares.

## 1. Why the key sets are strict

Every builder resolves its parameters with `values.get(name, default)`, so a
misspelled key used to be dropped by the dict and replaced by the default:
`lora_alph: 32` ran at `16` with nothing to say so.

The builders are the only place that knows which names are real, so each one
declares its set and calls `reject_unknown_model_keys`
(`fedbrew/models/config_keys.py`).

Four keys are **injected into every builder** by
`factory._model_config` and never need declaring — `_INJECTED_KEYS` and
`_INJECTED_PREFIX` (`fedbrew/models/config_keys.py`):

```
name  input_dim  hidden_dim  num_classes
```

For `causal_lm`, manifest metadata is also injected under the `dataset_`
prefix — `dataset_sequence_length`, `dataset_ignore_index` and the tokenizer
fields. A config cannot set those; they come from the data.

## 2. The eight builders

| `model.name` | Task | Module | Keys |
| --- | --- | --- | --- |
| `mlp` | classification | `fedbrew/models/torch_mlp.py` | 4 |
| `cnn` | classification | `fedbrew/models/torch_cnn.py` | 3 |
| `small_cnn` | classification | `fedbrew/models/torch_cnn.py` | 3 |
| `femnist_resnet18` | classification | `fedbrew/models/femnist_resnet.py` | 5 |
| `openimage_shufflenet` | classification | `fedbrew/models/openimage_shufflenet.py` | 8 |
| `tiny_gpt2` | causal_lm | `fedbrew/models/tiny_gpt2.py` | 15 |
| `hf_causal_lm` | causal_lm | `fedbrew/models/hf_causal_lm.py` | 7 |
| `hf_causal_lm_lora` | causal_lm | `fedbrew/models/hf_causal_lm_lora.py` | 13 |

`cnn` and `small_cnn` share a module and a key set; they differ in width.

### 2.1 `mlp`

| Key | Default |
| --- | --- |
| `input_dim` | `5` |
| `hidden_dim` | `16` |
| `num_classes` | `2` |
| `dropout` | `0.0` |

The three dimensions are also injected from the dataset, so a config usually
sets none of them.

### 2.2 `cnn` and `small_cnn`

| Key | Default |
| --- | --- |
| `input_channels` | `3` |
| `hidden_dim` | `128` |
| `num_classes` | `10` |

### 2.3 `femnist_resnet18`

| Key | Default | Notes |
| --- | --- | --- |
| `input_channels` | `1` | FEMNIST is greyscale |
| `base_channels` | `32` | |
| `blocks_per_stage` | `(2, 2, 2, 2)` | four stages, ResNet-18 shape |
| `group_norm_groups` | `8` | GroupNorm, not BatchNorm — see §4; honoured at every width here |
| `dropout` | `0.1` | |

`num_classes` is injected; it is `62` for FEMNIST (10 digits + 52 letters).

### 2.4 `openimage_shufflenet`

The model is here and its run config is `configs/openimage/fedavg.yaml`, but
there is no OpenImage generator, so the dataset is not producible from this
repository: chapter 05 §2.

| Key | Default | Notes |
| --- | --- | --- |
| `input_channels` | `3` | |
| `stem_channels` | `24` | |
| `stage_channels` | `(116, 232, 464)` | **every value must be even** — see below |
| `blocks_per_stage` | `(4, 8, 4)` | must be the same length as `stage_channels` |
| `final_channels` | `1024` | |
| `group_norm_groups` | `8` | a request; reduced to 2 and 4 in the narrow branches — see §4.3 |
| `stem_pool` | `False` | |
| `dropout` | `0.1` | |

**An odd stage width is refused, not rounded.** A ShuffleNet unit splits its
input and builds both branches at `output_channels // 2`, then concatenates, so
an odd width would silently produce `output_channels - 1` channels. The builder
raises instead, in `ShuffleUnit.__init__` (`fedbrew/models/openimage_shufflenet.py`),
guarded by
`tests/test_shufflenet_channel_parity.py`.

### 2.5 `tiny_gpt2`

A small GPT-2 built from config, for testing the causal-LM path without a
download.

| Key | Default | Notes |
| --- | --- | --- |
| `vocab_size` | `258` | |
| `sequence_length` | `32` | |
| `n_embd` | `64` | must be divisible by `n_head` |
| `n_layer` | `2` | |
| `n_head` | `2` | |
| `n_positions` | `sequence_length` | raised to `sequence_length` if set lower |
| `dropout` | `0.0` | the fallback for the three below |
| `resid_pdrop` | `dropout` | |
| `embd_pdrop` | `dropout` | |
| `attn_pdrop` | `dropout` | |
| `layer_norm_epsilon` | `1e-5` | |
| `initializer_range` | `0.02` | |
| `bos_token_id` | `1` | |
| `eos_token_id` | `1` | |
| `pad_token_id` | `0` | **from the manifest for causal_lm**; a contradicting value is refused, and `null` means no padding token rather than 0 — §3.3 |

The three dropout probabilities default to `dropout`, so setting that one key
sets all three. `n_positions` is `max(sequence_length, configured)`, so a
config cannot build a model with a context shorter than its own sequences.

### 2.6 `hf_causal_lm`

A pinned Hugging Face causal LM loaded from an offline snapshot.

| Key | Default | Notes |
| --- | --- | --- |
| `preparation_config` | — | path to the `configs/llm_assets/` config that prepared the snapshot |
| `asset_manifest` | — | the manifest the preparation wrote |
| `sequence_length` | — | |
| `vocab_size` | — | |
| `pad_token_id` | — | **from the manifest for causal_lm**; a contradicting value is refused — §3.3 |
| `local_files_only` | `True` | any other value is refused |
| `trust_remote_code` | `False` | any other value is refused |

The last two are assertions rather than options: the builder raises on any
other value — `_require_strict_offline_flags` (`fedbrew/models/hf_causal_lm.py`).
They are in the key set so that writing
them is legal and consistent, not so that they can be changed.

### 2.7 `hf_causal_lm_lora`

`hf_causal_lm` plus a LoRA adapter. It accepts all seven keys above, plus six:

| Key | Default | Notes |
| --- | --- | --- |
| `r` | `8` | LoRA rank |
| `lora_alpha` | `16` | |
| `lora_dropout` | `0.05` | |
| `target_modules` | — | which projections get an adapter; required, stripped and de-duplicated |
| `bias` | `'none'` | the only accepted value; anything else would make frozen base-model bias tensors trainable |
| `adapter_name` | `'default'` | |

Those defaults have one home, `lora_config_from_model_values`
(`fedbrew/models/hf_causal_lm_lora.py`), and `run.json`'s `lora_config` block
is that function's return value rather than a second reading of the same keys.
It used to be the second reading, with its own copies of `8`, `16`, `0.05` and
`'none'`, so a default changed here would have left the record describing an
adapter that never ran.

The base model is built by `hf_causal_lm`, which runs the same strictness
check. Passing the whole mapping down would make the inner builder reject the
outer one's keys, so `forwarded_model_keys` narrows it at the boundary by the
same rules — `forwarded_model_keys` (`fedbrew/models/config_keys.py`).

**This is the only builder that changes what gets federated.** It sets
`model._fl_model_state_scope = "adapter"` in `build_hf_causal_lm_lora`
(`fedbrew/models/hf_causal_lm_lora.py`), and only
the adapter tensors move each round. §3.4.

## 3. The two tasks

`task.name` selects one. A task decides what a batch is, what the loss is, and
what leaves the client.

| | `classification` | `causal_lm` |
| --- | --- | --- |
| Module | `fedbrew/tasks/classification/torch_classification.py` | `fedbrew/tasks/causal_lm/torch_causal_lm.py` |
| Batch | `(features, targets)` | token ids and labels |
| Loss | `nn.CrossEntropyLoss()`, mean over examples | cross-entropy over **active tokens** |
| Accuracy | top-1 over `argmax(dim=1)` | next-token top-1 over active tokens |
| Counted in | examples | tokens |
| AMP | `use_amp` supported | **refused** — no autocast path, so `use_amp: true` is an error rather than a no-op |
| `fast_batching` | supported | not applicable |

Chapter 08 §2 gives both formulas.

**`runtime.use_amp: true` on `causal_lm` is refused, not ignored.**
`TorchCausalLMTask` takes no `use_amp` and its `train_step` has no `autocast`
or `GradScaler`, and the factory passed the flag to the classification task
only. So an LLM config could set it, pass validation, be echoed into
`run.json`, and run in fp32 throughout — the numbers right, the record wrong.
`AMP_AWARE_TASKS` in `factory.py` is now the one place that says which tasks
implement it: the factory reads it to decide both what to pass and what to
refuse, and `_validate_task` reports the same fact as a preflight error. An
extension task is never in the set, which matches `EXTENSION_TASK_KEYS` not
carrying `use_amp` either. Refused rather than implemented — adding autocast
here is a feature and would move every LLM number, and every shipped LLM
config already sets `use_amp: false`.

### 3.1 What "active" means

For `causal_lm`, a token counts only if its target is neither `ignore_index`
(default `-100`, covering padding and, under SFT, the prompt) nor
`pad_token_id` when one is set — `TorchCausalLMTask._loss_and_counts`
(`fedbrew/tasks/causal_lm/torch_causal_lm.py`). Everything —
loss, accuracy, `active_target_tokens` — is measured over that subset.

That second filter matches on the token's **value**, not on its position, so a
`pad_token_id` that is also a real vocabulary token takes every genuine
occurrence of that token out of the measurement with it. The task refuses the
one case where that happens — §3.3.

This is why `num_examples` means different things per task, and why
`federated_aggregation_weight` exists: it lets a task decide a client's weight
rather than assuming an example count. For `causal_lm` it is active target
tokens either way; `model.active_target_weighting` decides which ones — those
in the client's whole train split (`false`, the default), or those in the
batches the round trained on, counted once per step (`true`, the default when
the manifest's task is `causal_lm_sft`). Chapter 07 §3.1 has the full
definition, per task and per update mode.

### 3.2 Evaluation batch size differs by task

`client.eval_batch_size` defaults differently — `_eval_batch_size`
(`fedbrew/core/factory.py`):

| Task | Default |
| --- | --- |
| `classification` | `max(client.batch_size, 256)` |
| `causal_lm` | `client.batch_size` |

A causal-LM forward produces `batch × sequence_length × vocabulary` logits,
which reaches tens of gigabytes at the classification default. Evaluating at
the training batch size is the only safe default there.

### 3.3 `pad_token_id` comes from the manifest for causal_lm

Both `tiny_gpt2` and `hf_causal_lm` accept `pad_token_id`, and for `causal_lm`
the manifest's value is used: the padding token is a property of the tokenizer
that produced the data, not a free choice. Leaving it unset is the normal case.

**A manifest that declares no padding token gives the task none.** The task's
masking (§3.1) and its attention mask are both switched off, rather than
falling back to a token id. They used to default to `0`, which is
`tiny_causal_lm`'s padding id and an ordinary word everywhere else — `"!"` on
the shipped Qwen2.5 vocabulary — so a dataset that never padded had every id-0
target dropped from the loss, the accuracy and the aggregation weight, and
every id-0 input position zeroed in the attention mask, on the strength of a
default it had not asked for. The `0` in `tiny_gpt2`'s key table above is that
builder's own `GPT2Config` default and is unrelated; all three shipped
generators write `padding_token_id` explicitly, so none of them changed.

Both causal-LM builders take `null` for the key as well, and mean the same
thing by it. `hf_causal_lm` always did; `tiny_gpt2` read it through `int()` and
died with a bare `TypeError` inside the model factory, which an SFT manifest
reaches by writing `padding_token_id: null` and letting the factory copy it
over. No shipped config pairs `tiny_gpt2` with an SFT dataset, so nothing
failed — and nothing said the pairing was unsupported either.

A config that sets it to something **different** is refused
(`factory._add_causal_manifest_metadata`), the same way `model.sequence_length`
and `model.vocab_size` are refused when they contradict the manifest
(`hf_causal_lm._expected_sequence_length`, `_expected_vocab_size`). It used to
be overwritten in silence, which was the odd one of the three: a wrong sequence
length or vocabulary size fails loudly further on, whereas a wrong padding token
masks the wrong positions and produces a number.

A padding id equal to the manifest's `eos_token_id` is refused too, by the task
rather than the factory (`TorchCausalLMTask.__init__`). Because §3.1's filter
matches by value, such an id removes every real end-of-document target from the
loss, from the accuracy, and from the active-token count that becomes the SFT
aggregation weight — the model is never trained to emit EOS, and no metric
disagrees, since all three are measured over the same surviving positions.

That refusal is the last line rather than the first. `hf_causal_lm_text` used
to produce exactly such a manifest, by two routes: it read the tokenizer's own
`pad_token_id` before checking whether padding was enabled at all, and it fell
back to the EOS id when the tokenizer had no distinct padding token — which is
the norm for GPT-2- and Qwen-style tokenizers, and Qwen2.5 ships
`pad_token == eos_token` outright. Both are closed at the generator now
(`_padding_token_id`):

- **`pad_incomplete_window: false` records no padding token.** Every window is
  full, the dataset contains no padding, and a padding id in its manifest would
  be a filter over real tokens rather than a description of the data. This is
  the default and what the shipped dev config sets.
- **`pad_incomplete_window: true` with an EOS-valued padding token is refused
  where the config can still be changed** — turn padding off and drop the
  trailing partial window, or use a tokenizer whose padding token is distinct.
  Writing the manifest and letting the task reject it hours later would be a
  generation that reported success and a dataset no run could load.

A dataset that records no padding token at all is the other way to be safe, and
is what both SFT generators write: `padding_token_id: null`.

### 3.4 Model state scopes

What leaves a client each round.

| Scope | Set by | What moves |
| --- | --- | --- |
| `full` | the default `TaskAdapter` | every tensor in the model state |
| `adapter` | `hf_causal_lm_lora` | the LoRA tensors only |

`federated_model_state_metadata` reports the scope alongside
`total_parameters`, `trainable_parameters`, `communicated_parameters` and
`communicated_bytes` — `TaskAdapter` (`fedbrew/tasks/base.py`). The server
validates the scope on
every incoming result, so a full-model client cannot be aggregated into an
adapter-scoped run. Not every client rule trains the `adapter` scope:
`fedprox`, `scaffold` and `fedlalr` are refused with it, and chapter 07 §5.1
lists the combinations that train it.

Under LoRA, `trainable_parameters` and `communicated_parameters` are a small
fraction of `total_parameters`; that ratio is what the method is for, and both
numbers reach `client_update_metrics.csv` — chapter 08 §4.2.

**Tied weights are handled explicitly.** A model whose input embedding and
output head share a tensor would otherwise contribute it twice to the
aggregate; `tests/test_tied_weight_federation.py` guards that it does not.

## 4. Normalisation: GroupNorm, not BatchNorm

`femnist_resnet18` and `openimage_shufflenet` both use GroupNorm.

BatchNorm keeps running mean and variance buffers that are part of the model
state, so they get averaged across clients like parameters. Under non-IID
partitioning each client's batch statistics describe a different distribution,
and the average of them describes none of them. GroupNorm has no running
statistics, so the problem does not arise.

### 4.1 BatchNorm is refused, not merely discouraged

The paragraph above reads as a modelling preference. It is not the whole
story: **aggregation raises on a BatchNorm model**, so bringing one is a
crash rather than a worse number.

`BatchNorm*` registers a third buffer, `num_batches_tracked`, an `int64`
counter incremented once per training forward pass.
`WeightedStateAccumulator` averages floating tensors and requires every
non-floating one to be **bit-identical across clients**
— `_accumulation_dtype` (`fedbrew/core/torch_utils.py`) — because there is no
meaningful mean of a counter.
Two clients that took different numbers of optimizer steps therefore produce:

```
ValueError: non-floating state tensor '1.num_batches_tracked' differs between clients
```

Different step counts are the normal case, not the edge case: `local_iterations`
full passes over clients of unequal size is unequal steps by construction. The
failure is **mid-round, during aggregation** — after every participating
client has run its local iterations — and it recurs every round.

One honest qualification, because the accumulator's refusal is narrower than
"BatchNorm does not work": **the refusal is about non-floating buffers, not
about BatchNorm.** Any model carrying an integer buffer that advances during
training hits it; a pretrained checkpoint with a step counter is the other
common case. `tests/test_adamw_and_model_state_compat.py` covers the general
check with a synthetic `step` buffer.

### 4.2 The half of it the accumulator could not see

The paragraphs above describe the loud half. The accumulator is handed tensors
and not a model, so it cannot tell a buffer from a parameter, and its handling
of the two is one-sided by accident:

| Buffer | What the accumulator does | Loud? |
| --- | --- | --- |
| `num_batches_tracked` (int64) | requires bit-identical values across clients | **yes** |
| `running_mean`, `running_var` (float32) | adds into the running weighted sum, like a weight | **no** |

So every client at the same step count — `update_mode: single_batch`, or equal
shard sizes with `drop_last: true`, or `max_local_steps` — makes the counters
agree, the check passes, and the running statistics are averaged across
distributions that do not match. On a FedAdam, FedYogi or FedAdagrad arm the
server then applies an adaptive update to running statistics. Nothing said so.

**`factory.build_components` now refuses instead**, once per run, before the
first round:

```
ValueError: these federated state keys are registered buffers, not parameters:
norm.num_batches_tracked, norm.running_mean, norm.running_var. Averaging them
is not defined: ...
```

The predicate is `torch_utils.persistent_buffer_keys` — state-dict keys that
are neither parameters nor tied aliases of one. Both exclusions matter, and
both were measured against the models this repository ships:

| Model | `named_buffers()` | In the state dict | `persistent_buffer_keys()` |
| --- | --- | --- | --- |
| the five classification builders | — | — | `[]` |
| `tiny_gpt2` | `attn.bias`, `attn.masked_bias` | `lm_head.weight` (tied to `wte.weight`) | `[]` |
| Qwen2 | `rotary_emb.inv_freq` | — | `[]` |

A check written on `named_buffers()` would refuse both LLM models, whose
buffers are non-persistent and never reach the wire; one that skipped the
tied-alias exclusion would refuse GPT-2, whose `lm_head.weight` is one
parameter under two names. With both exclusions every shipped model returns an
empty list, so the refusal fires only on a model that does not exist yet.

It is scoped to the **federated** state rather than the model, so
`hf_causal_lm_lora` — which federates adapter tensors only — is judged on what
it sends and not on what the frozen base model happens to register.

So: no shipped model uses BatchNorm, no test trains one end to end, and there
is no support for one. `tests/test_batchnorm_is_unsupported.py` pins the
refusal against a real `nn.BatchNorm2d` and against this section, so the
chapter cannot drift back to describing it as a preference.

### 4.3 `group_norm_groups` is a request, and half the ShuffleNet does not get it

`group_norm_groups` is `8` on both models. `nn.GroupNorm` requires
`num_channels % num_groups == 0`, so one number cannot serve widths that are
not all multiples of it: the request is capped at the channel count and walked
down to the nearest divisor.

On `femnist_resnet18` that never bites — its widths are `32, 64, 128, 256`, all
multiples of 8, so all 20 layers get 8. On `openimage_shufflenet` it bites over
most of the network:

| channels | 24 | 58 | 116 | 232 | 1024 |
| --- | --- | --- | --- | --- | --- |
| groups built | 8 | **2** | **4** | 8 | 8 |
| layers | 2 | 13 | 26 | 14 | 1 |

39 of its 56 normalisation layers run at 2 or 4 groups. Refusing instead of
backing off would make `8` illegal for that model — the only counts dividing
every width above are 1 and 2, which is a different model from the one this
config describes — so the backoff stays and the run records what it did:
`run.json` carries `federated_model_state.group_norm_reductions`, one entry per
`(channels, requested, groups)` with the layer count, and no entry at all when
every request was honoured. P10-F32.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/models/config_keys.py` | `reject_unknown_model_keys`, `forwarded_model_keys`, `_INJECTED_KEYS` |
| `fedbrew/models/torch_mlp.py` | builds `mlp` |
| `fedbrew/models/torch_cnn.py` | builds `cnn` and `small_cnn` |
| `fedbrew/models/femnist_resnet.py` | builds `femnist_resnet18` |
| `fedbrew/models/openimage_shufflenet.py` | builds `openimage_shufflenet`, and the even-width check |
| `fedbrew/models/tiny_gpt2.py` | builds `tiny_gpt2` |
| `fedbrew/models/hf_causal_lm.py` | builds `hf_causal_lm` |
| `fedbrew/models/hf_causal_lm_lora.py` | builds `hf_causal_lm_lora`, and the adapter scope |
| `fedbrew/tasks/base.py` | `TaskAdapter` and the federated-state group |
| `fedbrew/tasks/classification/torch_classification.py` | the classification task |
| `fedbrew/tasks/causal_lm/torch_causal_lm.py` | the causal-LM task, `_loss_and_counts` |
| `fedbrew/core/factory.py` | `_add_causal_manifest_metadata` (the `pad_token_id` rule) and the eval-batch defaults |

Each builder's `_KNOWN_KEYS` frozenset is the authority on its key table.

### Commands

```bash
# Prove every key table here is the builder's own _KNOWN_KEYS.
python -m pytest tests/test_docs_model_keys.py -v

# The behavioural guards.
python -m pytest tests/test_batchnorm_is_unsupported.py \
                 tests/test_shufflenet_channel_parity.py \
                 tests/test_tied_weight_federation.py \
                 tests/test_lora_adapter_federation.py \
                 tests/test_causal_lm_task_model.py \
                 tests/test_pad_token_id_is_not_eos.py \
                 tests/test_causal_lm_train_step_requires_an_optimizer.py \
                 tests/test_classification_train_step_requires_an_optimizer.py \
                 tests/test_tiny_gpt2_accepts_a_null_padding_token.py \
                 tests/test_amp_is_refused_where_it_is_ignored.py \
                 tests/test_eval_batch_size_defaults.py \
                 tests/test_femnist_support.py

# Check a model config without training.
fedbrew run --config configs/dev/tiny_causal_lm.yaml --validate-only
```

### Invariants

1. **Each builder's `_KNOWN_KEYS` is the authority on its key set.** A new
   option must be added there or the builder rejects it. Adding the `.get()`
   is not enough.
2. **An unknown model key is an error.** That is the whole reason the check
   exists: `lora_alph: 32` silently ran at `16`.
3. **`forwarded_model_keys` narrows at the LoRA/base boundary.** Do not pass a
   builder's whole mapping to an inner builder; it will reject the outer keys.
4. **`local_files_only` and `trust_remote_code` are assertions.** The builder
   raises on any other value. They are not knobs.
5. **An odd ShuffleNet stage width raises.** Never round it; a silent
   `output_channels - 1` is what the check exists to prevent.
6. **`pad_token_id` comes from the manifest for causal_lm.** A config value
   that contradicts it is refused, as `sequence_length` and `vocab_size` are.
   A manifest whose padding token *is* its EOS token is refused as well: the
   loss mask matches by value and would eat every end-of-document target. A
   manifest that declares none leaves the task with none — never token 0. §3.3.
7. **GroupNorm, not BatchNorm, in the federated vision models.** Running
   statistics averaged across non-IID clients describe no client — and
   `num_batches_tracked` makes it a mid-round `ValueError` rather than a worse
   number, whenever two participating clients took different step counts. §4.1.
8. **A scope change is a protocol change.** `model_state_scope` must be
   reported in the metadata and is validated by the server on every result.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_model_keys.py` | Every per-builder key table equals that builder's `_KNOWN_KEYS`, and the injected-key list matches `config_keys.py`. |
| `tests/test_shufflenet_channel_parity.py` | An odd stage width is refused. |
| `tests/test_tied_weight_federation.py` | A tied weight is not counted twice. |
| `tests/test_lora_adapter_federation.py` | Adapter-scoped state federates without the base model. |
| `tests/test_lora_reporting.py` | The LoRA parameter counts are reported. |
| `tests/test_causal_lm_task_model.py` | The causal-LM task's model contract, and §3.3: token 0 is padding when the dataset declares it and a real token when it does not. |
| `tests/test_causal_lm_model_reuse.py` | `reuse_model` reaches the causal-LM task. |
| `tests/test_pad_token_id_is_not_eos.py` | §3.3: a padding token equal to the dataset's EOS token is refused, and the measurement that refusal replaces. |
| `tests/test_causal_lm_train_step_requires_an_optimizer.py` | `train_step` refuses to invent an optimizer, and what a per-call one would have run instead. |
| `tests/test_classification_train_step_requires_an_optimizer.py` | The same refusal on the classification task, and that both refuse the same way. |
| `tests/test_tiny_gpt2_accepts_a_null_padding_token.py` | §2.5: `tiny_gpt2` builds against a manifest that declares no padding token, and still defaults an absent key to 0. |
| `tests/test_amp_is_refused_where_it_is_ignored.py` | §3: `use_amp: true` on a task with no autocast path is refused at build and in preflight. |
| `tests/test_eval_batch_size_defaults.py` | The per-task evaluation batch-size defaults. |
| `tests/test_adamw_and_model_state_compat.py` | Model state stays loadable across optimizers, and the accumulator's non-floating-buffer check. |
| `tests/test_batchnorm_is_unsupported.py` | §4.1: a BatchNorm model raises in aggregation, and this chapter says so. |
| `tests/test_femnist_support.py` | The FEMNIST model and data path. |
| `tests/test_group_norm_request_is_recorded.py` | §4.3: the group counts the two models actually build, and that a reduced one reaches `run.json`. |

### Known failure modes

- **Adding a `.get()` without adding to `_KNOWN_KEYS`.** The builder rejects
  the key at load, and the message names the accepted set.
- **Setting `pad_token_id` for a causal-LM arm to a value the manifest
  contradicts.** Refused at build; leave it unset and take the manifest's.
- **Setting `lora_alpha` on the base builder.** It belongs to
  `hf_causal_lm_lora`; `hf_causal_lm` rejects it.
- **Choosing an odd `stage_channels` value.** Refused. Both branches are built
  at half width.
- **Bringing a BatchNorm model.** Aggregation raises on `num_batches_tracked`
  mid-round, once two clients differ in step count — §4.1. Substitute
  GroupNorm; there is no config that makes BatchNorm work.
- **Expecting `local_files_only: false` to enable a download.** The builder
  raises. Prepare the snapshot with `fedbrew prepare-llm` instead — chapter 05.
- **A typo in `asset_manifest`.** Caught by `fedbrew run --validate-only`,
  which checks the manifest and both asset directories exist before the job
  starts; the builder would otherwise catch it minutes in.
- **Assuming `num_examples` means examples trained.** For classification it
  is the train split's size, whatever the round read; for `causal_lm` it is
  active target tokens, and `federated_aggregation_weight` is what decides
  (chapter 07 §3.1).
- **Raising `client.eval_batch_size` for a causal-LM arm.** The default is the
  training batch size for a reason; the logits tensor is
  `batch × sequence_length × vocabulary`.
