# 05 — Data and partitioning

How a federated dataset is generated, how examples are split across clients,
what the manifest records, and how the three splits are kept apart.

A federated experiment reads a **manifest**, not a raw dataset. Generation is a
separate, one-off step: `fedbrew generate` writes the manifest and the client
shards, and `fedbrew run` only ever reads them. That separation is what makes a
partition reproducible independently of the run that uses it.

## 1. The two dataset backends

`data.name` selects one (`fedbrew/core/registry.py`).

| Name | Source | Used by |
| --- | --- | --- |
| `synthetic_classification` | built in memory from a seeded linear teacher | the quickstart, `configs/dev/` |
| `manifest_dataset` | a generated manifest on disk | everything else |

`data.name` is inferred from `data.path` when unset. `manifest_dataset`
requires `data.path`; `synthetic_classification` reads the four `data.*`
dimension keys instead — chapter 04 §6.

## 2. Generating

```bash
fedbrew generate --config data/configs/<name>.yaml
fedbrew inspect-data data/generated/<name>/manifest.json
```

Eight generators, each a `GeneratorSpec` in the `generators` registry
(`fedbrew/core/registry.py`) declaring the config sections it reads. The keys
inside each section are declared too — for these eight in
`_GENERATOR_SECTION_KEYS` (`fedbrew/data/generate.py`), for a generator loaded
through `dataset.extensions` on its registration (chapter 12 §4) — and a key a
section does not read is refused at generate time like a section a generator
does not read.

| Generator | Sections it reads beyond the shared two |
| --- | --- |
| `synthetic_classification` | `synthetic`, `splits` |
| `mnist` | `mnist` |
| `cifar10` | `cifar10` |
| `femnist` | `femnist`, `client_splits` |
| `tiny_causal_lm` | `causal_lm`, `splits`, `client_splits` |
| `hf_causal_lm_text` | `hf_causal_lm_text`, `causal_lm`, `splits`, `source_splits`, `client_splits` |
| `generic_sft` | `generic_sft`, `caps`, `tree_splits` |
| `oasst1_sft` | `oasst1_sft`, `sft`, `pilot_caps`, `tree_splits`, `splits` |

Every generator also reads the two shared sections: `dataset` and
`partition`. `client_splits` is not a third: the four `tensors` generators get
it implicitly, because the shared writer in `generate.py` — not they — cuts
each client's slices, and a `shards` generator declares it only if it cuts by
it. `generic_sft` and `oasst1_sft` do not, so a config that sets the section
for them is refused rather than accepted and dropped — §2.1.

**How much of this chapter is verified, per generator.** The section tables
above come from the `generators` registry and are true for all eight. What differs
is how much *behaviour* stands behind the prose:

| Generator | Shipped config | Dedicated generator test | This chapter's basis |
| --- | --- | --- | --- |
| `synthetic_classification` | 3 | `test_synthetic_label_signal.py` | **run, output shown in chapter 03 and the README** |
| `femnist` | 1 | `test_femnist_writer_split.py` | tests and a shipped config |
| `mnist` | 4 | — | shipped configs, one per partition strategy, exercised throughout the suite |
| `oasst1_sft` | 3 | `test_oasst1_sft_generator.py` | tests and shipped configs |
| `generic_sft` | 1 | `test_generic_sft_eligibility.py` | tests and a shipped config |
| `tiny_causal_lm` | **none** | `test_tiny_causal_lm_generator.py` | tests only |
| `hf_causal_lm_text` | **none** | `test_hf_causal_lm_text_generator_offline.py` | tests only |
| `cifar10` | **none** | **none** | **its allow-list alone** |

`cifar10` is the one row to treat with suspicion. Nothing in the repository
generates a CIFAR-10 dataset: no shipped config names it, and the only test
that mentions it passes the string to a different function. Its three keys are
documented because its `GeneratorSpec` says it reads them, not because a run
was observed doing so. Everything else in this chapter is backed by at least a
test.

**OpenImage is not supported end to end.** A model (`openimage_shufflenet`,
chapter 06 §2.4) and a run config (`configs/openimage/fedavg.yaml`) exist, but
none of the eight generators reads OpenImage, so nothing here produces the
manifest that config points at. It needs FedScale's OpenImage partition —
13,771 clients, 596 classes — resized to 96x96 and written to the manifest
format §5 documents, by hand, outside this repository; until then `fedbrew run`
on that config fails at load with a missing manifest. The synthetic data
`tools/generate_openimage_shaped_synthetic.py` writes has OpenImage's *shape*,
not its content, and is for measuring round cost (chapter 11 §7), never for
training.

**A section the generator does not read is an error.** The message says why:

```
generator 'mnist' does not read these sections: femnist. It reads: client_splits,
dataset, mnist, partition. An unread section is dropped and every key in it
takes its default, so a misspelling generates a different dataset.
```

Keys within a section are checked the same way, so `sequence_lenght: 512`
fails rather than silently generating 256-token windows.

### 2.1 The shared sections

| Section | Key | Meaning |
| --- | --- | --- |
| `dataset` | `name` | which generator runs |
| | `output_dir` | where the manifest and shards are written |
| | `raw_dir` | where the source data is cached |
| | `seed` | **the partition seed** — §3.6 |
| | `extensions` | generators defined outside the package, loaded before `name` is looked up: paths ending in `.py` or module names, as for `experiment.extensions` in chapter 04. The manifest records each with its SHA-256 |
| `partition` | `strategy` | one of the five in §3; `femnist` accepts only `natural` |
| | `num_clients` | how many clients |
| | `alpha` | `dirichlet` only |
| | `labels_per_client` | `label_skew` only |
| | `min_size`, `max_size`, `sigma` | `quantity_skew` only |
| `client_splits` | `train_ratio` | fraction of each client's examples used for training |
| | `eval_ratio` | fraction held out as that client's validation split |
| | `test_ratio` | fraction held out as that client's test split |

`dataset` and `partition` are read by all eight. `client_splits` is in this
table because it is the split section for six of them, but it is not universal
and is no longer treated as though it were: `generic_sft` and `oasst1_sft` cut
their splits in `tree_splits`, at the conversation tree by SHA, so a
`client_splits` ratio has nothing to divide. Both used to accept the section,
validate it and delete it, and all three shipped OASST1 configs carried
`client_splits.eval_ratio: 0.1` under a comment saying it was there "for the
common generator schema" — a number that could be edited to anything with no
effect and no message. It is declared per generator now, so setting it where
nothing reads it is refused by name, and the message says `tree_splits` is
what is in force.

## 3. The five partition strategies

`partition.strategy`. Four of them cut a pooled corpus into clients and are
dispatched in `_partition_train_indices` (`fedbrew/data/generate.py`), each in
its own module under `fedbrew/data/partitioners/`. The fifth, `natural`, is not
a cut at all: the corpus already carries the client identity, and FEMNIST —
this repository's flagship dataset — requires it.

The four that cut partition **training indices only**. The test shard is
handled separately — §4. `natural` splits within each client instead, which is
why §4 has two answers rather than one.

### 3.1 `iid`

`partition_iid(indices, num_clients, seed)`.

Shuffle every index with the seed, then split into `num_clients` nearly equal
contiguous blocks. Sizes differ by at most one. This is the no-heterogeneity
control: any gap between an `iid` arm and a skewed one is the cost of
heterogeneity and nothing else.

### 3.2 `dirichlet`

`partition_dirichlet(labels, num_clients, alpha, seed)`. Requires
`partition.alpha`.

For each class *k*, draw proportions over clients from `Dir(alpha)` and deal
that class's examples out accordingly:

```
p_k ~ Dirichlet(alpha * 1_C)     for each class k
client c receives p_k[c] of class k's examples
```

`alpha` controls concentration: small `alpha` (0.1) gives clients that hold one
or two classes; large `alpha` (100) approaches `iid`. It is the standard
non-IID benchmark knob, and it varies both label distribution **and** client
size at once.

### 3.3 `quantity_skew`

`partition_quantity_skew(indices, num_clients, min_size, max_size, seed, sigma)`.
Requires `partition.min_size` and `partition.max_size`.

Draws client sizes from a log-normal with shape `sigma`, rescaled into
`[min_size, max_size]`, then deals shuffled indices out to fill them. Labels
stay IID; only **how much** each client holds varies.

This is the strategy that isolates the size effect: it is what makes
`{split}_{metric}_avg` and `{split}_{metric}_sample_weighted_avg` diverge
without any label heterogeneity — chapter 08 §5.

### 3.4 `label_skew`

`partition_label_skew(labels, num_clients, labels_per_client, seed)`. Requires
`partition.labels_per_client`.

Each client sees **at most** `labels_per_client` distinct labels. The bound is
a guarantee, not a preference: a configuration that cannot satisfy it raises
rather than quietly exceeding it. `tests/test_label_skew_coverage.py` guards
that every label still reaches some client.

### 3.5 `natural`

No partitioner module and no dispatch entry: the corpus supplies the clients.
FEMNIST is written by its own generator (`fedbrew/data/femnist.py`), which
groups examples by the source `writer_id` and makes one client per writer —
3,597 of them in the shipped configuration — and **refuses any other strategy**
in `generate_femnist_from_config` (`fedbrew/data/femnist.py`), because there is
nothing for a partitioner to cut.

The consequences are not the other four's:

- **Client sizes are the corpus's**, spanning an order of magnitude, and no
  `min_size`/`max_size`/`alpha` knob touches them. `femnist.min_samples_per_client`
  only decides which writers are eligible at all.
- **There is no official test set to deal out.** Every writer's train, eval and
  test slices are cut from that writer's own examples, which is the opposite of
  what §4 does for the other four — see §4 and §4.2.
- **A client is a person.** That is what makes the partition realistic, and it
  is also the limitation in §4.2: every writer in the test pool is a writer the
  model trained on.

### 3.6 Partitions are reproducible from `dataset.seed`

All five take the seed and use it for every random choice, so regenerating with
the same config gives byte-identical shards. For `natural` the seed decides
which writers are selected and where each writer's own three slices fall, keyed
on the writer rather than on its position in the selection, so requesting a
different client count does not change an unrelated writer's data. The
partition seed is **separate from `experiment.seed`**: two runs at different
training seeds share one partition, which is what makes a seed spread measure
training variance rather than partition variance. Chapter 10.

### 3.7 Every client holds at least one example

`dirichlet` and `label_skew` both deal examples out by label, so both can leave
a client with nothing. `fill_empty_clients`
(`fedbrew/data/partitioners/empty_clients.py`, one definition shared by the
two) moves one example from the largest donor into each empty client, and
**raises** when it cannot — the corpus holding fewer examples than the roster
has clients is a configuration error, not something to return quietly. The
same postcondition is asserted again where the shard is written, so a
partitioner added later cannot reintroduce a zero-row shard: `generate`
reporting success and the run failing at evaluation with "has no non-empty
train split" is the failure this pair of checks exists to prevent.

What the fill cannot promise is a validation slice. A rescued client holds
exactly one example, and one example cannot be split, so it gets a train slice
and no `eval` — legal, and handled: §4's per-split `evaluate()` reports zero
examples for that split and leaves the client out of the `val_*` aggregate
rather than failing the round. Generation therefore counts those clients into
`partition_stats.json` as `clients_without_eval_split` and warns on the
terminal when there are any. It is not a small number at the interesting end
of `alpha`: the shipped `data/configs/mnist_dirichlet.yaml` (1000 clients,
`alpha: 0.1`) leaves 29 of them, so its `val_*` curves average 971 clients, not
1000.

## 4. Splits, and what keeps them apart

Each client's examples are divided by `client_splits` into `train`, `eval` (the
validation split) and `test`. The generated manifest records the ratios it
used.

**Where a client's test examples come from depends on the generator, and the
manifest says which.** There are two answers and the difference matters, so
`client_test_source` is recorded rather than assumed:

| `client_test_source` | Written by | Where a client's test slice comes from |
| --- | --- | --- |
| `partitioned_global_test` | the shared generator (`generate.py`) and `synthetic_classification` | the corpus's **official test set**, dealt out to mirror each client's training profile — §4.1 |
| `within_client_holdout_disjoint_from_eval` | `femnist.py` | a **third slice of that client's own examples**, disjoint from its train and eval slices — §4.2 |
| `identical_to_train` | no generator in the package; the value an out-of-tree generator writes when its splits hold the same rows, named in `manifest_validation.py` | **the same rows as train.** An analytic objective is not estimated from samples — `f_i` *is* the client — so nothing is held out, and a number read off `test_*` or `central_test_*` is a training number. Preflight prints a note saying so |

Under the first two, a client's test examples were never trained on — by construction
rather than by convention — and `tests/test_partition_disjointness.py` and
`tests/test_split_reader_isolation.py` guard that the three splits never
overlap and that a reader asking for one cannot see another. What differs is
*whose* examples they are, and that decides what a test number generalises to.

Writing them apart is half of it. **A reader that substitutes one split for
another undoes the separation with the data on disk still correct**, and one
did: the split-less `evaluate()` path fell back to a client's train slice when
its val slice was missing or empty, and reported those numbers under the
caller's evaluation metric names. Nothing in the metric name, the example count
or the CSV said which slice they came from. It refuses now — a client without a
val slice is evaluated on the splits it does have, named explicitly. Every
reader resolves exactly one split or raises.

The other way to undo it is to make the *rows* overlap while the split
bookkeeping stays correct, and one configuration could:
`hf_causal_lm_text` starts a window every `causal_lm.stride` tokens, so
`stride < causal_lm.sequence_length` makes consecutive windows share tokens,
and the windows are then dealt to clients and cut into train and eval by index.
Measured on 4,000 tokens with `sequence_length` 16 and four clients, `stride: 8`
put an eval window's **every** token into that client's own train windows — the
union of the two windows either side covers it, so the bound is not the
`sequence_length - stride` a single pair shares. That combination is refused at
generation (`_require_disjoint_client_windows`); overlapping windows stay
available at `client_splits.eval_ratio: 0`, where there is no client split to
leak across, since the source test split is cut at the record level before
tokenization.

### 4.1 The test set is partitioned to match the training heterogeneity

This section is about `partitioned_global_test`. §4.2 covers the other.

Handing every client a uniform slice of the test set would measure the global
model on a distribution no client actually has, which defeats the point of the
per-client statistics in chapter 08 §5. So the official test set is dealt out
*mirroring each client's training profile*
(`fedbrew/data/official_test_partitioning.py`, `partition_test_indices_like_train`):

| Training strategy | How test examples are weighted |
| --- | --- |
| `iid` | equally across clients |
| `quantity_skew` | in proportion to each client's training size |
| `dirichlet`, `label_skew` | **per label**, in proportion to that client's training count for that label |

For the two label-heterogeneous strategies this is done one label at a time, so
a client that trained only on classes 0 and 2 is tested mostly on classes 0 and
2. A label no client trained on falls back to size-proportional weighting, and
any client left with nothing is filled rather than allowed to have an empty
test split.

That is what makes `test_accuracy_worst10` meaningful: the worst clients are
worst on data resembling their own, not on a distribution they never saw.

### 4.2 FEMNIST's test split is per-writer, not held-out writers

**A FEMNIST test number measures generalisation to new examples from writers
the model already trained on.** This is a limitation of the shipped
FEMNIST data, stated here in full because this chapter owns it.

FEMNIST has no external test set. The LEAF corpus is a pool of writers, so a
test split has to come out of those writers: a group of writers held out whole,
or a third slice of each writer's own examples. `femnist.py` cuts the second,
recording
`client_test_source: within_client_holdout_disjoint_from_eval` and refusing
`client_splits.test_ratio <= 0` outright, in `generate_femnist_from_config`
(`fedbrew/data/femnist.py`). Every writer's
train, eval and test slices are disjoint, and `global_test.pt` is the
concatenation of the **test** slices, so nothing a model was selected on
appears in a test number.

What that does *not* buy is new-writer generalisation:

| Quantity | Does FEMNIST here measure it? |
| --- | --- |
| unseen examples, seen writers (the split LEAF's reference implementations use) | **yes** — this is what `central_test_*` and `test_*` are |
| unseen writers | **no** — every writer in the test pool contributed to training, to model selection, and to whatever hyperparameter choice produced the run |

Writer style is the dominant nuisance factor in handwriting, so the two are not
close. A claim of the form "generalises to new users", or a comparison against
a published FEMNIST number measured on held-out writers, is **not supported by
a number this repository produces**. The uses that are supported: convergence
comparison at a fixed writer population, and personalization deltas, where the
same writers on both sides is the point rather than the flaw.

Achieving held-out writers would mean partitioning the writer *set* before
generation and reserving a disjoint group. No generator here does that, and no
config asks for it.

The split roles are also enforced at the config layer: `val` is what model
selection may look at, `test` is for reporting only, and
`evaluation.test.clients: participating` is refused — chapter 04 §8.

## 5. What generation writes

From the quickstart dataset:

```
data/generated/synthetic_label_skew/
  manifest.json          the entry point; data.path points here
  clients.jsonl          one JSON record per client
  client_stats.csv       per-client counts, for quick inspection
  partition_stats.json   how the partition landed
  shards/
    client_0.pt ... client_4.pt
    global_test.pt
```

The two generators with a long interior — `femnist`, which prepares one shard
per writer, and `oasst1_sft`, which tokenizes 24,239 candidate responses —
report their progress on one line,
redrawn in place, while they work. Everything else in this command is fast
enough that a result line at the end is the honest form.

`partition_stats.json`'s headline numbers — total examples, the smallest and
largest client, the mean, and the label counts — are also printed by
`fedbrew generate` as it finishes. They used to reach the file and nothing
else, which meant the one number worth checking before training on a generated
dataset (how small the smallest client is) was only visible to a reader who
knew the file existed.

### 5.1 `manifest.json`

Nineteen keys, all of them descriptive — the manifest is a description of what
was generated, never a place to change behaviour.

| Key | Example | Meaning |
| --- | --- | --- |
| `dataset_name` | `synthetic_classification` | which generator wrote it |
| `format` | `torch_shards` | shard encoding |
| `client_shard_format` | `split_v2` | per-shard split layout |
| `num_clients` | `5` | |
| `num_classes` | `3` | |
| `input_dim` | `6` | |
| `label_rule` | `linear_teacher` | synthetic only |
| `partition_strategy` | `label_skew` | which of §3 ran |
| `partition_parameters` | `{"labels_per_client": 2}` | what that strategy was given — the knobs `_PARTITION_PARAMETERS` says it reads, and only those. `{}` for `iid`, which has none |
| `seed` | `42` | `dataset.seed`, the seed every random choice in the partition came from — §3.6 |
| `input_dtype` | `float32` | the feature tensors' dtype, exactly. Checked against every shard by `validate_manifest`, so it is a description rather than a note |
| `input_range` | `[0.0, 1.0]` | a **bound** on the feature values, taken over the pooled train and test tensors. A shard holding values outside it fails validation; one holding a narrower range does not. FEMNIST's is `[0, 255]` — the difference between the two datasets is now recorded for both |
| `client_splits` | `{"train_ratio": 0.8, "eval_ratio": 0.2}` | the ratios used |
| `client_test_source` | `partitioned_global_test` | §4 |
| `clients_file` | `clients.jsonl` | |
| `shards_dir` | `shards` | |
| `global_test` | `shards/global_test.pt` | the server's pooled test shard |
| `partition_stats_file` | `partition_stats.json` | |
| `client_stats_file` | `client_stats.csv` | |

Two more are optional, and a run copies both into `run.json` when present
(chapter 09 §3.4):

| Key | Written by | Meaning |
| --- | --- | --- |
| `reference` | a generator whose data has a known answer | what a run on this data is scored against — the dials that produced it, the reference optimum, the floors — copied verbatim |
| `extensions` | `fedbrew generate`, when `dataset.extensions` was set | which extensions generated the data: entry, resolved path, SHA-256 |

### 5.2 `clients.jsonl`

One record per client, carrying the counts and the label distribution:

```json
{
  "client_id": "client_0",
  "split": "train",
  "shard": "shards/client_0.pt",
  "num_examples": 28,
  "num_train_examples": 18,
  "num_eval_examples": 5,
  "num_test_examples": 5,
  "label_counts": {"0": 16, "2": 7},
  "train_label_counts": {"0": 12, "2": 6},
  "eval_label_counts": {"0": 4, "2": 1},
  "test_label_counts": {"0": 2, "2": 3},
  "num_labels": 2,
  "dominant_label": "0",
  "dominant_label_fraction": 0.6956521739130435
}
```

`num_labels: 2` under `labels_per_client: 2` is the skew being honoured.
`dominant_label_fraction` near 0.7 is how concentrated that client is.

Two fields read less obviously than they look. `split: "train"` records which
*source* split this client was carved from — the training corpus — not which of
its own three splits the record describes. And `num_examples` is **all three**
of the client's splits summed, not its training count: the partitioner's slice
is train + eval, while the test rows come from the official test set, so the
sum is the only figure that covers both origins. `manifest_validation` checks
that it does.

The per-split label counts are what let you check a partition without loading
a shard, and are what `inspect-data` summarises.

## 6. Validating a generated dataset

```bash
fedbrew inspect-data data/generated/<name>/manifest.json
```

Reports errors and warnings, then a summary: the partition strategy, client
count, total examples, the split ratios, and the per-client train/eval spread
as `total`, `min`, `max`, `mean`. The `min`/`max` gap is the heterogeneity —
if it is zero on a skewed strategy, the partition did not do what the config
asked.

Validation lives in `fedbrew/data/manifest_validation.py` and runs here and in
the `data` check of `fedbrew run --validate-only`. An ordinary `fedbrew run`
does not call it (chapter 04 §1): loading the manifest makes narrower checks of
its own — the manifest and client roster must parse, every roster line must
carry its required keys, and no `client_id` may repeat
(`fedbrew/data/manifest_dataset.py`) — and opens no shard until a client reads
it. Run `inspect-data` or
`--validate-only` on a dataset before a long job.

What the validation establishes is structure: the files the manifest names
exist, each client's shards have the declared row counts and the expected
arrays, and the per-split and total counts add up. It does not compare
samples, so it does not detect the same sample copied into two clients' shards
by a hand edit; that splits are disjoint is a property of the shipped
generators, which their partition tests check (§4, invariant 4). The digest a
run records identifies the manifest file, not the shard bytes — chapter 09 §3.4.

## 7. Offline preparation for the LLM generators

The four causal-LM generators need a tokenizer, and sometimes a source corpus,
that a compute node cannot download. Prepare them first:

```bash
fedbrew prepare-llm     --config configs/llm_assets/<model>.yaml
fedbrew prepare-oasst1  --config configs/llm_assets/<dataset>.yaml
```

Each writes a pinned snapshot and an **asset manifest** recording the
identifier and the resolved revision. A generator config then points at that
manifest through `tokenizer_asset_manifest` or `dataset_asset_manifest`, and
the resolved revision travels into the generated dataset's metadata — so a
tokenizer change is visible rather than silent.

`configs/llm_assets/` holds these configs. They are a different schema from run
configs: no `runtime` block, and their own allow-list.

## 8. Shard caching

`manifest_dataset` reads client shards from disk on demand and keeps a bounded
cache, `runtime.performance.shard_cache_bytes`, default 4 GiB.

Under the `centralized` strategy the cache is forced to `0`: the pooled view
reads every shard once and then keeps the concatenation resident, so a cache
would only hold a second copy — `_shard_cache_bytes` (`fedbrew/core/factory.py`).

Caching cannot change results, and that rests on two different things. A cached
shard is served as a fresh structure over the cached tensors, so rebinding a key
on a served payload -- `data["x"] = data["x"].to(device)` -- edits your copy and
leaves the cache holding what was generated. The tensors themselves are the
cached ones, because serving copies of them is what the cache exists to avoid,
so an in-place edit does reach the cache and cannot be prevented: torch has no
read-only tensor. It is detected instead. The next serve of that shard compares
each tensor's version counter against the one recorded when it was cached and
refuses a shard that changed under it (`fedbrew/data/cached_payload.py`), which
is one serve late by construction -- an edit to a payload nobody reads again
changes no later round. The pooled centralized payload is served the same way.

Chapter 11 covers when this matters and what `--staging` does about slow shared
storage.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/data/generate.py` | the CLI, the section allow-lists, the partition dispatch |
| `fedbrew/data/generate.py` | `_SHARED_SECTIONS`, and the dispatch through the `generators` registry — with the registry, the authority on which sections a generator reads |
| `fedbrew/data/partitioners/iid.py` | `partition_iid` |
| `fedbrew/data/partitioners/dirichlet.py` | `partition_dirichlet` |
| `fedbrew/data/partitioners/quantity_skew.py` | `partition_quantity_skew` |
| `fedbrew/data/partitioners/label_skew.py` | `partition_label_skew` |
| `fedbrew/data/partitioners/empty_clients.py` | `fill_empty_clients` — the postcondition the two label-skewing partitioners share |
| `fedbrew/data/femnist.py` | the `natural` partition: one client per writer, the three-way per-writer split, and the refusal of any other strategy |
| `fedbrew/data/official_test_partitioning.py` | `partition_test_indices_like_train` — deals the official test set out to match each client's training profile |
| `fedbrew/data/manifest_dataset.py` | reading a manifest at run time, and the shard cache |
| `fedbrew/data/cached_payload.py` | serving a memoised shard without handing it over — what §8 rests on, shared with the pooled centralized view |
| `fedbrew/data/manifest_validation.py` | what makes a manifest valid |
| `fedbrew/data/writers/manifest.py` | writing `manifest.json` and `clients.jsonl` |
| `fedbrew/data/llm_assets/prepare.py` | `prepare-llm` |
| `fedbrew/data/oasst1.py` | `prepare-oasst1` |
| `fedbrew/cli/inspect_generated_data.py` | `inspect-data` |
| `data/configs/` | the shipped generator configs |

### Commands

```bash
# Prove this chapter's strategy list, sections and manifest keys are current.
python -m pytest tests/test_docs_data_keys.py -v

# Generate and inspect, no network.
fedbrew generate --config data/configs/synthetic_label_skew.yaml
fedbrew inspect-data data/generated/synthetic_label_skew/manifest.json

# The partition and split guarantees.
python -m pytest tests/test_partition_disjointness.py \
                 tests/test_split_reader_isolation.py \
                 tests/test_femnist_writer_split.py \
                 tests/test_label_skew_coverage.py \
                 tests/test_quantity_skew_sizes.py \
                 tests/test_shard_cache.py
```

### Invariants

1. **Generation and running are separate steps.** `fedbrew run` never
   partitions; it reads a manifest. A partition is reproducible from
   `dataset.seed` alone, and the manifest records that seed and the strategy's
   own parameters, so a dataset regenerated under the same path with a
   different knob is distinguishable from the one it replaced. §5.1.
2. **An unread config section is an error**, as is an unread key within a
   section. A dropped section takes every default silently and generates a
   different dataset.
3. **A client's test examples were never trained on, and the manifest says
   where they came from.** Two sources are legitimate and `client_test_source`
   distinguishes them: `partitioned_global_test` for the generators with an
   official test set, `within_client_holdout_disjoint_from_eval` for FEMNIST,
   which has none. Never take a client's test examples from its own *train or
   eval* slice; taking them from its own examples is what a natural partition
   with no external test set has to do. §4.
4. **The three splits never overlap**, and a reader for one cannot see
   another. Guarded in both directions. A generator config that would make the
   *rows* overlap while the bookkeeping stayed right is refused rather than
   written: `causal_lm.stride` below `causal_lm.sequence_length` alongside a
   client eval split. §4.
5. **`label_skew`'s bound is a guarantee.** A configuration that cannot honour
   `labels_per_client` raises rather than exceeding it.
6. **`dataset.seed` and `experiment.seed` are different seeds.** Sharing one
   partition across training seeds is what makes a seed spread meaningful.
7. **The manifest is descriptive.** Never add a key to it that changes run
   behaviour; that belongs in the run config.
8. **No client is written with zero examples.** A partition that cannot give
   every client one is refused — in the partitioner, and again at the shard
   writer so the next partitioner inherits the check. A client left with no
   `eval` slice is legal, and is counted in `partition_stats.json` and warned
   about rather than passed over. §3.7.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_data_keys.py` | The strategy list, generator sections and manifest keys here match `generate.py` and a real generated manifest. |
| `tests/test_generator_reads_declared_keys.py` | No generator reads a key its section's allow-list rejects, so no `.get()` here describes an option the loader refuses. |
| `tests/test_partition_disjointness.py` | No example appears in two clients or two splits. |
| `tests/test_split_reader_isolation.py` | A reader for one split cannot see another, and none substitutes a split it can reach for one it cannot. |
| `tests/test_label_skew_coverage.py` | Every label reaches some client, and the per-client bound holds. |
| `tests/test_quantity_skew_sizes.py` | Sizes land inside `[min_size, max_size]`. |
| `tests/test_empty_client_partitions.py` | A corpus too small for its roster is refused, no zero-row shard is written, and the clients left without a val slice are counted and announced. |
| `tests/test_causal_lm_window_overlap.py` | Overlapping causal-LM windows and a per-client eval split cannot both be configured, and the leak that combination produced is measured rather than asserted. |
| `tests/test_partition_provenance.py` | Two datasets differing only in a partition parameter or the seed have different manifests, and only the knobs the configured strategy reads are recorded. |
| `tests/test_manifest_input_claims.py` | `input_dtype` and `input_range` describe the shards the manifest names: the dtype exactly, the range as a bound, both checked by `validate_manifest`. |
| `tests/test_femnist_writer_split.py` | FEMNIST's natural writer partition, and that its three per-writer slices are disjoint and exhaustive. |
| `tests/test_shard_cache.py` | The cache is bounded, and serves without handing over what it keeps. |
| `tests/test_generator_atomic_write.py` | A killed generator leaves no half-written manifest. |
| `tests/test_dataset_paths_resolve.py` | Every dataset path a run config names, commented alternatives included, is one a generator config here produces — unless the config declares in full that its data comes from outside this repository. |
| `tests/test_classification_client_test_partitions.py` | Client test splits come from the global shard, for the generators that have one. |
| `tests/test_dataset_provenance.py` | `client_test_source` and `client_shard_format` reach `run.json`, so a number can be placed on the right side of §4. |
| `tests/test_mnist_source_provenance.py` | MNIST's `source` names the reader that ran, a torchvision failure that is not a download failure is not rerouted to the mirror, and every archive is checked against the digests torchvision publishes. |
| `tests/test_synthetic_label_signal.py` | The synthetic teacher produces a learnable signal. |

### Known failure modes

- **Editing a manifest by hand.** It is a description of what was generated;
  changing it makes it disagree with the shards it names. Regenerate instead.
- **Changing `dataset.seed` between arms of a comparison.** That changes the
  partition, so the arms are no longer comparable. Change `experiment.seed`.
- **Expecting `fedbrew run` to partition.** It reads a manifest.
- **A misspelled generator section, or a misspelled key inside one.** Rejected
  with the list of sections (or keys) that generator actually reads — do not
  work around it by moving the key.
- **Assuming `num_examples` in `clients.jsonl` is the training count.** It is
  every split; `num_train_examples` is the training count.
- **Running an LLM generator without preparing assets.** The tokenizer cannot
  be downloaded on a compute node. Chapter 02 §2.2.
- **Setting `shard_cache_bytes` for a centralized arm.** It is forced to `0`;
  the pooled view holds the whole concatenation already.
- **Looking for the test-set partitioner under `tests/`.** It is production
  code in `fedbrew/data/official_test_partitioning.py`, imported by
  `generate.py`. It was renamed out of pytest's `test_*.py`
  discovery pattern, which its previous name matched.
