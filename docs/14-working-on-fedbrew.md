# 14 — Working on fedbrew

Chapters 01 to 13 say what this codebase *is*. This one says how to change it
without undoing the discipline that got it here.

It is the only chapter that is method rather than fact, and that has a
consequence worth stating at the top: **most of it cannot be guarded.** A test
can prove that `fedbrew/data/partitioners/label_skew.py` exists and that
`fedbrew run --validate-only` is a real invocation. No test can prove you
audited before you fixed, or that you ran the thing rather than read it. §7
guards what is guardable — the paths, the symbols, the command spellings, the
cross-references, and that the fix each trap in §6 describes is still in the
code — and says plainly which claims are on trust.

Read §6 first if you read nothing else. It is the longest section because it
is the point of the chapter: five changes in this repository's history where
the obvious fix, or the one an audit report recommended, would have introduced
a real defect, one correct fix that did not travel, one cause written into the
tree before it was measured, one form of citation that read as checked, and two
correct fixes committed on a gate that could not see the tests they broke. Every
one of them was correct as a description of the symptom.

## 1. The working contract

**Audit read-only before fixing.** Read the whole affected path before
changing a line of it — the callers, the tests that pin it, the configs that
set it. Findings §6.3 and §6.5 are both cases where the fix was safe in the
function it touched and unsafe two calls away. The read is not overhead; it is
where the second defect is found.

**One commit per finding.** A commit fixes one thing and says what it fixed,
why the obvious alternative was wrong, and what evidence says it works. If a
change needs the word "also", it is two commits.

Note what this does *not* buy you here. The repository was first published as
a single commit, so none of the reasoning behind the code as published is in a
commit message — §5. The rule is for the history since, and it only helps if
the argument also lands somewhere the code carries.

**Verified means you ran it.** Reading the code and concluding it works is not
verification, and neither is a passing test you did not try to break.

| Claim | What verifying it looks like here |
| --- | --- |
| A formula is right | run it on real inputs and print both numbers — for example `_sample_weighted_avg` against `_avg` on the same three clients |
| A refactor is neutral | compare ASTs, not tokens, and check the suite counts match exactly on both sides |
| A guard works | mutate the thing it guards and watch it fail (§2) |
| A fix helps | measure the before and after and record both numbers — §6.5 measured the checkpoint on disk with the duplicate and without it |
| A command works | run it and paste what it printed, as chapter 03 does |

**A skipped test has not passed, and the environment is part of the gate.**
`python -m pytest` in an environment without an extra runs none of the tests
gated on it, and its summary line counts them beside the passes. So the gate a
change needs depends on what the change reaches. A change to the metric columns
a run writes, to data generation under `fedbrew/data/`, or to a task adapter
under `fedbrew/tasks/` reaches code the LLM tests exercise, and those tests skip
unless the `llm` extra is installed: such a change runs the suite a second time,
in an environment that has the extra, before it is pushed. The full gate does
that for every push (chapter 13 §2.2). §6.9 is why this
is a rule — two correct fixes, each committed on a green gate, each breaking
tests that gate had skipped.

**Report a deviation rather than silently choosing.** If the instruction and
the code disagree, do the work, and say in one sentence what you departed from
and why. A deviation put in writing can be reversed by whoever disagrees; a
deviation taken quietly cannot, because nobody knows it happened. The two
config `README.md` inventories survived the documentation pass this way: the
plan said shrink them to pointers, reading them showed they were file
inventories no chapter carried, and the departure was reported rather than
assumed.

## 2. Guard discipline

A guard is a test whose job is to fail when documentation and code disagree.
Chapter 13 covers how to write one. Three rules are worth repeating because
each was learned by getting it wrong.

**A guard is not finished when it passes. It is finished when a deliberate
error makes it fail.** Passing proves nothing on its own — a guard that
silently matches nothing passes forever. Break the chapter on purpose, watch
the guard reject it, put it back. Roughly sixty mutations were run across
chapters 01 to 13; two guards passed a mutation they should have caught, and
both failed for the same reason, which is the next rule.

**Locate the section, then search inside it.** A whole-file `assertIn` finds
the word somewhere else in the chapter and passes. Deleting `centralized` from
chapter 07's *server* registry line passed because the same word appears on
the client line; dropping the `attempts` row from chapter 09's key table
passed because the word recurs in a tests table two sections down. Split the
chapter on its heading first. Use `re.search(..., re.MULTILINE)` — `assertRegex`
takes no flags, so `^` and `$` anchor to the whole file and every row appears
to fail while the message dumps the entire chapter.

**Never exempt a guard to make a sentence pass. Rewrite the sentence.**
`tests/test_cli_commands_exist.py` reads any `fedbrew <word>` in the tree as an
invocation, so prose like "`fedbrew` is a benchmark framework" trips it. Adding
`is` to `NON_COMMAND_WORDS` fixes one sentence and blinds the guard across the
whole tree. That happened five times during the documentation pass and the
prose was rewritten five times. The exemption list is a review moment, not an
escape hatch: an entry in it is a claim that something is deliberately outside
the rule, and it should be small enough to read.

## 3. When an instruction and the codebase disagree

Resolve toward the codebase's stated contract, do the work, and say that you
did.

The contract is what the code and its tests already promise — the invariants
in each chapter's `## For agents`, the refusals in `validate_config`, the
column set `client_metric_names` decides. An instruction that contradicts one
of those is usually a instruction written without that context, not a decision
to abandon it.

§6.3 is the worked example. The finding said to gate the per-client CSV
rewrite on `should_save_checkpoint`. That is a correct description of how to
make the writes sparse, and it silently breaks resume, because a contract
elsewhere — `_load_existing_metric_history` rebuilding history out of those
files — depends on them being current. The fix went the other way, and the
rejected alternative is written into `flush_client_csvs`' docstring — *"Throttling
the rewrite instead is not an option"*, with the reason — so the next reader
who thinks of it finds the answer at the function rather than reaching for it
again.

What this does *not* license: overruling an explicit decision because you
prefer another one. The test is whether the instruction conflicts with
something the codebase enforces, or merely with your judgement. If it is
judgement, say so once and then do what was asked.

## 4. Never

- **Never restate what the code already enumerates.** A hand-kept list beside a
  computed answer drifts, and drifts silently. §6.4 is exactly this. If you
  need the answer, ask the function that decides it.
- **Never add a config key nothing reads.** The removed-key table in chapter 04
  §10 exists because several were added and then honoured nowhere; a config
  that sets one loads clean and does something else. Preflight refuses them by
  name now. If a key needs a second spelling, it needs neither.
- **Never fix a symptom when the class is fixable.** Renaming one wrong string
  leaves the mechanism that produced it. §6.4 again: the wrong spelling was one
  of two defects in the same list, and only one of them was visible.
- **Never trust a note over the code.** `AUDIT/` is gitignored, local-only, and
  describes the tree as it was when each report was written; several of its
  findings were resolved by deleting the option entirely. The same applies to
  any deferred-findings list, including one you wrote yourself — deferred
  finding #4 of the documentation pass claimed an LLM run without
  `HF_HUB_OFFLINE` would hang on a download, and reading
  `fedbrew/data/llm_assets/manifest.py` showed it raises `AssetManifestError`
  naming the `prepare-llm` command instead. The note was wrong, the chapter
  written from it was wrong, and both were corrected from the code.

## 5. Where the reasoning lives

Not in `git log`. This repository was first published as a single commit;
later changes are ordinary commits. The development history that produced the
published code, with its rejected alternatives and measured effects, is not
part of what shipped, so that code has no per-change history to read and no
commit hash worth citing.

That has a consequence for how you write a non-obvious change here: **the
reasoning has to go where the code is, because that is the only place the
reasoning for the code around it lives.** Three places, in order of
preference:

| Where | For |
| --- | --- |
| A comment at the call | why *this* call is the exception. `fedbrew/data/partitioners/label_skew.py`'s `strict=False` is four lines of comment on one keyword, and §6.1 is why. |
| The function's docstring | why the function is shaped this way, and what the obvious alternative would cost. `_validate_divergence_metric_is_reachable` in `fedbrew/core/config.py` says why it is not in `validation.py`. |
| A test's module docstring | the mechanism a behavioural test exists to pin. `tests/test_metric_filter_scope.py` carries the argument its assertions enforce. |

A decision recorded in none of the three is a decision the next person will
reverse, correctly, because nothing told them it was one.

## 6. Recurring traps

Ten entries. Five are changes where the mechanical or recommended fix would
have introduced a real defect; §6.6 is a correct fix that did not travel; §6.7
is a cause written into the tree before it was measured; §6.8 is a form of
citation that read as evidence and was checked by nothing; §6.9 is two correct
fixes that a green gate passed without running the tests they broke; §6.10 is a
job that reported its last line's status as its own. Each was a correct
description of the symptom. Each is here because the shape recurs.

### 6.1 A lint rule applied uniformly — `strict=True` on `_assign_allowed_labels`

`fedbrew/data/partitioners/label_skew.py` · `_assign_allowed_labels`, the
`strict=False, deliberately` comment

**What was suggested.** Ruff's B905 flags every `zip()` without an explicit
`strict=`. Six calls were flagged. The mechanical fix is `strict=True`
everywhere, and for five of the six it was right: they zip sequences that are
equal-length by construction, where a future mismatch would silently test
fewer pairs than the test claims to.

**What was wrong with it.** The sixth zips labels against client slots. The
invariant `_require_label_coverage` establishes is

```
num_clients * labels_per_client >= len(unique_labels)
```

— `>=`, not `==`. Surplus slots are the normal case and are *meant* to be
dropped, because the loop below tops each client back up to
`labels_per_client`. `strict=True` raises on every configuration that is not
exactly tight. That includes the shipped one: `data/configs/synthetic_label_skew.yaml`
is 3 labels against 5 clients × 2 slots = 10. The quickstart in chapter 03
generates that dataset, so the uniform fix would have broken the first command
a new user runs, in a commit whose message said "no behaviour changes".

**What was done instead.** `strict=False` at that one call, with the invariant
written beside it in a comment that says why. Five `True`, one `False`, each
argued.

**The shape.** A lint rule encodes a general truth. Whether it applies to a
particular call is a question about *that* invariant, and the answer is not
distributed evenly. The five-to-one ratio is precisely why "fix them all the
same way" fails: it is right often enough to feel safe.

### 6.2 A fix verified on the formula, not on the fixture — the raw teacher

`fedbrew/data/synthetic_classification.py` · `synthetic_teacher`, whose
columns are centred to sum to zero

**What was suggested.** Both synthetic generators drew targets from
`torch.randint`, never conditioned on the features stored beside them — so `y`
was independent of `X`, the Bayes-optimal classifier was the constant majority
class, and the accuracy ceiling was `1/num_classes`. A shipped config ran 20
rounds on it with `save_best: true`, selecting a checkpoint by maximising
noise. The report proposed a fixed random linear teacher, `torch.randn`, shared
by every client and the global test set.

**What was wrong with it.** `SyntheticClassificationDataset` shifts client *i*'s
features by `+i`, which is the dataset's covariate heterogeneity. A constant
shift *c* contributes `c * sum(W[:, k])` to class *k*'s logit — a per-class
constant that grows with the client index. With a raw teacher, clients 1 and 2
of the shipped three-client fixture came out **19:1 and 19:1** on a two-class
problem. A majority-class predictor scores 95% and the accuracy metric stops
meaning anything: a different route to the same failure the fix was for, where
the fixture reads as working while measuring nothing.

**What was done instead.** The teacher's columns are centred to sum to zero,
which makes it blind to a constant added to every input dimension. The shift
stays what it is documented to be — covariate heterogeneity, balanced labels.
`synthetic_teacher` and `synthetic_labels` live in one module and both
generators call them, so the two cannot drift apart. Measured on held-out rows,
a linear probe went 0.250 → 0.831 against a chance level of 0.250, and
per-client label balance was checked to confirm no client collapsed.

**The shape.** The formula was right and the fixture was wrong. Verifying a
data fix means running it on the data that ships and looking at the resulting
distribution, not confirming the algebra.

### 6.3 A throttle whose premise was false — gating on `should_save_checkpoint`

`fedbrew/core/artifacts.py` · `flush_client_csvs`, which appends;
`fedbrew/core/loop.py` · `_load_existing_metric_history`, which reads it back

**What was suggested.** The per-client CSVs were rewritten in full on every
round that wrote a checkpoint — 1.8M rows over a 500-round FEMNIST run at
`clients: all`. The gate existed to bound that cost, on the premise,
stated in the function's own docstring, that writing a checkpoint is a sparse
interval-controlled event. It is not: `save_last` writes `latest.pt` every
round regardless of `interval`, defaults to true once a `checkpointing` block
exists, and every shipped training config sets it. The gate was true every
round. The finding's fix: gate on `should_save_checkpoint` alone, so the
interval actually applies.

**What was wrong with it.** It loses data. A resume rewinds to `latest.pt` and
`_load_existing_metric_history` rebuilds the per-client history out of these
files, keeping the rows before the checkpoint's round. Any round the CSV lagged
behind would lose its rows permanently — and with FEMNIST's `interval: 100000`
the files would never be written mid-run at all.

**What was done instead.** The premise was the false thing, not the throttle.
`save_last` makes every round a rewind point, so the files have to be current
every round, and the only way to afford that is to stop rewriting them.
`flush_client_csvs` appends the round's new rows in one buffered write plus an
`fsync`, tracked by a per-run cursor, and falls back to the full atomic rewrite
whenever appending cannot be trusted: no file yet, a header that does not match,
a history that shrank because a resume replaced it, or a new metric name
widening the column set.

Two consequences the finding did not name had to be handled in the same commit.
An append is not atomic, so a killed run can leave a short final line — both
readers now drop an incomplete last row with a warning and keep everything
before it. Detecting that row uncovered a **pre-existing** fault:
`csv.DictReader` fills a short row's missing columns with `None`, and every
coercion in these readers was `or 0` or `str()`, so a torn row parsed cleanly
into a record of zeros for a client named `"None"` rather than failing.

**The shape.** When the cheap fix loses data, the premise is usually what is
wrong. Ask what the throttle assumed before deciding how tight to make it.

### 6.4 A rename that preserves the second defect — `bottom10`

`fedbrew/core/logging.py` · `_client_test_metric_names`, which asks
`client_metric_names` rather than restating it

**What was suggested.** The legend printed at run start promised
`test_accuracy_bottom10`, a spelling the loop stopped emitting when the suffix
became `worst{P}`. It named a column no run produced and omitted the one every
run did, and `tests/test_logging.py` pinned the wrong name in place. The obvious
fix is to rename the literal.

**What was wrong with it.** It fixes the symptom and leaves the mechanism. The
list had a second defect a rename does not touch: it promised `_std` and `_min`
*unconditionally*, so a config with those `client_statistics` toggles off got a
legend describing two columns its CSV would not contain, and a config with
`worst_percent: 2.5` got a legend naming `worst10`. One hand-kept list, two
independent ways to be wrong, only one of them visible.

**What was done instead.** Ask `client_metric_names` — the function that
*decides* what a run emits — instead of restating its answer. A name now
survives into the legend only if the run will actually write it, and the
worst-percent entry carries the configured percentage, its gloss built from
the name so `worst2p5` reads "the mean over the worst 2.5% of clients".
The curation stays, because a legend that lists everything explains nothing;
what changed is that the subset is now checked against the superset. Nine tests
fail against the previous `logging.py`.

**The shape.** The wrong spelling was not the defect. The defect was a hand-kept
restatement of a computed answer, and the spelling was one of its symptoms.
Renaming would have closed the visible half and left the mechanism to produce
the next one.

### 6.5 Removing redundancy that was masking a missing check — checkpoint dedup

`fedbrew/core/loop.py` · `_build_checkpoint_payload`, which pops the
duplicate; `_restore_server_state`, which refuses a checkpoint carrying
neither copy

**What was suggested.** `_build_checkpoint_payload` already held the round's
model as `server_payload["model_state"]` — one clone, made inside
`aggregate_stream`. It then called `server.save_state()` for the
optimizer-specific extras, and that method independently re-cloned the same
`self._model_state` under its own key. Both copies, numerically identical with
separate storages, were pickled into the same file, on every round of every run
for every strategy. Nothing read both. Stop writing it twice.

**What was wrong with it.** Nothing, in isolation — and that is the point.
`FedAvgServer.load_state` *skips* the model restore when `model_state` is
absent instead of raising. While the duplicate existed, a checkpoint could not
plausibly lack both copies, so the missing check had no reachable consequence.
Removing the redundancy makes it reachable: a checkpoint carrying neither copy
would resume from **freshly initialised weights and report a successful
resume** — the run silently starting over at round *R* with its metric history
intact, which is the worst class of bug this codebase has, because the artifacts
look correct.

**What was done instead.** Fixed in the writer, not in `save_state()`: the
duplicated keys are popped from the snapshot after the top level is built, so
`save_state()` stays a complete snapshot for its three other callers and no
strategy's `load_state` contract changes. `_restore_server_state` re-injects the
top-level keys when the nested state has none and prefers the nested copy when
it is there, so every checkpoint already on disk restores exactly as before —
**and refuses outright a checkpoint that carries neither.** Measured end to end,
the checkpoint had been carrying a second model-sized copy, and the file now
holds one: `tests/test_checkpoint_no_duplicate_model.py` checks that the file a
real run writes stays under twice the model's bytes.

**The shape.** Redundancy sometimes masks a missing check. Before deleting a
second copy of anything, find what would happen if *neither* copy were there,
and add the refusal in the same commit as the removal.

### 6.6 A correct local fix, with its reasoning written down — `_key_table`

`tests/test_docs_artifacts.py` checked chapter 9's `run.json` key table by
searching the whole chapter for each key name. Someone noticed that this could
not fail, fixed it, and wrote the reason beside the fix:

> Scoped, not a whole-file search: every key name also appears in the
> invariants and the tests table, so `attempts` deleted from the table would
> still be "found" somewhere in the chapter and the check would pass while the
> table was wrong.

That is a correct diagnosis, a correct fix, and the general lesson stated in
full. It did not propagate. The guard *directly above it in the same file* —
`test_every_default_artifact_file_is_documented` — had the same bug, and so did
guards in chapters 2, 10 and 11. A later sweep found eight guards across the
suite that could not fail, five of them this shape, and confirmed each by
deleting the row it claimed to protect and watching the suite stay green.

**The tempting move.** Fix the guard in front of you and write down why, which
is what good practice looks like and is what happened here.

**What it costs.** A note explains the defect to whoever reads that file next.
It does not reach the reader of a different chapter's guard, and it does not
reach the person adding a new table check next month, who will reach for the
idiom already in the tree — `assertIn(name, self.text)` — because it is
shorter and it passes. The shape kept spreading for as long as the fix was
local, and every copy read as deliberate.

**What it should have been.** `tests/docs_sections.py`: `section_of`,
`table_after` and `fenced_block_after`, which every docs guard now reaches for.
Each raises on a missing anchor rather than returning empty, because a scoping
helper that silently scopes to nothing turns a guard that could fail into one
that cannot — the same defect, relocated. Converting the eight guards found two
more instances that a reader would not have seen: a conversion scoped to
section 1 rather than to section 1's *table*, where the section's own prose
mentions `torch` twice, and a regex that matched one of four `FL_*` variables
and passed while a row was deleted. Both were caught by re-mutating after
converting, not by review.

The rule this leaves: **a shape that has recurred is a missing helper, not a
missing note.** The second time you fix the same class of defect, the fix is a
shared function and a conversion of every existing instance. Writing the reason
down is necessary and is not sufficient — it is what happened here, and the
shape spread anyway, inside the same file.

### 6.7 A cause written down before it was measured — `torch_num_threads`

Two `--all` runs of `examples/simplex-lsq` disagreed: `central_test_feasible_gap`
read `−8.3e-17` on one and `−2.8e-17` on the other, with
`runtime.deterministic: true` and the same seed. A cause was written the same
day. A threaded reduction adds its pieces in whatever order the threads finish;
`runtime.deterministic` pins the RNG and the kernel choice but not the split;
so `runtime.performance.torch_num_threads: 1` is the fix. It went into all five
`examples/*/run.py` as the comment beside the key, into a README section headed
"`deterministic: true` is not bit-reproducibility", and into a proposal to
narrow the claim chapter 10 makes for `THROUGHPUT_ONLY_PERFORMANCE_KEYS` — a
draft of that chapter change existed before the check below was run.

**What was wrong with it.** The manipulation was never performed. Re-measured
at 1, 8 and 64 threads, single-arm and `--all`, on that example and on
`examples/nonconvex-simplex`: every non-timing column of every CSV
bit-identical, and the unpinned runs land on `−2.7755575615628914e-17` every
time. Torch keeps an intra-op region on one thread below a grain size of 32768
elements (chapter 10 now says so beside the key), and these problems are 16-
and 31-dimensional. The pin could not have had an effect at any setting. The
symptom was real; the mechanism named for it could not have produced it; what
did is not recoverable.

**The tempting move.** A symptom, a mechanism that would produce it, and a fix
after which the symptom is gone — that reads as a diagnosis, and it is two of
its three parts. The step that moves the named variable and watches the effect
follow was skipped because the runs after the pin agreed, which is consistent
with the pin working and exactly as consistent with the pin doing nothing.

**What it costs.** Five copies of a false cause in the shipped tree, each
reading as measured. A README section arguing from it that a config-level
classification was wrong. A chapter edit drafted on its strength, one review
away from landing under the heading a reader opens to learn what changes the
numbers — guarded by nothing, since the classification's guard checks the key
list and not the prose. And it was the audit's own author, applying the audit's
own method: the same discipline that filed 112 findings against the framework
for publishing numbers without the check that would falsify them, turned on
itself, failed the same way at the first opportunity.

**What it should have been.** Vary the named cause and show the effect appears
and disappears — one run per thread count and a diff, minutes. Where the
mechanism has a threshold, a grain size, a rank or a dtype boundary, check
which side of it the problem sits on before invoking it. And read the column:
`feasible_gap` is a cancelling difference whose true value is `0`, so "the last
one to three significant digits" of a value near `1e-17` is *all* of its
digits, which says the column is rounding noise before it says anything about
determinism.

The retraction is in `examples/simplex-lsq/README.md` under "A withdrawn claim
about thread count", and every `run.py` comment now says what the pin does and
does not do. It is deliberately not in `FINDINGS.csv`: that file is a census
of framework defects, and an authoring error in an example is a different kind
of thing; the manifest's value is that it is a clean dataset of one kind.

The rule this leaves: **a mechanism consistent with a symptom is a hypothesis,
and it does not go into the tree until the variable it names has been moved.**
§6.2 is this trap on the input side — verified on the formula, not the fixture.
This is it on the explanation side.

### 6.8 A citation precise enough to look checked — line ranges in docs/

`docs/*.md` · `tests/test_docs_references_resolve.py` ·
`DocsSymbolCitationsResolveTest`

The chapters cited the package by file and line range. 177 of them. The form
cannot be shown here, because the guard this trap produced now refuses it
anywhere in `docs/` — which is the shortest statement of what was wrong.

**What was wrong with it.** 33 of the 177 — **19%** — already pointed somewhere
their own sentence contradicted. Thirty-one named a symbol their module defines
and cited lines that do not contain it; two cited a range that ends past the
end of the file. Six were not drift at all but citations of unrelated code: the
range offered for `shard_cache_bytes` landed in `_refuse_federated_buffers`,
which is about averaging registered buffers, and the one offered for the
server's read of `client.epsilon` landed in `AMP_AWARE_TASKS`, a list of tasks
that can honour mixed precision. Both named real files. Both read as evidence.

The rot is not per-citation, which is why reading could never have caught it.
`fedbrew/core/loop.py` grew by about 120 lines above the cited region, and
**seven symbols moved out from under their citations at once**, by 114 to 187
lines — `_stream_fit_results`, `_release_client`, `_evaluate_central_test_set`,
`_aggregate_client_split_metrics`, `_client_distribution_statistics`, and in
`fedbrew/core/runner.py` `_refuse_a_foreign_seed` and `_artifact_file_names`.
One insertion, seven false citations, no edit to any chapter.

**The tempting move.** Every chapter here has a guard, and those guards were
holding. What they hold is the chapter's *claims* — that a manifest has
nineteen keys, that a strategy list is current, that a cited path exists. A
line range is not a claim in that sense. A file existing is not the same as a
range within it meaning anything, and the pointer sat inside the guarded
sentence looking like the most checked thing on the line.

**What it costs.** Not a wrong number: a reader sent to the wrong code while
being told exactly where to look. That is worse than no pointer, for the same
reason §6.7's cause is worse than no cause — it spends the reader's trust and
returns nothing. And it compounds with what the chapters are for: a reader
follows a citation precisely when they do not already know the answer, so the
citation fails exactly at the moment its accuracy matters.

**What it should have been.** What `examples/` already did. That tree had its
two line citations stripped for this reason and named the symbol instead,
under a guard — and the argument for keeping `docs/` out of it was that each
chapter had a guard of its own, which was true and did not help. Naming a
symbol beside its file is a claim a machine can check, and 205 of them now are.
A line range cannot be checked in a way that would have caught any of this:
verifying only that the file is that long passes while the range points
somewhere unrelated, and verifying that the range contains the cited symbol
makes the symbol do the work and the range pure maintenance debt.

The rule this leaves: **a citation that looks precise is worse than one that
looks vague, because precision reads as evidence of checking.** A line range
advertises that someone looked; nothing about it survives the next insertion
above it, and nothing about it says so. §6.7 is this on the explanation side —
a mechanism that reads as measured. This is it on the reference side.

### 6.9 A gate that skipped what it passed — the `llm` extra

`tests/test_causal_lm_end_to_end.py` · the round's column set, written out;
`tests/test_hf_causal_lm_text_generator_offline.py` · `_generator_config`

**What happened.** Two findings were fixed a day apart, each with the full gate
green before its commit, and each fix was right.

- P07-F06: `_aggregate_client_split_metrics` (`fedbrew/core/loop.py`) writes
  `{split}_num_clients` beside every evaluated split's aggregates, so two runs
  averaging over different numbers of clients say so in the data rather than
  only in their configs. The commit updated every test it could see that
  pinned the column set.
- P10-F21: `_require_disjoint_client_windows` (`fedbrew/data/hf_causal_lm_text.py`)
  refuses a `causal_lm.stride` below `causal_lm.sequence_length` when a client
  eval split is cut, after measuring that at half stride a client's eval window
  can be entirely tokens it trained on. The commit checked that no shipped
  config and no chapter used a smaller stride.

**What was wrong with it.** Three tests broke, and the gate did not see them.
`tests/test_causal_lm_end_to_end.py` writes a round's exact column set out by
hand, on purpose, and did not list the two new columns. The generator fixture
in `tests/test_hf_causal_lm_text_generator_offline.py` used `sequence_length` 8,
`stride` 4 and a client eval split — exactly the configuration P10-F21 refuses —
so two tests raised before reaching anything they check. All three sit behind
`skipUnless(find_spec("transformers"))`, and the environment the gate ran in had
no `llm` extra, so the gate counted them among its skips, and a skip reads as a
pass in the summary line. The CI `llm` job is the one place they ran, and that
is where they surfaced.

Running each test just before and just after its finding settled which side
was wrong: each passed before and failed after, on exactly the two columns and
exactly the refusal. The tests were stale; the code was right.

**What was done instead.** The tests moved, not the code. The two column names
joined the written-out set. The fixture's stride went to 12: above
`sequence_length` rather than equal to it, because 8 is also the default stride
and the manifest's stride assertion could no longer have told a configured
value from a defaulted one, which the comment beside the key says. And because
each test had stopped at its first failure, nothing after that point had run
for as long as it had been failing — so with those two changes and no others,
every test in both files was run to the end and passed before the fix counted.

**The rule this leaves:** **a gate that skips a test has not passed it, and the
environment a gate runs in is part of what the gate asserts.** "Passed, N
skipped" is a claim about the tests that ran. A change that reaches what the
skipped tests exercise — metric columns, data generation, a task adapter —
needs a gate that runs them: the suite again, with the `llm` extra installed
(Commands, below). §2's first rule is this shape one level down: a guard that
matches nothing passes forever, and so does a gate run where the tests that
could fail are skipped.

### 6.10 A job whose status was its last line's — `set -e` left off

`SLURMs/example_sweep_packed.sh` · the closing `wait` loop

**What happened.** A SLURM job running a sequence of `fedbrew run` steps came
back COMPLETED, exit 0, within seconds, and none of its runs had started. The
batch shell had no `conda`, so the activation line failed and every
`fedbrew run` after it exited 127. The script had been written without
`set -e`, on purpose: some of its steps are meant to fail — a resume
interrupted by `timeout`, and a resume from `best.pt` that must be refused —
and `-e` would have ended the job at the first. Nothing was put in its place.
Each step printed its exit status and the script moved on, and its last line
was `date`. A script exits with its last command's status, so the scheduler
recorded the exit status of `date`.

The check that would have caught it existed, and failed at once when it was
run by hand against the job's output directory. But it was a separate command,
run after the job, and nothing made the job's status depend on it.

**What was done instead.** Every step states the exit status it expects — 0,
124 for the interrupt, 2 for the refusal — and any other status fails the job
on the spot. The environment is checked before the first run: `fedbrew` has to
resolve inside the environment the job names, and the device the job needs has
to be visible. And the checker is the job's last step, so COMPLETED means the
checks held. Before the job goes back to the queue it is shown failing once on
something it should catch — no GPU visible fails it in seconds — and run once
end to end on CPU.

The two shipped scripts were already right, for the same reason:
`SLURMs/example_sweep.sh` ends on the run itself, and
`SLURMs/example_sweep_packed.sh` waits on every run it started and exits with
their combined status. The comment beside that loop is this section, written
for background processes.

**The rule this leaves:** **a script's exit status is its last command's, so
leaving `-e` off is a decision to check every status by hand.** A script that
does not is reporting whatever happens to come last, and "the job completed"
is then a statement about `date`. Make the check the last command, and before
a job goes to the queue, show it failing on something it should catch.

### 6.11 The shape they share

| Trap | The tempting move | What it costs |
| --- | --- | --- |
| §6.1 | apply the lint rule uniformly | breaks the quickstart's own dataset |
| §6.2 | verify the formula | fixture reads as working while measuring nothing |
| §6.3 | make the expensive write sparse | resume loses rows permanently |
| §6.4 | rename the wrong string | second defect survives, mechanism intact |
| §6.5 | delete the redundant copy | silent fresh-weights resume becomes reachable |
| §6.6 | fix the guard in front of you, and say why | the shape spreads anyway, one file over |
| §6.7 | write the cause beside the fix that made the symptom go away | five copies of a measurement that never happened, and a chapter edit drafted on it |
| §6.8 | cite the file *and the line*, because it is more precise | 33 of 177 pointers already wrong, six of them at unrelated code, under guards that were all passing |
| §6.9 | commit on a green gate | three broken tests shipped inside its skip count |
| §6.10 | leave `set -e` off so the expected failures do not stop the job | COMPLETED, exit 0, for runs that never started |

The first five share one shape: the suggested fix correctly described the
symptom, and the defect it would have introduced lived one call, one config or
one resume away. §6.6 is the sixth's inverse and worth keeping separate — the
fix was right and the reasoning was written down; what failed was the
assumption that a correct local fix travels. §6.7 is different again: nothing
computed was wrong and the pin is harmless; what was wrong was a claim, and
the cost fell on what the tree *says* rather than on what it produces — which
in a repository whose findings are about published numbers is not the lesser
of the two. The cost of the read that finds it is minutes. The cost of not
doing it is a run that finishes and reports numbers.

§6.8 is §6.7's shape at scale, and adds the part §6.7 could not show: there,
one false claim was written once and caught by re-measuring it. Here the false
claims were written correctly and *became* false, 33 of them, with no edit to
the file they were in. Nothing a reader or a reviewer does catches that. The
only thing that does is refusing to write the form at all, which is why §6.8's
fix is a ratchet rather than a correction — and why the correction alone,
applied to all 177, would have bought roughly one more year of looking right.

§6.9 is the one entry where nothing written was wrong: both fixes were right and
both commit messages accurate. What failed was the instrument. A gate is a
statement about the environment it ran in, and that one could not run the tests
the two changes reached, so its green was true of what it ran and silent about
the rest.

§6.10 is §6.9 one level up. There the gate could not run the tests it
reported on; here the job could not run the work, and its status was that of
the one command that could still succeed.

## For agents

### Paths

| Path | What it is |
| --- | --- |
| `docs/` | chapters 00–14; this one is method, the rest are fact |
| `CONTRIBUTING.md` | a pointer to this chapter, not a second copy of it |
| `fedbrew/data/partitioners/label_skew.py` | §6.1 — the deliberate `strict=False` and the invariant beside it |
| `fedbrew/data/synthetic_classification.py` | §6.2 — `synthetic_teacher`, centred columns |
| `fedbrew/core/artifacts.py` | §6.3 — `flush_client_csvs`, the append path and its fallbacks |
| `fedbrew/core/loop.py` | §6.3 and §6.5 — `_load_existing_metric_history`, `_build_checkpoint_payload`, `_restore_server_state` |
| `fedbrew/core/logging.py` | §6.4 — the plan header's metric list, derived from `client_metric_names` |
| `tests/test_causal_lm_end_to_end.py` | §6.9 — a round's column set, written out, `{split}_num_clients` included |
| `tests/test_hf_causal_lm_text_generator_offline.py` | §6.9 — the generator fixture, its stride above `sequence_length` and why |
| `.github/workflows/tests.yml` | the `llm` job, where the LLM-gated tests run on every push |
| `fedbrew/core/config.py` | `validate_config`, on the run path via `load_config` |
| `fedbrew/core/validation.py` | `validate_full_config`, reached only by `--validate-only` |

### Commands

```bash
# The fast gate, before every commit: the tests marked fast (chapter 13 §2.2).
python -m pytest -m fast -n 16
ruff check tests tools fedbrew examples
ruff format --check tests tools fedbrew examples

# The full gate, before every push: the whole suite, then the same suite where
# the llm extra is installed, so the tests gated on it run instead of skipping.
python -m pytest -n 16
pip install -e ".[dev,llm]"
python -m pytest -n 16

# What a config actually does, without training.
fedbrew run --config configs/dev/smoke.yaml --validate-only

# The end-to-end guard, excluded by default. CI runs it on every push.
python -m pytest -m quickstart
```

### Invariants

1. **A guard is mutation-tested before it counts.** Passing proves nothing.
2. **A guard searches inside a located section**, never the whole file.
3. **An exemption list is a review moment.** Rewrite the prose rather than
   widen `NON_COMMAND_WORDS` or its equivalents.
4. **One commit per finding**, naming the rejected alternative.
5. **A refusal goes where it will be reached.** `validate_config` runs on every
   run via `load_config`; `validate_full_config` runs only under
   `--validate-only`. A check that must block a run belongs in the former.
6. **`AUDIT/` is never a source of current behaviour.** Gitignored, local-only,
   absent in a fresh clone, and written against a pre-fix tree.
7. **A deviation from an instruction is stated, not taken quietly.**
8. **A skipped test has not passed.** The full gate runs a second time with the
   `llm` extra installed, before every push, which is what a change to metric
   columns, data generation or a task adapter needs.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_working_on_fedbrew.py` | Every module path and function name cited in §6 resolves, and the code characteristic each trap describes is still present; `CONTRIBUTING.md` points here and stays thin; the chapter says which of its claims are unguardable, and cites no commit hash. |
| `tests/test_cli_commands_exist.py` | The `fedbrew ...` invocations above are real subcommands. |
| `tests/test_cli_flags_exist.py` | `--validate-only` and `--config` exist on `fedbrew run`. |
| `tests/test_docs_references_resolve.py` | The `docs/` paths named here exist. |
| `tests/test_docs_testing.py` | `OptionalExtraCoverageTest`: every extra a test is gated on is installed by some CI job — the half of §6.9 a machine can check. |

**What no test checks.** Sections 1 through 5 are method. Nothing verifies that
an audit happened before a fix, that a commit is one finding, that a guard was
mutated, that a change was gated where its tests could run, or that a deviation
was reported. Those hold because someone chose to
follow them, and the evidence is indirect: the comments beside the exceptions,
the rejected alternatives in the docstrings, and §6 here. Treat this section as
the honest limit of the guarding approach the rest of the set relies on, not as
an oversight to be closed.

### Known failure modes

- *Applying a mechanical fix across every site it matches.* §6.1. The rule is
  general; whether it applies here is a question about this invariant.
- *Verifying a data change against the formula instead of the fixture.* §6.2.
  Generate it and look at the distribution.
- *Making an expensive write sparse without asking what reads it back.* §6.3.
  Resume reads almost everything.
- *Renaming the wrong string.* §6.4. Ask the function that decides the answer.
- *Deleting a redundant copy without adding the refusal.* §6.5.
- *Writing a cause into the tree before moving the variable it names.* §6.7.
  A mechanism consistent with the symptom is a hypothesis; one run per setting
  and a diff is the test.
- *Reading "passed, N skipped" as everything passed.* §6.9. A change to metric
  columns, data generation or a task adapter runs the suite again with the
  `llm` extra installed.
- *Leaving `set -e` off a job script and checking nothing in its place.* §6.10.
  A script exits with its last command's status; make that command the check.
- *Putting a refusal in `validation.py` when it must block a run.* That module
  is reached only under `--validate-only`; `validate_config` is on the run path.
- *Citing `AUDIT/`, or a deferred-findings note, as current behaviour.* Both
  describe a tree that has since changed. Read the code.
