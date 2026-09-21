# 13 — Testing

Running the suite, what each guard protects, and how to add one that keeps a
documented claim true.

`python -m pytest` is the gate for behaviour; `ruff check` and
`ruff format --check` gate style, over `tests`, `tools`, `fedbrew` and
`examples` — between them every tracked `.py` in the tree. All
three arrive with the `dev` extra, and so does mypy, which is configured in
`pyproject.toml` and **is not enforced**: it reports, and no CI job fails on
it. It was previously installed by nothing, so the one command this chapter
named that a fresh clone could not run was the one nobody could run — "not
enforced" and "not installable" are different claims and only the first was
intended.

There are two gates. `python -m pytest -m fast -n 16` runs the tests that read
only text and configs, in under a minute, and is the gate for a commit.
`python -m pytest -n 16` runs everything and is the gate for a push, once with
the `llm` extra installed and once without. §2.2 says what `fast` means, how
the mark is enforced, and what each gate costs.

Use the module form, not bare `pytest`: three modules import helpers from
`tests/` (`test_femnist_writer_split`, `test_ignored_client_options`,
`test_scaffold_fedprox_communication_cost`), and only `python -m` puts the
working directory on `sys.path`. Bare `pytest` fails collection on all three.

## 1. Running

```bash
pip install -e ".[dev]"

python -m pytest -m fast -n 16       # the fast gate, before a commit (§2.2)
python -m pytest -n 16               # the full gate, before a push
python -m pytest                     # the default suite, in one process
python -m pytest -m quickstart       # the end-to-end guard, excluded by default
python -m pytest tests/test_docs_*.py  # only the documentation guards
python -m pytest -k metric           # by name
```

`testpaths = ["tests"]` in `pyproject.toml`, so only that directory is
collected. Do not rely on it to hide a misnamed module: a production file
under `fedbrew/` whose name starts with `test_` matches pytest's discovery
pattern, and `testpaths` is the only thing between it and being imported as a
test suite. `fedbrew/data/official_test_partitioning.py` carried such a name until it was
renamed for exactly that reason.

`hypothesis` is optional. The property-based aggregation tests skip when it is
absent rather than failing, so the suite runs without it — it just checks less.

### 1.1 The complexity ratchet

`ruff check` includes `C901` at `max-complexity = 22`. That is not a target; it
is what the two worst functions in the tree measured when the rule was adopted —
`apply_cli_overrides` and `_validate_algorithm_compatibility` — so adopting it
failed nothing. Seventeen functions sit above the conventional threshold of 10
and are deliberately left alone: splitting a function because a metric says so
is refactoring to the metric rather than to a problem. `fedbrew/core/logging.py`
is the clearest case, the second-largest module in the package and whole on
purpose.

What was missing was not smaller functions. It was a number anyone could see:
nothing observed complexity at all, so the count of functions above 10 could
grow with nothing to notice it.

The ceiling only falls. Lowering it is free. Raising it is what makes a
complexity gate stop meaning anything — the cheapest response to a `C901` is one
digit in `pyproject.toml` — so `tests/test_complexity_ratchet.py` fails if the
configured value rises above the 22 it was adopted at, and fails again if the
ceiling drifts more than one step above the real worst case, which is how a
limit becomes decorative without ever being raised. It also compares the count
above 10 stated here, in `pyproject.toml` and in its own docstring against
ruff's, because that count had already gone stale once: all three said sixteen
while ruff counted 19.

The two checks that measure need ruff, which comes with the `dev` extra. Where
it is not installed — the `core-only` CI job installs no extras — they skip and
say why, and `ruff check` in the `tests` job still enforces the ceiling.

## 2. The `quickstart` marker

One test is excluded from the default suite:

```toml
addopts = "-m 'not quickstart'"
markers = [
    "quickstart: runs the docs/03 commands end to end; excluded by default",
]
```

`tests/test_docs_quickstart_e2e.py` runs the four commands in chapter 03 for
real — generate, inspect, validate, train — and checks the artifacts against
what the chapter says. It trains a model, and it is excluded because it is the only
test that does, not because it dominates the clock.

Do not read a fixed ratio between the two. Both were measured on one CPU login
node on 2026-09-02: idle, the guard cost 12s against a default suite of 35s;
under a load average of 54 on the same box, 115s against 109s. The guard is a
single training run and the suite is a thousand mostly-instant tests, so
contention moves them by different factors and which one is "the expensive
half" depends on the machine, not on the code. What is stable is that both are
in tens of seconds to a couple of minutes.

That held for a checkout with no generated data. One holding the FEMNIST
dataset took 38.6 minutes for the default suite in one process on 2026-09-18,
which is what §2.2 is about.

`-m quickstart` on the command line overrides `addopts`, because a
command-line `-m` wins. CI runs all four jobs on every push
(`.github/workflows/tests.yml`), which is the right place for the quickstart
one: a broken quickstart is caught most cheaply by a machine, and most
expensively by a new reader who cannot tell whether the fault is theirs.

## 2.1 The install-shape jobs

Three of the four CI jobs differ only in what is installed, because that is
what they are for:

| Job | Install | What only it can catch |
| --- | --- | --- |
| `tests` | `.[dev]` | the default suite, on the environment a contributor has |
| `core-only` | bare `.` | a dependency the package uses but never declares |
| `llm` | `.[dev,llm]` | the 21 tests gated behind `skipUnless(transformers)` / `skipUnless(peft)` |
| `quickstart` | `.[dev]` | the chapter 03 commands, end to end |

`tests` is also the only job that runs on more than one interpreter: 3.10 and
3.12. 3.10 is the floor `pyproject.toml` declares, and running the suite on it
rather than on the newest release is what keeps `requires-python` honest — a
tree tested only on the newest Python accepts syntax its own floor cannot parse
and nothing says so. 3.12 is the other end, which had no job at all:
`requires-python` sets no upper bound, so every version above the floor was
claimed and none was tested. The upper entry tracks the newest Python the
pinned torch publishes CPU wheels for and moves in the same commit as
`FROZEN_TORCH_RELEASE`. `tests/test_docs_testing.py` checks that the floor is in
the matrix and that something above it is, so neither end can be dropped
quietly.

A test that is skipped everywhere is a test that nothing runs, and the suite
summary reports it beside the passes. The `llm` job exists because that was
the state of the LLM tests: no job and no default environment installed the
extra, so the 21 tests behind it had never executed, and eight of them had
gone stale against config keys and methods the code retired — the same defect
class as a documentation table naming a flag that no longer exists, in the one
place the guard sweep could not see. Every test under the extra builds its
model and tokenizer fixtures in a temporary directory, so the job needs no
network beyond pip.

## 2.2 The `fast` mark and the two gates

| Gate | Command | Run it | Tests | Wall time, 2026-09-18 |
| --- | --- | --- | ---: | ---: |
| fast | `python -m pytest -m fast -n 16` | before a commit | 1,714 | 43 s |
| full | `python -m pytest -n 16` | before a push, with and without the `llm` extra | 2,104 | 79 s without the extra, 104 s with |

Measured on a CPU login node whose per-user quota is 32 cores, in a checkout
holding generated FEMNIST and MNIST data, at a load average between 7 and 23
from other users. On the same node the default suite took 38.6 minutes in one
process (2,319.7 s), and a clean export with no generated data, which is what
CI runs, took 57 s without the `llm` extra and 96 s with it, at 16 workers.

**Where the 38 minutes went.** Not into training. The 20 slowest tests took 33
of them, and each one preflights a shipped FEMNIST or MNIST config, usually
beside the run path to check that the two agree. Preflight validates the
dataset the config names: for FEMNIST, 3,597 shards and 702 MB, each one
opened, loaded and range-checked, about 13 s a call at one thread, and the
slowest test made 13 calls. The 1,776 tests that did nothing heavier than read files took
191 s between them. CI has no generated data, so none of this showed there.

**What changed.** All of it is in `tests/conftest.py` and the marks; nothing
under `fedbrew/` changed.

| Change | Why | Measured |
| --- | --- | --- |
| Every test process runs one thread: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to 1 before torch loads | torch otherwise starts 191 threads a process, so 16 workers would pass the node's limit of 2,000 tasks, and the default oversubscribed the 32-core quota | one FEMNIST preflight: 24–28 s at the default, 13.4 s at one thread |
| `pytest-xdist`, in the `dev` extra | 2,102 tests that are independent of each other, run one at a time | with the next row, the full suite went from 38.6 min to 79 s |
| A dataset under `data/generated/` is validated once per process, keyed on the arguments and on the size and modification time of every file in it | the same dataset was validated up to 13 times a test | the slowest test: 367 s to 15 s |

**Why 16 workers.** On that node the fast half took 51 s at 4 workers, 41 s at
8, 33–41 s at 16 and 44 s at 32. The whole suite, before the dataset change,
took 373 s at 16 and 310 s at 64, which is what `-n auto` chose there because it
counts physical cores; both runs were bounded by their slowest test. Past 16,
the 32-core quota is the limit. On a machine with fewer cores, use `-n auto`.

**What `fast` means.** A test marked `fast` reads text, configs and small
in-memory objects, and does none of the following:

| What a fast test must not do | Counted at |
| --- | --- |
| started a subprocess | `subprocess.Popen` |
| ran autograd backward | `torch.autograd.backward` |
| stepped an optimizer | a global `torch.optim` step hook |
| iterated a DataLoader | `DataLoader.__iter__` |
| called torch.load | `torch.load` |
| called torch.save | `torch.save` |

`tests/conftest.py` counts each of these per test and fails a `fast` test that
did any of them, at teardown, naming what it did. It checks in every run,
including CI's, so a wrong mark cannot get past a full gate.

**How the marks were set.** From a per-test record of that work across eight
runs: in one process, at 16 workers, in shuffled orders, with and without the
`llm` extra, and in a clean export of the tree with no generated data, which is
what CI sees. A test is marked only if it did none of that in any run. Two
kinds of test did none of it in every run and are still unmarked, because what
they do depends on what ran before them:

- A test in a class whose `setUpClass` trains or generates data. That work is
  charged to whichever test of the class a worker runs first, so a mark on one
  of its tests would pass or fail with the schedule.
- A test that loads a shipped example's extension (`examples/*/problem.py`).
  Each example checks its gradient with a backward pass when it is imported,
  and that is charged to whichever test imports it first in a process. The
  guard found three such tests on its first run.

The mark sits at the narrowest level that is true: `pytestmark` for a module
whose every test qualifies, a class decorator for a class, and a method
decorator otherwise.

**Adding a test.** Mark it `fast` if it reads only text and configs, and the
guard will say if it does not qualify. An unmarked test runs only in the full
gate, which costs speed, not coverage.

**Order.** A test that passes only after another test has run will fail under
`-n`, where the order changes from run to run. Measuring this found two. Both
checked the shipped generator configs without loading the `dataset.extensions`
each config names, and one of them skipped any config it could not resolve, so
run alone it checked 14 of 24 and passed (FINDINGS `POST-F17`). Each now loads
what it needs.

## 3. The three kinds of test

**Behavioural.** Most of the suite. They run code and assert on what it
produces: `test_aggregation_correctness.py`, `test_fedopt_server.py`,
`test_partition_disjointness.py`.

**Invariant.** They assert a property rather than a value:
`test_aggregation_properties.py` checks permutation invariance and
single-client identity; `test_non_finite_aggregation.py` checks that a NaN
changes values without changing the column set.

**Documentation guards.** They diff a chapter against the code that defines
what it describes. This is the group the previous documentation set did not
have, and the reason it went stale.

## 4. The documentation guards

Each chapter has one. They are bidirectional by design: a thing the code has
and the chapter omits is a gap; a thing the chapter names and the code lacks is
a lie.

| Guard | Chapter | Diffs against |
| --- | --- | --- |
| `tests/test_docs_architecture.py` | 01 | the six registries, the abstract base methods, every cited path |
| `tests/test_docs_installation.py` | 02 | `pyproject.toml`, `COMMANDS`, the environment variables the tree reads |
| `tests/test_docs_quickstart.py` | 03 | the commands, configs and artifact list |
| `tests/test_docs_quickstart_e2e.py` | 03 | the four commands, actually run — marked `quickstart` |
| `tests/test_docs_config_keys.py` | 04 | `_KNOWN_EXTRA_KEYS`, dataclass defaults, `_REMOVED_KEYS`, the enums |
| `tests/test_docs_data_keys.py` | 05 | the generator registry's sections, a generated manifest, the provenance table |
| `tests/test_docs_model_keys.py` | 06 | each builder's `_KNOWN_KEYS`, per builder |
| `tests/test_docs_algorithms.py` | 07 | the registries per registry, the pairings, the cost notices |
| `tests/test_docs_metric_names.py` | 08 | `client_metric_names`, `CLIENT_METRIC_BASES`, the fixed CSV schemas |
| `tests/test_docs_artifacts.py` | 09 | a real `run.json`, `DEFAULT_ARTIFACT_FILES`, the write strategies |
| `tests/test_docs_reproducibility.py` | 10 | the seed derivation, executed |
| `tests/test_docs_performance.py` | 11 | the dataloader gates, the staging keys, the tool list |
| `tests/test_docs_extending.py` | 12 | every extension point named exists; every labelled excerpt is a verbatim quote of the file it names; §2's two commands, actually run |
| `tests/test_docs_testing.py` | 13 | this chapter, against the suite |
| `tests/test_docs_working_on_fedbrew.py` | 14 | every cited module, symbol and commit; `CONTRIBUTING.md`; the unguardable-claims note |
| `tests/test_docs_illustrative_examples.py` | 15 | the five-example table against `examples/README.md`, the ground-truth rule against every `problem.py`, and the two unbuilt items against the tree |

Seventeen more guard documentation without belonging to one chapter:

| Guard | Protects |
| --- | --- |
| `tests/test_cli_commands_exist.py` | every `fedbrew <subcommand>` written anywhere in the tree |
| `tests/test_cli_flags_exist.py` | every `fedbrew <subcommand> --flag`, against that subcommand's argparse parser |
| `tests/test_docs_references_resolve.py` | every `docs/*.md` path written anywhere resolves, and neither `docs/` nor `examples/` cites the package by line number: a docs citation names a symbol beside the file it lives in, and that symbol is checked against that file |
| `tests/test_docs_inventory_counts.py` | the counts stated in prose -- chapters, run configs, test modules -- are the directory listings |
| `tests/test_readme_is_an_overview.py` | `README.md` stays a pointer, does not become a second reference, names only extras that exist, and lists no limitations: each is stated by the chapter that owns it |
| `tests/test_optional_dependency_imports.py` | nothing in `fedbrew/` imports an extra's package at module level, which is what lets `torchvision` live in an extra |
| `tests/test_console_is_the_only_renderer.py` | nothing in `fedbrew/` but `fedbrew/core/console.py` imports rich, writes an escape sequence, or embeds console markup -- one palette, one marker set, one TTY gate |
| `tests/test_console_surface.py` | the renderer that monopoly protects: a redirected stream gets no escape sequence and no redraw, a block's label column is measured, and a value is never read as markup |
| `tests/test_metric_glosses.py` | every column a run writes has a composed gloss, `METRIC_BASE_GLOSSES` tracks `CLIENT_METRIC_BASES`, and `_avg` and `_sample_weighted_avg` differ in words |
| `tests/test_plan_header.py` | the plan header's four blocks, its resolved-not-configured values, and the five -- and only five -- places amber is allowed |
| `tests/test_validation_rail.py` | `--validate-only` streams every check (including after one fails), withholds the plan when one does, and repeats every error in the verdict |
| `tests/test_run_reporting.py` | which rounds the terminal reports (evaluation rounds, decided by the schedules), the live footer staying off a redirected stream, the selection it names coming from the fit phase, and `--quiet`'s single outcome line |
| `tests/test_round_block.py` | every column a run can produce classifies into a group, a split and a qualifier, and a name neither authority knows lands in a loud `unclassified` group rather than a plausible one |
| `tests/test_data_command_reporting.py` | `generate`/`prepare-llm`/`prepare-oasst1` report the stages that earn a line, the partition statistics reach the terminal, and all four printing commands accept the same three flags |
| `tests/test_no_local_only_references.py` | nothing in the shipped tree cites a path inside the gitignored `AUDIT/`, in code and configs as well as chapters |
| `tests/test_metric_filter_scope.py` | what `client.metrics` and `server.metrics` reach, in the code and in chapter 08 §4.3 |
| `tests/test_divergence_metric_reachable.py` | `load_config` refuses a `divergence.metric` that `client.metrics` or `server.metrics` would filter out, and the check stays on the run path |

And two predate the set and set its pattern:

| Guard | Protects |
| --- | --- |
| `tests/test_shipped_config_explicitness.py` | every shipped config states the settings that change the numbers |
| `tests/test_matmul_precision.py` | the value check, and that chapter 10 keeps saying the setting changes results |

## 5. Writing a documentation guard

The pattern, in the order it matters.

**1. Find the authority.** The function or constant that *decides* the thing —
`client_metric_names`, `_KNOWN_EXTRA_KEYS`, a builder's `_KNOWN_KEYS`. If the
chapter's table is a restatement of that value, diff against it. If there is no
single authority, that is worth knowing before writing the chapter.

**2. Diff both ways.** `assertEqual` on two sets, not `assertIn` in a loop. One
direction catches an omission; the other catches an invention.

**3. Scope the search to the table, not the file.** This has caught me out
twice. Dropping `centralized` from chapter 07's *server* registry line passed a
whole-file search because it also appears on the client line; dropping the
`attempts` row from chapter 09's key table passed because the word appears
again in the tests table. Locate the section, then search inside it.

**4. Use `re.search(..., re.MULTILINE)`, not `assertRegex`.** `assertRegex`
takes no flags, so `^` and `$` anchor to the whole file — every row "fails"
while dumping the entire chapter into the failure message.

**5. Match whitespace-insensitively for prose.** A sentence that wraps across
two lines will not match as written. Collapse whitespace first.

**6. Assert the scan found something.** A guard that walks a tree and silently
matches nothing passes forever. Every scan-based test here asserts a floor
first.

**7. Then mutate the chapter and watch it fail.** A guard is not finished when
it passes. It is finished when a deliberate error in the chapter makes it fail,
with a message that says which claim broke.

## 6. What is not covered

Stated plainly, because a test suite's gaps are as useful as its contents.

- **Measured performance numbers.** Chapter 11's figures are dated
  measurements, not assertions. `tools/` reproduces them.
- **Prose accuracy.** A guard checks that a chapter names the right keys, not
  that its explanation of them is correct.
- **Multi-GPU, real clusters, real network.** There is no transport layer to
  test.
- **The generators without a shipped config.** Chapter 05 §2 says which, and
  `cifar10` has no generator test at all.
- **Prose style.** `ruff format` normalises the code and is enforced in CI;
  nothing checks how a chapter reads.

## For agents

### Paths

| Path | What it is |
| --- | --- |
| `tests/` | the suite: behavioural tests, invariant tests, and the documentation guards in §4 |
| `pyproject.toml` | `testpaths`, `addopts`, the `quickstart` and `fast` markers, ruff config |
| `tests/conftest.py` | one thread per test process, the `fast` mark's guard, and each shipped dataset validated once per process |
| `.github/workflows/tests.yml` | four jobs: the default suite, the `llm` extra, a bare core install, and `-m quickstart` |
| `tests/test_docs_metric_names.py` | the richest documentation guard, and the template for a new one |
| `tests/test_cli_commands_exist.py` | the tree-walking pattern every scan-based guard follows |
| `fedbrew/data/official_test_partitioning.py` | production code, renamed out of pytest's `test_*.py` pattern |

### Commands

```bash
# Prove this chapter names the guards that exist, and no others.
python -m pytest tests/test_docs_testing.py -v

# The fast gate, before a commit: the tests marked fast, across 16 processes.
python -m pytest -m fast -n 16

# The full gate, before a push: everything, with and without the llm extra.
python -m pytest -n 16

# The excluded end-to-end guard. CI runs this too.
python -m pytest -m quickstart

# Only the documentation guards, which is the fast check after editing a chapter.
python -m pytest tests/test_docs_*.py tests/test_readme_is_an_overview.py

# Lint and format, as CI runs them.
ruff check tests tools fedbrew examples
ruff format --check tests tools fedbrew examples
```

### Invariants

1. **No production module under `fedbrew/` is named `test_*.py`.** Such a
   name matches pytest's discovery pattern, and only `testpaths` stands
   between it and being imported as a test suite.
2. **The `quickstart` marker stays registered and excluded by default**, and CI
   keeps running it.
3. **Some CI job installs every optional extra whose tests exist.** A
   `skipUnless` with no job behind it is a test that has never run.
4. **A scan-based guard asserts its scan found something** before asserting
   what it found.
5. **A documentation guard diffs both directions** against the authority, not
   a list kept in the test.
6. **A guard is mutation-tested before it counts.** Passing proves nothing on
   its own.
7. **Every chapter has a guard**, and this chapter's table names it. Adding a
   chapter without one is caught here.
8. **`ruff check` and `ruff format --check` both gate**, over `tests`,
   `tools`, `fedbrew` and `examples`. Run `ruff format` before committing; CI
   fails on an unformatted file.
9. **Every tracked `.py` file lives under one of those four paths**, and the
   command is written identically everywhere it appears. Both directions are
   guarded: a new directory of Python that no path covers fails, and so does a
   copy of the command that drifts from the one CI runs.
10. **A test marked `fast` does none of the work in §2.2's table.**
    `tests/conftest.py` fails it otherwise, in every run.
11. **No test depends on another test having run first.** Under `-n` the order
    changes every run, so such a test fails some of the time.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_testing.py` | Every guard this chapter names exists, every `test_docs_*.py` file is named here, the quickstart marker is registered and excluded, CI runs all four jobs including the one that installs the `llm` extra, and the `fast` marker is registered and §2.2's table names exactly the work `tests/conftest.py` refuses. |
| `tests/test_cli_commands_exist.py` | The commands in this chapter's blocks are real. |
| `tests/test_docs_references_resolve.py` | The chapter paths named here resolve. |
| `tests/test_lint_covers_the_tree.py` | The lint paths above cover every tracked `.py`, and every copy of the command — here, in `README.md`, in `CONTRIBUTING.md`, in chapter 14 and in CI — names the same ones. |

### Known failure modes

- **Adding a chapter without a guard.** `test_docs_testing.py` fails: it walks
  `docs/` and requires a companion.
- **Adding a `test_docs_*.py` without listing it here.** Same test, other
  direction.
- **Adding a directory of Python and not the lint path for it.** Nothing about
  it fails: the files are simply never checked, which is how `examples/` went
  unlinted from the commit that created it. Guarded now, in both directions.
- **Writing a scan-based guard with no floor assertion.** It will pass forever
  and check nothing.
- **Using `assertRegex` with `^`.** No flags; the anchor applies to the whole
  file.
- **Searching the whole chapter for a claim that lives in one table.** Passes
  when the claim is duplicated elsewhere in the chapter, which it usually is.
- **Running `pytest` from outside the repository root.** `testpaths` is
  relative.
- **Marking a test `fast` because it ran quickly.** Speed is not the criterion.
  The guard counts work, and a test whose class trains in `setUpClass`, or that
  loads a shipped example, does that work in whichever test comes first.
- **A test that relies on an earlier one.** It passes in one process and fails
  under `-n`. `POST-F17` was two of these.
- **Assuming `pytest` runs the quickstart guard.** It does not; `-m quickstart`
  does, and so does CI.
