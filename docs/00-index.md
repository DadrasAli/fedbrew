# 00 — Index

fedbrew — a modular federated-learning benchmark framework — runs a round loop
that selects clients, applies a local update rule on each, aggregates the
results with a server strategy, evaluates, and writes a fixed set of artifacts.
Everything configurable is a YAML file; everything runnable is a subcommand of
the single `fedbrew` console script.

This directory is the primary reference. `README.md` in the repository root is
an overview and a pointer here.

## How to read this

Each chapter is self-contained: it names its own file paths, its own commands,
and its own tests, and does not assume you have read the ones before it. Read
02 → 03 once to get a run on disk, then go straight to whichever chapter owns
your question.

Every chapter ends with a `## For agents` section carrying the exact paths,
commands, invariants, guarding tests and known failure modes for that area.
If you are an automated agent, that section is the contract; the prose above it
is context.

## Chapters

| # | Chapter | Scope |
| --- | --- | --- |
| 00 | [Index](00-index.md) | This file: the map, the conventions, and how the docs are kept true. |
| 01 | [Architecture](01-architecture.md) | The round loop, the six registries, the client/server protocol, and which module owns which decision. |
| 02 | [Installation](02-installation.md) | Python and PyTorch requirements, the optional extras, offline and HPC installation, environment variables, and how to verify an install. |
| 03 | [Quickstart](03-quickstart.md) | Generate a small federated dataset, inspect it, validate a config, train, and read the result — on the synthetic problem, no downloads. |
| 04 | [Configuration](04-configuration.md) | Every config block and key: type, default, whether the default is implicit, and what rejects an unknown or contradictory one. |
| 05 | [Data and partitioning](05-data-and-partitioning.md) | The generator, the four partition strategies, the manifest format, train/val/test splits and their isolation, and offline dataset preparation. |
| 06 | [Models and tasks](06-models-and-tasks.md) | The eight model builders and the exact key set each accepts; the two tasks and the metric contract they must satisfy. |
| 07 | [Algorithms](07-algorithms.md) | The server strategies and client update rules, what each one communicates per round, and which config keys each requires. |
| 08 | [Metrics](08-metrics.md) | Every metric the codebase computes: mathematical definition, exact column or key name, the config keys that switch it on, and the aggregation rule across clients. |
| 09 | [Artifacts](09-artifacts.md) | The files a run writes, the `run.json` schema, checkpoints, the global run index, and what resume replays from. |
| 10 | [Reproducibility](10-reproducibility.md) | Seeding, determinism, the settings that change the numbers, what is guaranteed across machines, and resume equivalence. |
| 11 | [Performance and cost](11-performance-and-cost.md) | Throughput and memory behaviour, data staging, communication cost per algorithm, and the measured numbers with their dates. |
| 12 | [Extending](12-extending.md) | The two ways into the six registries — a config's `extensions` key, or the package itself — and what adding a strategy, rule, task, dataset, model, partitioner or metric takes on each. |
| 13 | [Testing](13-testing.md) | Running the suite, what each guard protects, and how to add a test that keeps a documented claim true. |
| 14 | [Working on fedbrew](14-working-on-fedbrew.md) | Method, not facts: the working contract, guard discipline, and the changes where the obvious fix would have introduced a real defect. |
| 15 | [Illustrative examples](15-illustrative-examples.md) | The five controlled problems with known optima in `examples/`: what each demonstrates, which shipped algorithms can solve it, and the discipline a new one must follow. |

## Conventions

- Paths are relative to the repository root — the directory holding
  `pyproject.toml`, `fedbrew/`, `configs/` and `tests/`.
- Commands are written as `fedbrew <subcommand>`. There are eleven
  subcommands; `fedbrew --help` lists them, and
  `fedbrew/cli/dispatch.py` is where they are declared.
- Config keys are written in dotted form (`runtime.performance.matmul_precision`)
  and correspond to nested YAML.
- A default written as **implicit** is one no shipped config states. Chapter 04
  marks every one of them, because a reader of a config cannot see it.
- Measured numbers carry the date they were measured. They are not guarded by
  tests and will drift; treat them as orders of magnitude.

## How these docs are kept true

The previous documentation set was deleted because it had drifted two CLI
generations stale while still reading as authoritative. The rule that replaces
it: **every factual claim here must be derivable from code, and the claims that
would be expensive to get wrong are guarded by a test that fails when the code
and the prose disagree.**

Each chapter has a companion guard: `tests/test_docs_<area>.py` diffs the
chapter's tables against the code that defines them, in both directions — a
thing the code has and the chapter omits is a gap, and a thing the chapter
names and the code lacks is a lie.

Three older guards set the pattern:

| Guard | What it pins |
| --- | --- |
| `tests/test_cli_commands_exist.py` | Every `fedbrew ...` command written anywhere in the shipped tree is a real subcommand. It walks the tree rather than a hand-kept list. |
| `tests/test_shipped_config_explicitness.py` | Settings that change the numbers are written into every shipped config rather than defaulted into. |
| `tests/test_matmul_precision.py` | The documentation keeps saying that `matmul_precision` changes results, under the heading a reader would look for. |

[Chapter 13](13-testing.md) lists every guard and which chapter's claims it
protects — and guards that list in both directions, so a chapter without a
guard and a guard without a chapter both fail the suite.

A chapter is not finished until its guard fails when the chapter is wrong.
That is checked by mutating the chapter and watching the guard reject it, not
by the guard merely passing.

## For agents

**Paths**

| Path | What it is |
| --- | --- |
| `docs/` | This documentation set. One file per chapter, numbered, `NN-slug.md`. |
| `README.md` | Overview only. It links here; it is not a second reference. |
| `CONTRIBUTING.md` | A pointer to chapter 14. Also not a second reference. |
| `fedbrew/` | The package. Subpackages: `core/`, `clients/`, `servers/`, `models/`, `tasks/`, `data/`, `cli/`. |
| `fedbrew/cli/dispatch.py` | The `COMMANDS` dict: the single source of truth for what subcommands exist. |
| `configs/` | 97 run configs (they carry a `runtime` block) and 4 asset-preparation configs under `configs/llm_assets/`. |
| `tests/` | 203 test modules. Several exist only to keep documentation and code in agreement. |
| `AUDIT/` | **Local-only and gitignored — absent in a fresh clone.** Audit reports describing the code as it was when each was written; they are **not** updated after a fix, and several findings they raise were resolved by deleting the option entirely. Never cite one as current behaviour, and never reference an `AUDIT/` path from a chapter: a reader cannot open it. |

**Commands**

```bash
# Verify a change to any chapter has not broken a documentation guard.
python -m pytest tests/test_docs_*.py \
                 tests/test_cli_commands_exist.py \
                 tests/test_cli_flags_exist.py

# Full suite. Excludes the quickstart end-to-end guard, which trains.
python -m pytest

# That guard, which CI runs on every push.
python -m pytest -m quickstart

# The command surface a chapter is allowed to reference.
fedbrew --help
```

**Invariants**

1. Every `fedbrew ...` command appearing in any chapter must be a key of
   `COMMANDS` in `fedbrew/cli/dispatch.py`. A stale command in documentation is
   read exactly when someone is already stuck.
2. Every chapter ends with a `## For agents` section. No chapter delegates that
   material to another chapter or to a `CLAUDE.md`; there is no `CLAUDE.md`.
3. Every chapter is readable alone. Cross-references point to another chapter
   for *more* detail, never for a fact the current chapter needs.
4. No chapter states a default, column name, key name or enum value that is not
   derivable from a named module. Where the chapter states one, it names the
   module.
5. Numbers that were measured rather than derived are labelled as measured and
   dated.

**Tests that guard this file**

| Test | Claim it protects |
| --- | --- |
| `tests/test_cli_commands_exist.py` | The eleven subcommands named above, and every command in every chapter. |

**Known failure modes**

- *Citing `AUDIT/` as current behaviour, or linking to it from a chapter.*
  That directory is gitignored and local-only, so it does not exist in a fresh
  clone — a chapter that cites an `AUDIT/` path is pointing a reader at nothing.
  The reports are also records of a pre-fix state: several findings they
  describe were resolved by removing the option entirely. Verify against code.
- *Adding a chapter without adding its row above.* The chapter table is
  hand-kept; a new file that is not listed here is invisible.
- *Writing a fact into `README.md` instead of a chapter.* The README is an
  overview. A fact that lives only there has no chapter owning it and no test
  guarding it, which is how the previous set drifted.
- *Assuming the repository root is the checkout's top directory.* The package,
  configs and tests live in the directory holding `pyproject.toml`.
