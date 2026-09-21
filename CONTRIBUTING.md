# Contributing

Everything about how to work on this codebase lives in
[docs/14-working-on-fedbrew.md](docs/14-working-on-fedbrew.md): the working
contract, guard discipline, how to resolve an instruction that conflicts with
the code's own contract, and — the longest part — five changes in this
repository's history where the obvious fix would have introduced a real defect.

The rest of the documentation set is indexed at
[docs/00-index.md](docs/00-index.md). Chapter
[13](docs/13-testing.md) covers the test suite and how to add a guard.

This file is a pointer. It is deliberately not a second copy of any of that,
because two locations for the same rule is how the previous documentation set
drifted.

## The short version

```bash
python -m pytest -m fast -n 16                  # the gate for a commit
python -m pytest -n 16                          # the full gate, before a push
ruff check tests tools fedbrew examples
ruff format --check tests tools fedbrew examples
python -m pytest -m quickstart                  # excluded by default; CI runs it
```

- One commit per finding, naming the alternative you rejected.
- A guard is finished when a deliberate error makes it fail, not when it passes.
- Verified means you ran it.
- If an instruction conflicts with something the codebase enforces, resolve
  toward the codebase and say that you did.
- The audit's findings are a dataset, not prose:
  [FINDINGS.csv](FINDINGS.csv), read with [FINDINGS.md](FINDINGS.md).
