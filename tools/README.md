# tools/

Scripts you run by path, deliberately **not** part of the installed package —
there is no `__init__.py` here and `pyproject.toml` does not list `tools` under
`packages.find`.

The dividing line: anything a user runs as part of a normal workflow gets a
module in `fedbrew/cli/` and a subcommand on the single `fedbrew` console
entry point, dispatched from `COMMANDS` in `fedbrew/cli/dispatch.py`. What is
left here is operational or one-off — it would be noise in the shipped wheel.

Run each with `python tools/<script>.py`; every one takes `--help`.

**What each script measures, and when to reach for it, is
[chapter 11 §7](../docs/11-performance-and-cost.md).** It is not repeated here:
that chapter states which of its measured numbers came from which script, so
the two would have to stay in step, and keeping one list is cheaper than
checking two.
