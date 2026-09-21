"""What `generate`, `prepare-llm` and `prepare-oasst1` say while they work.

All three were silent. `generate` printed a five-line summary at the end and
nothing before it; both `prepare` commands printed one line, identical whether
they had just re-downloaded a model or found a verified cache and done nothing
at all. The worst cases were a real `generate` of the OASST1 SFT dataset and a
`prepare-llm` that failed, each silent until it ended.

What each one now reports follows one rule: a line only if it carries
information the reader does not already have from typing the command. That is
why the config load, the imports and the mkdirs have no line anywhere here,
and why the cache check has one in both prepare commands -- it is the single
most consequential thing either of them decides.

The three flags are defined once, in fedbrew.core.console, and added to every
command that prints. A flag that works on `fedbrew run` and not on
`fedbrew generate` is worse than no flag.
"""

from __future__ import annotations

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from fedbrew.core import console
from fedbrew.core.console import DONE, RAIL, build_surface, silent_rail
from fedbrew.data import generate as generate_module
from fedbrew.data import oasst1 as oasst1_module
from fedbrew.data.llm_assets import prepare as prepare_module

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_CONFIG = REPO_ROOT / "data" / "configs" / "synthetic_label_skew.yaml"


@pytest.mark.fast
class TheFlagsAreTheSameEverywhereTest(unittest.TestCase):
    """One definition, four commands."""

    COMMANDS = (
        ("run", "fedbrew.core.runner"),
        ("generate", "fedbrew.data.generate"),
        ("prepare-llm", "fedbrew.data.llm_assets.prepare"),
        ("prepare-oasst1", "fedbrew.data.oasst1"),
    )

    def test_every_printing_command_accepts_all_three(self) -> None:
        from importlib import import_module

        for name, module_path in self.COMMANDS:
            with self.subTest(command=name):
                module = import_module(module_path)
                for flag in ("--quiet", "--verbose", "--no-rich"):
                    args = module.parse_args(_minimal_argv(name) + [flag])
                    attribute = flag.lstrip("-").replace("-", "_")
                    self.assertTrue(
                        getattr(args, attribute),
                        f"{name} parsed {flag} into something falsy",
                    )

    def test_quiet_and_verbose_are_mutually_exclusive_everywhere(self) -> None:
        from importlib import import_module

        for name, module_path in self.COMMANDS:
            with self.subTest(command=name):
                module = import_module(module_path)
                with self.assertRaises(SystemExit):
                    with _silenced_stderr():
                        module.parse_args(_minimal_argv(name) + ["--quiet", "--verbose"])


@pytest.mark.fast
class TheSilentRailTest(unittest.TestCase):
    """The null object that keeps every library caller's signature and silence."""

    def test_it_renders_nothing_but_still_counts(self) -> None:
        buffer = io.StringIO()
        rail = build_surface(file=buffer, quiet=True).rail(["a"])
        with rail.stage("a") as stage:
            stage.warn("something")
        rail.result("b", "value")

        self.assertEqual(buffer.getvalue(), "")
        self.assertEqual(rail.warnings, 1)

    def test_the_default_for_every_entry_point_is_silence(self) -> None:
        """A caller that wants the files and not the commentary must not have
        to construct a renderer to say so."""

        self.assertIsInstance(silent_rail(), console.Rail)


class GenerateTest(unittest.TestCase):
    def _generate(self, **surface_options) -> str:
        buffer = io.StringIO()
        document = yaml.safe_load(GENERATOR_CONFIG.read_text(encoding="utf-8"))
        with TemporaryDirectory() as scratch:
            document["dataset"]["output_dir"] = f"{scratch}/generated"
            config_path = Path(scratch) / "generator.yaml"
            config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
            surface = build_surface(file=buffer, **surface_options)
            generate_module.generate_from_config(
                config_path, surface.rail(["partition", "clients", "labels"])
            )
        return buffer.getvalue()

    def test_the_partition_statistics_reach_the_terminal(self) -> None:
        """They were computed on every run and written only to
        partition_stats.json. "The smallest client has 9 examples" is the
        sentence a reader needs before training on the result, and finding it
        meant opening a file they did not know existed."""

        rendered = self._generate()

        self.assertIn(f"{RAIL} {DONE} clients", rendered)
        self.assertIn("examples", rendered)
        self.assertIn("mean", rendered)
        self.assertIn(f"{RAIL} {DONE} labels", rendered)
        self.assertIn("classes", rendered)

    def test_the_partition_line_names_the_parameter_actually_in_force(self) -> None:
        """A config may carry alpha, labels_per_client and sigma together;
        only the one belonging to the configured strategy does anything."""

        rendered = self._generate()

        self.assertIn("label_skew", rendered)
        self.assertIn("labels_per_client=", rendered)
        self.assertNotIn("alpha=", rendered)

    def test_path_a_gets_result_lines_and_no_pending_state(self) -> None:
        """Every stage of this path is short. An animated state would be a
        claim about where the time goes that is not true."""

        rendered = self._generate()

        self.assertNotIn("\r", rendered)
        self.assertNotIn(console.PENDING, rendered)

    def test_verbose_adds_the_lines_that_do_not_earn_one_by_default(self) -> None:
        """--verbose has to mean something on every command that accepts it. A
        flag that parses and does nothing is worse than one that is absent:
        it teaches the reader that the flag is available and then ignores it."""

        normal = self._generate()
        verbose = self._generate(verbose=True)

        self.assertNotIn("seed", normal)
        self.assertIn("seed", verbose)
        self.assertIn("config", verbose)
        # Everything the default shows is still shown.
        self.assertIn("partition", verbose)
        self.assertGreater(len(verbose.splitlines()), len(normal.splitlines()))

    def test_quiet_leaves_the_output_path_and_nothing_else(self) -> None:
        rendered = self._generate(quiet=True)

        self.assertEqual(rendered.count("\n"), 1, f"not one line: {rendered!r}")
        self.assertIn("Output:", rendered)

    @pytest.mark.fast
    def test_the_shards_generators_are_dispatched_from_the_registry(self) -> None:
        """Five branches differing only in the module they imported. Collapsed
        into one registry so the stage around them is wired once rather than
        five times, and so an out-of-tree generator is dispatched the same way."""

        from fedbrew.core.registry import generators, register_builtin_components

        register_builtin_components()
        shards = [name for name in generators.builtin() if generators.get(name).kind == "shards"]
        self.assertEqual(len(shards), 5)
        for dataset_name in shards:
            with self.subTest(dataset=dataset_name):
                generate = generators.get(dataset_name).resolve()
                self.assertEqual(
                    generate.__name__,
                    f"generate_{dataset_name}_from_config",
                    f"{dataset_name} resolves to a function not named for it",
                )


@pytest.mark.fast
class TheVerboseDetailTest(unittest.TestCase):
    def test_a_detail_is_silent_at_the_default_verbosity(self) -> None:
        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail(["cache root"])
        rail.detail("cache root", "/data/raw/models/qwen")
        self.assertEqual(buffer.getvalue(), "")

    def test_a_detail_is_a_result_line_under_verbose(self) -> None:
        buffer = io.StringIO()
        rail = build_surface(file=buffer, verbose=True).rail(["cache root"])
        rail.detail("cache root", "/data/raw/models/qwen")
        self.assertIn(f"{RAIL} {DONE} cache root", buffer.getvalue())
        self.assertIn("/data/raw/models/qwen", buffer.getvalue())


@pytest.mark.fast
class TheProgressReportingGeneratorsTest(unittest.TestCase):
    """The longest silence in the tool was inside one of these.

    A real `generate` of the OASST1 SFT dataset spends most of its time
    tokenizing 24,239 candidate responses one at a time, and none of it said
    anything. The generator now reports each phase and counts both of its
    bounded loops through `on_progress`, and `generate` routes that to
    `Stage.tick`, which throttles and pulses -- the same line the round loop
    draws while it visits clients.
    """

    def test_the_wiring_is_read_off_the_signature(self) -> None:
        """A generator is handed `on_progress` exactly when it accepts it.

        This used to be a declared set beside the dispatch table, checked here
        against the signatures. A generator in the set without the parameter
        was a TypeError at the moment someone generated; one with the
        parameter and not in the set was instrumentation nobody saw. The
        signature is now the only source, so the two cannot disagree.
        """

        import inspect

        from fedbrew.core.registry import generators, register_builtin_components

        register_builtin_components()
        accepts = set()
        for dataset_name in generators.builtin():
            spec = generators.get(dataset_name)
            if spec.kind != "shards":
                continue
            generate = spec.resolve()
            wired = generate_module._accepts_progress(generate)
            self.assertEqual(
                wired,
                "on_progress" in inspect.signature(generate).parameters,
                f"{dataset_name}: the wiring and the signature disagree",
            )
            if wired:
                accepts.add(dataset_name)
        self.assertEqual(accepts, {"femnist", "oasst1_sft"})

    def test_a_generator_called_as_a_library_still_reports_to_nobody(self) -> None:
        """The parameter defaults to None everywhere, so importing and calling
        one of these directly behaves exactly as it did."""

        import inspect

        from fedbrew.core.registry import generators, register_builtin_components

        register_builtin_components()
        for dataset_name in ("femnist", "oasst1_sft"):
            with self.subTest(dataset=dataset_name):
                generate = generators.get(dataset_name).resolve()
                parameter = inspect.signature(generate).parameters["on_progress"]
                self.assertIsNone(parameter.default)

    def test_the_femnist_writer_counter_no_longer_prints_for_itself(self) -> None:
        """It wrote straight to stdout from inside a generator -- the second
        console layer the package forbids -- and once `generate` wrapped the
        call in a rail stage it interleaved with the stage's redrawn line. The
        counter was right; only its destination was wrong."""

        source = (REPO_ROOT / "fedbrew" / "data" / "femnist.py").read_text(encoding="utf-8")
        self.assertNotIn("print(", source)
        self.assertIn("on_progress(", source)

    def test_a_counted_note_reaches_the_stage_as_a_throttled_redraw(self) -> None:
        """Stage.tick is what lets a generator report as often as its loop
        iterates: 24,239 calls become at most ten redraws a second."""

        buffer = io.StringIO()
        surface = build_surface(file=buffer, force_rich=True)
        rail = surface.rail(["generate oasst1_sft"])
        with rail.stage("generate oasst1_sft") as stage:
            for position in range(1, 5_001):
                stage.tick(f"tokenizing {position:,}/5,000 responses")
            stage.done("20 clients")

        rendered = buffer.getvalue()
        self.assertIn("tokenizing", rendered)
        self.assertLess(
            rendered.count("tokenizing"),
            5_000,
            "every call redrew; the throttle did nothing",
        )
        self.assertIn("20 clients", rendered)


@pytest.mark.fast
class PrepareCommandsTest(unittest.TestCase):
    """Both prepare commands' headline gap: the final line was identical
    whether they had re-downloaded everything or found a verified cache."""

    def test_both_declare_the_stages_they_will_report(self) -> None:
        for module in (prepare_module, oasst1_module):
            with self.subTest(module=module.__name__):
                self.assertIn("cache", module.STAGES)
                self.assertTrue(all(module.STAGES), "a stage with no label")

    def test_the_llm_stages_cover_the_facts_the_command_discards(self) -> None:
        """vocabulary size, model type and the resolved commit are all checked
        already and all thrown away. The resolved revision is the most
        actionable fact the command has: a pin that has drifted is invisible
        everywhere else until a run's numbers change."""

        self.assertEqual(prepare_module.STAGES, ("cache", "model", "tokenizer", "revision"))

    def test_the_oasst1_stages_are_per_split_plus_the_verification(self) -> None:
        self.assertEqual(
            oasst1_module.STAGES, ("cache", "train split", "validation split", "verify")
        )

    def test_a_cache_hit_says_so(self) -> None:
        """The one line both commands printed could not distinguish a 30-second
        re-download from a 1.7-second cache verification."""

        buffer = io.StringIO()
        surface = build_surface(file=buffer)
        rail = surface.rail(prepare_module.STAGES)
        with rail.stage("cache") as stage:
            stage.done("verified, nothing to fetch")

        self.assertIn("verified, nothing to fetch", buffer.getvalue())

    def test_the_parameter_count_degrades_rather_than_failing(self) -> None:
        """A header must never be the thing that fails a preparation that
        otherwise succeeded."""

        class _Opaque:
            def parameters(self):
                raise RuntimeError("no parameters here")

        self.assertIsNone(prepare_module._parameter_count(_Opaque()))

    def test_the_parameter_count_is_human_scaled(self) -> None:
        class _Model:
            def __init__(self, total):
                self._total = total

            def parameters(self):
                class _P:
                    def __init__(self, n):
                        self._n = n

                    def numel(self):
                        return self._n

                return [_P(self._total)]

        self.assertEqual(prepare_module._parameter_count(_Model(494_000_000)), "494M parameters")
        self.assertEqual(prepare_module._parameter_count(_Model(1_500_000_000)), "1.5B parameters")
        self.assertEqual(prepare_module._parameter_count(_Model(42)), "42 parameters")


def _minimal_argv(command: str) -> list[str]:
    if command == "run":
        return []
    return ["--config", "unused.yaml"]


def _silenced_stderr():
    import contextlib

    return contextlib.redirect_stderr(io.StringIO())


if __name__ == "__main__":
    unittest.main()
