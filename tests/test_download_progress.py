"""Tests for redirecting third-party download progress into fedbrew's own
console output (fedbrew/core/download_progress.py + logging.py's
print_download_progress).

huggingface_hub, datasets and torchvision each drive their download progress
through a tqdm-compatible class bound as a plain module attribute, with no
callback parameter of their own -- see download_progress.py's module
docstring for what was read to establish that. These tests cover the
mechanism against fakes (fast, no real download), plus a two-part guard per
real library: that the attribute this module patches still exists where
expected when the library is installed, so a version bump fails the suite
loudly instead of silently reverting to that library's own bars; and that
the redirect is a working no-op when the library is absent, since all three
are optional extras and CI installs without them.
"""

from __future__ import annotations

import io
import sys
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any

import pytest

from fedbrew.core.download_progress import (
    _driven_tqdm_class,
    _redirect_progress,
    _submodule,
    redirect_datasets_progress,
    redirect_huggingface_hub_progress,
    redirect_torchvision_progress,
)
from fedbrew.core.logging import _format_bytes, _format_download_line, print_download_progress

pytestmark = pytest.mark.fast


class _FakeTqdm:
    """A minimal stand-in for tqdm.tqdm: enough surface for the driven
    subclass to build on (accepts arbitrary kwargs, exposes disable)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.disable = kwargs.get("disable", False)

    def update(self, n: int = 1) -> None:
        raise AssertionError("the driven subclass must override update, not delegate to this")

    def close(self) -> None:
        raise AssertionError("the driven subclass must override close, not delegate to this")


class _Module:
    """A fake module namespace -- just an object with a `tqdm` attribute."""

    def __init__(self, tqdm_cls: Any = _FakeTqdm) -> None:
        self.tqdm = tqdm_cls


@contextmanager
def _library_absent(top_level: str) -> Iterator[None]:
    """Make `top_level` -- and everything under it -- unimportable for the
    block, whether or not it is actually installed.

    Simulated rather than uninstalled, because the same assertions have to
    hold in both CI jobs: the default suite installs the `dev` extra and the
    core-only job installs none, so neither one can be relied on to supply
    the library or to lack it.

    Both halves are needed. Purging sys.modules, because an already-imported
    module is handed straight back from there and never reaches the import
    machinery at all -- with only the finder in place, a library some
    earlier test imported would still resolve. And a meta_path finder that
    raises, because that is where a genuinely missing distribution fails,
    with the same ModuleNotFoundError this raises.
    """

    prefix = f"{top_level}."
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == top_level or name.startswith(prefix)
    }
    for name in saved:
        del sys.modules[name]

    class _Blocker:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
            if fullname == top_level or fullname.startswith(prefix):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        # Restore the objects themselves rather than re-importing: the
        # modules other tests already hold references to must stay the same
        # objects, and nothing here should pay to import torchvision twice.
        sys.modules.update(saved)


class RedirectProgressContextManagerTests(unittest.TestCase):
    def test_restores_the_original_class_after_a_normal_return(self) -> None:
        module = _Module()
        original = module.tqdm

        with _redirect_progress(module, "tqdm", lambda *a, **k: None) as active:
            self.assertTrue(active)
            self.assertIsNot(module.tqdm, original)
            self.assertTrue(issubclass(module.tqdm, original))

        self.assertIs(module.tqdm, original)

    def test_restores_the_original_class_after_an_exception(self) -> None:
        module = _Module()
        original = module.tqdm

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with _redirect_progress(module, "tqdm", lambda *a, **k: None):
                self.assertIsNot(module.tqdm, original)
                raise RuntimeError("boom")

        self.assertIs(module.tqdm, original)

    def test_degrades_when_the_attribute_is_missing(self) -> None:
        module = SimpleNamespace()  # no `tqdm` attribute at all

        with _redirect_progress(module, "tqdm", lambda *a, **k: None) as active:
            self.assertFalse(active)
            self.assertFalse(hasattr(module, "tqdm"))

    def test_degrades_when_the_attribute_is_not_a_class(self) -> None:
        module = SimpleNamespace(tqdm="not a class")

        with _redirect_progress(module, "tqdm", lambda *a, **k: None) as active:
            self.assertFalse(active)
            self.assertEqual(module.tqdm, "not a class")


class DrivenTqdmClassTests(unittest.TestCase):
    def test_reports_name_total_and_accumulated_bytes(self) -> None:
        calls: list[tuple[Any, ...]] = []
        driven = _driven_tqdm_class(
            _FakeTqdm,
            lambda name, done, total, elapsed, *, finished: calls.append(
                (name, done, total, finished)
            ),
        )

        bar = driven(desc="model.safetensors", total=1000, unit="B")
        bar.update(400)
        bar.update(600)
        bar.close()

        names = {call[0] for call in calls}
        self.assertEqual(names, {"model.safetensors"})
        totals = {call[2] for call in calls}
        self.assertEqual(totals, {1000})
        # Constructor call + two updates + close, all finished=False except the
        # very last (close).
        self.assertFalse(calls[-2][3])
        self.assertTrue(calls[-1][3])
        self.assertEqual(calls[-1][1], 1000)

    def test_forces_disable_true_so_the_wrapped_class_never_renders(self) -> None:
        driven = _driven_tqdm_class(_FakeTqdm, lambda *a, **k: None)
        bar = driven(desc="x", total=10)
        self.assertTrue(bar.disable)
        self.assertTrue(bar.kwargs["disable"])

    def test_falls_back_to_a_positional_desc_or_a_default_name(self) -> None:
        driven = _driven_tqdm_class(_FakeTqdm, lambda *a, **k: None)
        with_positional = driven("positional-desc", total=1)
        self.assertEqual(with_positional._fedbrew_name, "positional-desc")

        with_neither = driven(total=1)
        self.assertEqual(with_neither._fedbrew_name, "download")

    def test_rapid_updates_are_throttled_but_close_always_reports(self) -> None:
        calls: list[bool] = []
        driven = _driven_tqdm_class(
            _FakeTqdm,
            lambda name, done, total, elapsed, *, finished: calls.append(finished),
        )
        bar = driven(desc="x", total=1_000_000)
        call_count_after_init = len(calls)
        for _ in range(50):
            bar.update(1)  # far under the 0.1s throttle window, back-to-back
        call_count_after_updates = len(calls)
        bar.close()

        # The 50 rapid updates must not have produced 50 more calls -- that's
        # the whole point of throttling a callback driven from a byte-level
        # tqdm.update() -- but close() must always report regardless.
        self.assertLess(call_count_after_updates - call_count_after_init, 50)
        self.assertTrue(calls[-1])

    def test_close_is_idempotent_a_second_call_does_not_report_again(self) -> None:
        """Regression: observed against a real torchvision MNIST download --
        tqdm's own __exit__ (from torchvision's `with tqdm(...) as pbar:`)
        and its __del__ safety net both call close(), and every file was
        reported "finished" twice before this guard existed."""

        calls: list[bool] = []
        driven = _driven_tqdm_class(
            _FakeTqdm,
            lambda name, done, total, elapsed, *, finished: calls.append(finished),
        )
        bar = driven(desc="x", total=10)
        bar.update(10)
        bar.close()
        bar.close()
        bar.close()

        self.assertEqual(calls.count(True), 1)

    def test_a_raising_callback_is_swallowed_not_propagated(self) -> None:
        """A bug in our own reporting must never look like the download
        itself failed -- callers of the redirected functions treat most
        exceptions from the wrapped call as "the library failed, fall back"
        (generate.py's MNIST loader in particular) and would mis-attribute
        one here."""

        def _explode(*args: Any, **kwargs: Any) -> None:
            raise ValueError("formatting bug")

        driven = _driven_tqdm_class(_FakeTqdm, _explode)
        bar = driven(desc="x", total=10)  # must not raise
        bar.update(5)  # must not raise
        bar.close()  # must not raise


class RedirectFactoryTests(unittest.TestCase):
    """The three named factories target the exact module + attribute the
    module docstring says each library resolves its download progress
    through, and must survive that library being missing entirely.

    Two tests per library. The "targets a real class" half is skipped if
    that (optional) library isn't installed -- matching this repo's existing
    pattern for LLM/vision extras (see tests/test_llm_asset_preparation.py).
    The "no-op when absent" half is skipped for nothing: absence is
    simulated, so it runs identically on an install with the extras and on
    one without.

    Regression for the second half: `_submodule` used to let
    ModuleNotFoundError escape, so on any install without the `llm` extra
    `redirect_huggingface_hub_progress` raised before `from_pretrained` was
    ever called -- red in both CI jobs, and invisible to a `.[llm]` machine
    running the suite locally.
    """

    def _real_submodule(self, dotted_name: str) -> Any:
        """The actual submodule object, via the same importlib route
        download_progress.py's own `_submodule` uses.

        Not `import a.b.c; a.b.c`: huggingface_hub's and datasets' own
        `utils/__init__.py` both do `from .tqdm import ..., tqdm, ...`,
        which rebinds the package's `tqdm` attribute to the *class* and
        shadows the submodule there -- confirmed the hard way, this test
        class originally did plain attribute access and got
        `AttributeError: type object 'tqdm' has no attribute 'tqdm'`.
        """

        import importlib

        return importlib.import_module(dotted_name)

    def _assert_no_op_when_absent(self, top_level: str, dotted_name: str, redirect: Any) -> None:
        """With `top_level` absent, `redirect` must still be a usable context
        manager that reports False and reports no progress."""

        with _library_absent(top_level):
            self.assertIsNone(
                _submodule(dotted_name),
                f"{top_level} absence was not simulated -- the rest would pass vacuously",
            )

            reported: list[Any] = []
            body_ran = False
            with redirect(lambda *a, **k: reported.append(a)) as active:
                self.assertFalse(active, f"nothing to patch with {top_level} absent")
                body_ran = True

        self.assertTrue(body_ran, "the no-op must still run the block it wraps")
        self.assertEqual(reported, [], "an absent library cannot report any progress")

    def test_huggingface_hub_redirect_is_a_no_op_when_the_library_is_absent(self) -> None:
        self._assert_no_op_when_absent(
            "huggingface_hub", "huggingface_hub.utils.tqdm", redirect_huggingface_hub_progress
        )

    def test_datasets_redirect_is_a_no_op_when_the_library_is_absent(self) -> None:
        self._assert_no_op_when_absent(
            "datasets", "datasets.utils.tqdm", redirect_datasets_progress
        )

    def test_torchvision_redirect_is_a_no_op_when_the_library_is_absent(self) -> None:
        self._assert_no_op_when_absent(
            "torchvision", "torchvision.datasets.utils", redirect_torchvision_progress
        )

    def test_huggingface_hub_redirect_targets_a_real_class(self) -> None:
        try:
            module = self._real_submodule("huggingface_hub.utils.tqdm")
        except ModuleNotFoundError:
            self.skipTest("huggingface_hub not installed")

        original = module.tqdm
        self.assertIsInstance(original, type, "huggingface_hub.utils.tqdm.tqdm is not a class")
        with redirect_huggingface_hub_progress(lambda *a, **k: None) as active:
            self.assertTrue(active, "huggingface_hub moved its tqdm attribute")
            self.assertIsNot(module.tqdm, original)
            self.assertTrue(issubclass(module.tqdm, original))
        self.assertIs(module.tqdm, original)

    def test_datasets_redirect_targets_a_real_class(self) -> None:
        try:
            module = self._real_submodule("datasets.utils.tqdm")
        except ModuleNotFoundError:
            self.skipTest("datasets not installed")

        original = module.tqdm
        self.assertIsInstance(original, type, "datasets.utils.tqdm.tqdm is not a class")
        with redirect_datasets_progress(lambda *a, **k: None) as active:
            self.assertTrue(active, "datasets moved its tqdm attribute")
            self.assertIsNot(module.tqdm, original)
            self.assertTrue(issubclass(module.tqdm, original))
        self.assertIs(module.tqdm, original)

    def test_torchvision_redirect_targets_a_real_class(self) -> None:
        try:
            module = self._real_submodule("torchvision.datasets.utils")
        except ModuleNotFoundError:
            self.skipTest("torchvision not installed")

        original = module.tqdm
        self.assertIsInstance(original, type, "torchvision.datasets.utils.tqdm is not a class")
        with redirect_torchvision_progress(lambda *a, **k: None) as active:
            self.assertTrue(active, "torchvision moved its tqdm attribute")
            self.assertIsNot(module.tqdm, original)
            self.assertTrue(issubclass(module.tqdm, original))
        self.assertIs(module.tqdm, original)


class FormatDownloadLineTests(unittest.TestCase):
    def test_bytes_scale_through_the_1000_based_units(self) -> None:
        self.assertEqual(_format_bytes(0), "0B")
        self.assertEqual(_format_bytes(512), "512B")
        self.assertEqual(_format_bytes(1_500), "1.5KB")
        self.assertEqual(_format_bytes(2_500_000), "2.5MB")
        self.assertEqual(_format_bytes(3_500_000_000), "3.5GB")

    def test_percent_bytes_and_rate_are_all_present(self) -> None:
        line = _format_download_line("model.safetensors", 500_000, 1_000_000, 1.0)
        self.assertIn("model.safetensors", line)
        self.assertIn(" 50%", line)
        self.assertIn("500.0KB/1.0MB", line)
        self.assertIn("500.0KB/s", line)

    def test_percent_is_a_question_mark_without_a_known_total(self) -> None:
        line = _format_download_line("data.parquet", 500_000, None, 1.0)
        self.assertIn("?%", line)

    def test_percent_is_clamped_at_100(self) -> None:
        line = _format_download_line("x", 1_100, 1_000, 1.0)
        self.assertIn("100%", line)

    def test_rate_is_a_placeholder_before_enough_time_has_passed(self) -> None:
        line = _format_download_line("x", 10, 1_000, 0.0)
        self.assertIn("--B/s", line)


class PrintDownloadProgressTests(unittest.TestCase):
    """print_download_progress builds its own console via _build_console();
    under pytest's captured, non-interactive stdout, Console().is_terminal is
    False, so these exercise the non-interactive branch -- one line, only on
    completion. The interactive (live single-line, carriage-return-redrawn)
    branch is not exercised by any existing test in this suite either (no
    test here forces a real TTY), matching precedent."""

    def test_intermediate_progress_prints_nothing_when_not_a_terminal(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_download_progress("model.bin", 100, 1000, 0.5, finished=False)
        self.assertEqual(buffer.getvalue(), "")

    def test_completion_prints_exactly_one_line(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_download_progress("model.bin", 1000, 1000, 2.0, finished=True)
        output = buffer.getvalue()
        self.assertEqual(output.count("\n"), 1)
        self.assertIn("model.bin", output)
        self.assertIn("100%", output)


if __name__ == "__main__":
    unittest.main()
