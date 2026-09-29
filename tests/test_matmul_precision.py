"""matmul_precision changes the numbers, and is in the numerics block for it.

torch.set_float32_matmul_precision("high") puts every fp32 matmul on
TensorFloat32 (10 stored mantissa bits) or a bfloat16 pair (~16), against 24
for "highest" -- in every forward and backward pass. It sat in a block whose
comment said "none of these change results", and run.json recorded the value
the config asked for rather than the one torch held.

torch also does not reject an unknown value: it emits a UserWarning and keeps
the current setting, so `matmul_precision: hihg` trained at "highest" while
run.json recorded "hihg".
"""

from __future__ import annotations

import glob
import subprocess
import sys
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.core.config import MATMUL_PRECISIONS, load_config, validate_config
from fedbrew.core.runtime_setup import configure_runtime

#: Where the documentation has to keep saying that matmul_precision changes
#: results. The section anchor keeps its leading hashes so a table-of-contents
#: link to the same section cannot be mistaken for the heading itself.
#:
#: This moved from README.md to docs/10 when the reproducibility material did.
#: The guard follows the claim rather than the file: leaving it pointed at a
#: README section that no longer exists would have made it fail for the wrong
#: reason, and deleting it would have let the claim move somewhere no reader
#: checking reproducibility would look.
_DOC = "docs/10-reproducibility.md"
_REPRODUCIBILITY_HEADING = "## Reproducibility"
_CHANGES_NUMERICS_HEADING = "**do** change numerics"


@pytest.mark.fast
class MatmulPrecisionValueTest(unittest.TestCase):
    def _config(self, precision: object) -> object:
        config = load_config("configs/femnist/fedavg.yaml")
        config.numerics.matmul_precision = precision
        return config

    def test_a_typo_is_refused_at_config_load(self) -> None:
        for precision in ("hihg", "HIGH", "fp32", 32, True):
            with self.subTest(precision=precision):
                with self.assertRaises(ValueError) as caught:
                    validate_config(self._config(precision))
                self.assertIn("matmul_precision", str(caught.exception))

    def test_every_supported_value_is_accepted(self) -> None:
        self.assertEqual(MATMUL_PRECISIONS, {"highest", "high", "medium"})
        for precision in sorted(MATMUL_PRECISIONS):
            with self.subTest(precision=precision):
                validate_config(self._config(precision))

    def test_leaving_it_unset_is_still_allowed(self) -> None:
        config = load_config("configs/femnist/fedavg.yaml")
        config.numerics.matmul_precision = None
        validate_config(config)


class TorchAcceptsATypoTest(unittest.TestCase):
    """Not in the fast gate: it starts a child interpreter."""

    def test_torch_itself_does_not_reject_a_typo(self) -> None:
        """The premise of the guard, pinned so it is not taken on trust.

        Checked in a child interpreter. With the pip wheel of torch 2.5.1+cpu,
        an unknown value leaves the process corrupted: the next
        `torch.library.Library(...)` in it -- importing torchvision or
        torch._dynamo does one -- fails with "could not parse dispatch key:
        hihg", so under pytest-xdist whichever test a worker ran next failed.
        """

        script = (
            "import warnings, torch\n"
            "before = torch.get_float32_matmul_precision()\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    torch.set_float32_matmul_precision('hihg')\n"
            "assert any(issubclass(w.category, UserWarning) for w in caught), caught\n"
            "assert torch.get_float32_matmul_precision() == before\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


@pytest.mark.fast
class MatmulPrecisionIsRecordedTest(unittest.TestCase):
    """run.json has to say what the run did, not what it was asked to do."""

    def setUp(self) -> None:
        self._before = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, self._before)

    def _runtime(self, precision: str | None) -> dict[str, object]:
        config = load_config("configs/femnist/fedavg.yaml")
        config.numerics.matmul_precision = precision
        config.runtime.device = "cpu"
        # Passed explicitly since configure_runtime stopped reading it: the
        # runner owns the one read, and this is what it would hand over.
        return configure_runtime(config, config.numerics.deterministic)

    def test_the_configured_value_is_the_one_reported(self) -> None:
        for precision in ("highest", "high", "medium"):
            with self.subTest(precision=precision):
                self.assertEqual(self._runtime(precision)["matmul_precision"], precision)

    def test_an_unset_key_reports_what_torch_holds_not_none(self) -> None:
        """The gap the old code left: unset recorded null, torch held highest."""

        torch.set_float32_matmul_precision("highest")
        self.assertEqual(self._runtime(None)["matmul_precision"], "highest")


@pytest.mark.fast
class HighestTurnsCudnnTf32OffTest(unittest.TestCase):
    """matmul_precision reaches convolutions too: cuDNN's TF32 switch follows it.

    torch's matmul setting leaves cuDNN alone, and cuDNN allows TF32 by
    default, so a CUDA run at "highest" ran every convolution in TF32 while
    run.json said "highest".
    """

    def setUp(self) -> None:
        precision, allowed = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
        self.addCleanup(torch.set_float32_matmul_precision, precision)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", allowed)

    def _runtime(self, precision: str | None) -> dict[str, object]:
        return MatmulPrecisionIsRecordedTest._runtime(self, precision)  # type: ignore[arg-type]

    def test_highest_turns_it_off_and_the_others_on(self) -> None:
        for precision, allowed in (
            ("highest", False),
            ("high", True),
            ("medium", True),
            ("highest", False),
        ):
            with self.subTest(precision=precision):
                record = self._runtime(precision)
                self.assertIs(torch.backends.cudnn.allow_tf32, allowed)
                self.assertIs(record["cudnn_allow_tf32"], allowed)

    def test_an_unset_key_leaves_it_and_records_what_cudnn_holds(self) -> None:
        for held in (True, False):
            with self.subTest(held=held):
                torch.backends.cudnn.allow_tf32 = held
                record = self._runtime(None)
                self.assertIs(torch.backends.cudnn.allow_tf32, held)
                self.assertIs(record["cudnn_allow_tf32"], held)

    def test_the_documentation_says_so(self) -> None:
        text = Path(_DOC).read_text(encoding="utf-8")
        self.assertIn("`cudnn_allow_tf32`", text)
        self.assertIn("**It covers convolutions too.**", text)


@pytest.mark.fast
class PerformanceBlockDocumentationTest(unittest.TestCase):
    def test_no_config_still_claims_the_whole_block_is_free(self) -> None:
        """The comment used to read "none of these change results"."""

        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            text = Path(path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertNotIn("none of these change results", text)

    def test_every_config_setting_it_says_what_it_costs(self) -> None:
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            text = Path(path).read_text(encoding="utf-8")
            if "matmul_precision:" not in text:
                continue
            with self.subTest(path=path):
                self.assertIn("NOT throughput-only", text)

    def test_the_documentation_lists_it_as_changing_numerics(self) -> None:
        """The prose that says so must not drift away from the setting.

        Reinstated after the first `docs/` was deleted for public release,
        when it read that set's performance chapter; retargeted from README.md
        to docs/10-reproducibility.md when the reproducibility material moved
        there. The invariant is unchanged across all three and is not about
        wording: matmul_precision silently changes every fp32 matmul, so the
        user-facing document has to keep saying so, in the section a reader
        checking reproducibility would actually open.

        Anchored on the two headings rather than on a sentence, so the prose
        stays free to be rewritten. Both anchors are asserted unique first --
        `split(..., 1)[1]` takes the text after the FIRST occurrence, so a
        table-of-contents entry or a second mention would silently move the
        window being searched and the assertion would stop meaning anything.
        """

        doc = Path(_DOC).read_text(encoding="utf-8")

        self.assertEqual(
            doc.count(_REPRODUCIBILITY_HEADING),
            1,
            f"{_DOC} must contain exactly one {_REPRODUCIBILITY_HEADING!r} "
            "heading for this test to locate the right section",
        )
        reproducibility = doc.split(_REPRODUCIBILITY_HEADING, 1)[1]

        self.assertEqual(
            doc.count(_CHANGES_NUMERICS_HEADING),
            1,
            f"{_DOC} must contain exactly one {_CHANGES_NUMERICS_HEADING!r} "
            "subheading, inside the reproducibility section",
        )
        changes_numerics = reproducibility.split(_CHANGES_NUMERICS_HEADING, 1)[1]

        # Stop at the next heading of any level: the claim has to be inside
        # this subsection, not merely somewhere later in the file.
        subsection = changes_numerics.split("##", 1)[0]
        self.assertIn("matmul_precision", subsection)


if __name__ == "__main__":
    unittest.main()
