"""An empty experiment.output_dir is refused at load, not just at preflight.

Path("") is Path("."), so an empty value writes run.json, round_metrics.csv,
the per-client CSVs and every checkpoint into the working directory: for an
HPC job, whatever the submit script last cd'd to; for an interactive run, the
repository. Nothing fails and nothing warns. The artifacts are just somewhere
else, which for a batch job means somewhere nobody looks.

validate_full_config has reported it as experiment.output_dir_empty for a long
time, and that module is reached only by --validate-only, so the report did
not stop a run. The refusal is now in validate_config, which load_config
calls.

"." is deliberately still accepted: someone who writes it has said where they
want the artifacts. The value refused is the one nobody chose.
"""

from __future__ import annotations

import copy
import glob
import tempfile
import unittest
from pathlib import Path

import pytest

from fedbrew.core.config import load_config, validate_config

BASE = "configs/dev/smoke.yaml"


def _refusal(config) -> str | None:
    try:
        validate_config(config)
    except ValueError as error:
        return str(error)
    return None


def _with(value: object):
    config = copy.deepcopy(load_config(BASE))
    config.experiment.output_dir = value
    return config


@pytest.mark.fast
class RefusedTest(unittest.TestCase):
    def test_empty(self) -> None:
        message = _refusal(_with(""))
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("output_dir", message)
        self.assertIn("working directory", message)

    def test_whitespace_only(self) -> None:
        """Not empty, so it survives an emptiness check, and it makes a
        directory whose name is the spaces."""

        self.assertIsNotNone(_refusal(_with("   ")))

    def test_not_a_string(self) -> None:
        message = _refusal(_with(3))
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("must be a string", message)


@pytest.mark.fast
class AcceptedTest(unittest.TestCase):
    """A check that fires on a deliberate choice is one people turn off."""

    def test_an_explicit_dot(self) -> None:
        self.assertIsNone(_refusal(_with(".")))

    def test_an_ordinary_relative_path(self) -> None:
        self.assertIsNone(_refusal(_with("outputs/experiment")))

    def test_a_path_with_an_environment_variable(self) -> None:
        """Preflight warns about an unexpanded one; loading must not refuse it."""

        self.assertIsNone(_refusal(_with("$FL_OUTPUT_ROOT/run")))


@pytest.mark.fast
class ItIsOnTheRunPathTest(unittest.TestCase):
    def test_load_config_itself_refuses(self) -> None:
        import yaml

        raw = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
        raw["experiment"]["output_dir"] = ""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(raw, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError) as caught:
                load_config(path)
            self.assertIn("output_dir", str(caught.exception))
        finally:
            Path(path).unlink()


class NoShippedConfigTripsItTest(unittest.TestCase):
    def test_every_config_still_loads(self) -> None:
        offenders = []
        checked = 0
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                load_config(path)
            except ValueError as error:
                if "output_dir" in str(error):
                    offenders.append(f"{path}  {error}")
                continue
            except Exception:  # noqa: BLE001 - generator configs are not run configs
                continue
            checked += 1
        self.assertGreater(checked, 20, "the config scan found almost nothing")
        self.assertEqual(offenders, [], f"shipped configs trip the new check: {offenders}")


if __name__ == "__main__":
    unittest.main()
