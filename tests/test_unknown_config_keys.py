"""A key nothing reads is a typo, and a typo must not take the default.

Every optional setting in this codebase is resolved with .get(name, default).
The dataclass split files an unrecognised key into `extra`, the reader never
looks there, and the run proceeds on the default: `client.max_grad_nrom: 1.0`
disabled clipping, `evaluation.val.client: sample:5` evaluated all 3597
clients, `divergence.blowup_facter: 2.0` left the factor at 10, and
`--validate-only` reported READY with no mention of any of them. A model config
that said `lora_alph: 32` ran at 16.

Three layers carry the defect and each is checked here: the run config, the
model builders, and the generator YAML.

The run config's `defaults` block escaped the first layer. It is read at load
and never stored, and the check walks what is stored, so `defaults.bogus_key: 7`
loaded and any key beside the two real ones was dropped. `DefaultsBlockTest`
holds it to the same contract. FINDINGS.csv POST-F16.

The root of the run config escaped too. `load_config` read the blocks it knew
by name and never looked at the rest, so `evaluaton:`, `divergance:` and
`client_statstics:` each loaded and the run took that block's defaults.
`RootKeysTest` and `RootKeysThroughTheCliTest` hold the root to the same
contract, through `fedbrew run` and `--validate-only` as well as
`load_config`. FINDINGS.csv POST-F26.
"""

from __future__ import annotations

import copy
import glob
import json
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.cli import dispatch
from fedbrew.core import extensions, registry
from fedbrew.core.config import DEFAULTS_KEYS, load_config, root_config_keys, validate_config
from fedbrew.core.extensions import load_extensions
from fedbrew.data.generate import _extensions_of, _load_yaml, _validate_generator_keys
from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.models.femnist_resnet import _KNOWN_KEYS as RESNET_KEYS
from fedbrew.models.hf_causal_lm_lora import _KNOWN_KEYS as LORA_KEYS


class RunConfigTest(unittest.TestCase):
    """The six keys the audit probed, each in the section it belongs to."""

    #: section path -> (key as misspelled, value) from the probed config.
    PROBED: tuple[tuple[str, str, Any], ...] = (
        ("client", "max_grad_nrom", 1.0),
        ("evaluation.val", "client", "sample:5"),
        ("runtime.performance", "reuse_modle", False),
        ("runtime", "checkpointng", {"save_best": True}),
        ("server", "aggregation_weight", "uniform"),
        ("divergence", "blowup_facter", 2.0),
    )

    def _with_key(self, section: str, key: str, value: Any) -> Any:
        config = load_config("configs/femnist/fedavg.yaml")
        if section == "runtime.performance":
            config.runtime.extra["performance"][key] = value
        elif section == "evaluation.val":
            config.evaluation.val.extra[key] = value
        elif section == "divergence":
            config.divergence.extra[key] = value
        else:
            getattr(config, section).extra[key] = value
        return config

    @pytest.mark.fast
    def test_every_probed_misspelling_is_refused(self) -> None:
        for section, key, value in self.PROBED:
            with self.subTest(section=section, key=key):
                with self.assertRaises(ValueError) as caught:
                    validate_config(self._with_key(section, key, value))
                self.assertIn(key, str(caught.exception))

    @pytest.mark.fast
    def test_the_message_names_the_section_and_what_it_accepts(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_config(self._with_key("client", "max_grad_nrom", 1.0))
        message = str(caught.exception)
        self.assertIn("client.max_grad_nrom", message)
        # The correct spelling is in the accepted list, so the fix is visible.
        self.assertIn("max_grad_norm", message)

    @pytest.mark.fast
    def test_a_removed_key_is_named_with_its_reason(self) -> None:
        """client.type was checked against the registry and then discarded.

        The factory dispatches on update_rule alone, so client.type: scaffold
        beside update_rule: fedavg validated clean and ran FedAvg.
        """

        config = self._with_key("client", "type", "scaffold")
        with self.assertRaises(ValueError) as caught:
            validate_config(config)
        message = str(caught.exception)
        self.assertIn("client.type", message)
        self.assertIn("update_rule", message)

    @pytest.mark.fast
    def test_a_section_with_no_options_says_so(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_config(self._with_key("divergence", "blowup_facter", 2.0))
        self.assertIn("named fields", str(caught.exception))

    def test_every_shipped_config_still_loads(self) -> None:
        """The allow-lists are a claim about what the readers consume."""

        paths = sorted(glob.glob("configs/**/*.yaml", recursive=True))
        self.assertGreater(len(paths), 20)
        for path in paths:
            if "llm_assets" in path:  # asset fragments, not run configs
                continue
            with self.subTest(path=path):
                validate_config(load_config(path))


@pytest.mark.fast
class DefaultsBlockTest(unittest.TestCase):
    """`defaults` accepts its two keys and refuses everything else."""

    SMOKE = Path(__file__).resolve().parent.parent / "configs" / "dev" / "smoke.yaml"

    def _load_with(self, extra_line: str) -> None:
        text = self.SMOKE.read_text(encoding="utf-8")
        anchor = "  local_iterations: 1\n"
        self.assertEqual(text.count(anchor), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.yaml"
            path.write_text(text.replace(anchor, anchor + extra_line), encoding="utf-8")
            load_config(path)

    def test_the_two_keys_are_the_whole_block(self) -> None:
        self.assertEqual(DEFAULTS_KEYS, {"global_rounds", "local_iterations"})
        self._load_with("")

    def test_an_unknown_key_is_refused_and_the_accepted_ones_named(self) -> None:
        """The probe that loaded: nothing read it, and nothing said so."""

        for line, key in (
            ("  bogus_key: 7\n", "defaults.bogus_key"),
            # Reads as a default for every client; nothing reads it here.
            ("  learning_rate: 0.1\n", "defaults.learning_rate"),
            # A misspelling beside the real key, which alone would be required.
            ("  local_iteration: 20\n", "defaults.local_iteration"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    self._load_with(line)
                message = str(caught.exception)
                self.assertIn(key, message)
                self.assertIn("global_rounds, local_iterations", message)


#: The three root blocks the release audit misspelled, and what each typo cost:
#: the shipped synthetic config's value the loader dropped for its default.
MISSPELLED_ROOT_BLOCKS = (
    ("evaluation", "evaluaton"),
    ("divergence", "divergance"),
    ("client_statistics", "client_statstics"),
)
SYNTHETIC = Path("configs/synthetic/fedavg.yaml")
SMOKE = Path("configs/dev/smoke.yaml")

STRATEGY_EXTENSION = textwrap.dedent(
    """
    from fedbrew.core import registry


    def register():
        registry.server_strategies.register(
            "root_probe_strategy", lambda **kwargs: None, config_keys={"root_probe_knob"}
        )
    """
)


def _write(directory: Path, raw: dict[str, Any], name: str = "run.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _renamed(old: str, new: str) -> dict[str, Any]:
    raw = yaml.safe_load(SYNTHETIC.read_text(encoding="utf-8"))
    raw[new] = raw.pop(old)
    return raw


@pytest.mark.fast
class RootKeysTest(unittest.TestCase):
    """Every top-level key is a block the loader reads, or it is refused."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def test_the_accepted_root_is_the_blocks_and_the_component_paths(self) -> None:
        self.assertEqual(
            root_config_keys(),
            {
                "experiment",
                "defaults",
                "server",
                "client",
                "data",
                "model",
                "runtime",
                "evaluation",
                "client_statistics",
                "divergence",
                "server_config",
                "client_config",
            },
        )

    def test_each_audited_misspelling_is_refused_naming_it_and_the_block_meant(self) -> None:
        for old, new in MISSPELLED_ROOT_BLOCKS:
            with self.subTest(key=new):
                with self.assertRaises(ValueError) as caught:
                    load_config(_write(self.directory, _renamed(old, new)))
                message = str(caught.exception)
                self.assertIn(f"{new!r} (did you mean {old!r}?)", message)
                self.assertIn(", ".join(sorted(root_config_keys())), message)

    def test_a_key_like_no_block_is_refused_without_a_guess(self) -> None:
        raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        raw["zzz_unrelated"] = 1
        with self.assertRaises(ValueError) as caught:
            load_config(_write(self.directory, raw))
        self.assertIn("'zzz_unrelated'. A config may write only these", str(caught.exception))

    def test_a_misspelled_required_block_is_named_rather_than_reported_missing(self) -> None:
        raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        raw["sever"] = raw.pop("server")
        with self.assertRaises(ValueError) as caught:
            load_config(_write(self.directory, raw))
        self.assertIn("'sever' (did you mean 'server'?)", str(caught.exception))

    def test_the_removed_task_block_keeps_its_own_reason(self) -> None:
        raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        raw["task"] = {"name": "classification"}
        with self.assertRaises(ValueError) as caught:
            load_config(_write(self.directory, raw))
        self.assertIn("the top-level task block has been removed", str(caught.exception))

    def test_the_unmodified_config_loads_with_the_values_the_typos_lost(self) -> None:
        config = load_config(SYNTHETIC)
        self.assertEqual(config.evaluation.train.clients, "all")
        self.assertEqual(config.divergence.blowup_absolute, 23.0)

    def test_the_component_path_spelling_still_loads(self) -> None:
        raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        _write(self.directory, raw.pop("server"), "server.yaml")
        _write(self.directory, raw.pop("client"), "client.yaml")
        raw["server_config"] = "server.yaml"
        raw["client_config"] = "client.yaml"
        config = load_config(_write(self.directory, raw))
        self.assertEqual(config.server.strategy, "fedavg")
        self.assertEqual(config.client.update_rule, "local_sgd")

    def test_a_key_an_extension_declares_still_loads(self) -> None:
        entry = self.directory / "strategy.py"
        entry.write_text(STRATEGY_EXTENSION, encoding="utf-8")
        self.addCleanup(extensions._loaded.pop, str(entry.resolve()), None)
        for table in ("_items", "_origins", "_config_keys"):
            self.addCleanup(
                getattr(registry.server_strategies, table).pop, "root_probe_strategy", None
            )
        raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        raw["experiment"]["extensions"] = [str(entry)]
        raw["server"]["strategy"] = "root_probe_strategy"
        raw["server"]["root_probe_knob"] = 3
        config = load_config(_write(self.directory, raw))
        self.assertEqual(config.server.extra["root_probe_knob"], 3)


class RootKeysThroughTheCliTest(unittest.TestCase):
    """The same refusals through `fedbrew run` and `fedbrew run --validate-only`.

    Unmarked: the valid-override case trains the smoke config for two rounds,
    and the shipped examples' extensions run a backward pass when imported.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def _fedbrew(self, *argv: str) -> tuple[int | str | None, str]:
        """Exit code (None when it returned) and everything printed."""

        output = StringIO()
        code: int | str | None = None
        with redirect_stderr(output), redirect_stdout(output):
            try:
                dispatch.main(list(argv))
            except SystemExit as exit_:
                code = exit_.code
        return code, " ".join(output.getvalue().split())

    def _run_refused(self, config: Path) -> str:
        code, text = self._fedbrew(
            "run", "--config", str(config), "--quiet", "--output-dir", str(self.directory / "out")
        )
        self.assertEqual(code, 2)
        self.assertIn("RUN REFUSED", text)
        self.assertFalse((self.directory / "out").exists(), "a refused run wrote its output dir")
        return text

    def test_fedbrew_run_refuses_each_audited_misspelling(self) -> None:
        for old, new in MISSPELLED_ROOT_BLOCKS:
            with self.subTest(key=new):
                text = self._run_refused(_write(self.directory, _renamed(old, new)))
                self.assertIn(f"{new!r} (did you mean {old!r}?)", text)

    def test_validate_only_fails_on_each_audited_misspelling(self) -> None:
        for old, new in MISSPELLED_ROOT_BLOCKS:
            with self.subTest(key=new):
                code, text = self._fedbrew(
                    "run",
                    "--config",
                    str(_write(self.directory, _renamed(old, new))),
                    "--validate-only",
                )
                self.assertEqual(code, 1)
                self.assertIn(f"{new!r} (did you mean {old!r}?)", text)

    def test_a_misspelled_override_flag_is_refused_before_anything_runs(self) -> None:
        """The run CLI's overrides are named flags into known blocks; there is
        no flag that writes an arbitrary top-level key. A misspelled one is
        refused by the parser."""

        code, text = self._fedbrew(
            "run",
            "--config",
            str(SMOKE),
            "--rouns",
            "2",
            "--output-dir",
            str(self.directory / "out"),
        )
        self.assertEqual(code, 2)
        self.assertIn("unrecognized arguments: --rouns", text)
        self.assertFalse((self.directory / "out").exists())

    def test_a_valid_override_applies_through_fedbrew_run(self) -> None:
        out = self.directory / "out"
        code, _ = self._fedbrew(
            "run", "--config", str(SMOKE), "--quiet", "--output-dir", str(out), "--rounds", "2"
        )
        self.assertIsNone(code)
        record = json.loads((out / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["config"]["server"]["global_rounds"], 2)

    def test_every_shipped_run_config_still_loads(self) -> None:
        paths = [
            path
            for path in sorted(
                [*Path("configs").rglob("*.yaml"), *Path("examples").rglob("*.yaml")]
            )
            if isinstance(loaded := yaml.safe_load(path.read_text(encoding="utf-8")), dict)
            and "server" in loaded
        ]
        self.assertGreaterEqual(len(paths), 97)
        for path in paths:
            with self.subTest(path=str(path)):
                load_config(path)


class ModelBuilderTest(unittest.TestCase):
    @pytest.mark.fast
    def test_a_misspelled_hyperparameter_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            reject_unknown_model_keys(
                {"name": "hf_causal_lm_lora", "lora_alph": 32},
                LORA_KEYS,
                "hf_causal_lm_lora",
            )
        message = str(caught.exception)
        self.assertIn("model.lora_alph", message)
        self.assertIn("lora_alpha", message)

    @pytest.mark.fast
    def test_the_factory_injected_keys_are_accepted(self) -> None:
        """name, the three shared dimensions and every dataset_* passthrough."""

        reject_unknown_model_keys(
            {
                "name": "femnist_resnet18",
                "input_dim": 784,
                "hidden_dim": 64,
                "num_classes": 62,
                "dataset_sequence_length": 512,
                "dataset_tokenizer_identifier": "qwen",
                "dropout": 0.1,
            },
            RESNET_KEYS,
            "femnist_resnet18",
        )

    def test_every_shipped_config_builds_its_model(self) -> None:
        from fedbrew.core.factory import _model_config
        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        built = 0
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                config = load_config(path)
            except Exception:
                continue  # asset fragments; covered by RunConfigTest
            with self.subTest(path=path):
                try:
                    models.get(config.model.name)(_model_config(config))
                    built += 1
                except ValueError as exc:
                    self.assertNotIn("does not read", str(exc))
                except Exception:
                    pass  # optional dependency or prepared asset absent
        self.assertGreater(built, 15)


class GeneratorConfigTest(unittest.TestCase):
    @pytest.mark.fast
    def test_a_misspelled_section_is_refused(self) -> None:
        config = _load_yaml("data/configs/femnist_natural.yaml")
        config["client_split"] = config.pop("client_splits")
        with self.assertRaises(ValueError) as caught:
            _validate_generator_keys(config, "femnist")
        self.assertIn("client_split", str(caught.exception))

    @pytest.mark.fast
    def test_a_misspelled_key_inside_a_section_is_refused(self) -> None:
        for section, key in (
            ("partition", "alpah"),
            ("client_splits", "eval_ration"),
            ("dataset", "sed"),
        ):
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(_load_yaml("data/configs/femnist_natural.yaml"))
                config.setdefault(section, {})[key] = 0.5
                with self.assertRaises(ValueError) as caught:
                    _validate_generator_keys(config, "femnist")
                self.assertIn(f"{section}.{key}", str(caught.exception))

    @pytest.mark.fast
    def test_a_section_belonging_to_another_generator_is_refused(self) -> None:
        config = copy.deepcopy(_load_yaml("data/configs/mnist_iid.yaml"))
        config["femnist"] = {"source": "flwrlabs/femnist"}
        with self.assertRaises(ValueError) as caught:
            _validate_generator_keys(config, "mnist")
        self.assertIn("femnist", str(caught.exception))

    @pytest.mark.fast
    def test_a_misspelled_key_in_an_llm_section_is_refused(self) -> None:
        """The four LLM generators need prepared HF assets, so nothing here can
        run them. That is why their behaviour is untested; it was never a
        reason for a typo in their sections to take a default instead."""

        for source, generator, section, key in (
            (
                "data/configs/oasst1_qwen05b_20clients.yaml",
                "oasst1_sft",
                "oasst1_sft",
                "sequence_lenght",
            ),
            (
                "data/configs/oasst1_qwen05b_20clients.yaml",
                "oasst1_sft",
                "tree_splits",
                "client_eval_ration",
            ),
            (
                "data/configs/oasst1_qwen05b_4clients_hpc.yaml",
                "oasst1_sft",
                "pilot_caps",
                "maximum_global_test_respones",
            ),
            (
                "data/configs/medmcqa_qwen05b_20clients.yaml",
                "generic_sft",
                "generic_sft",
                "client_feild",
            ),
            (
                "data/configs/medmcqa_qwen05b_20clients.yaml",
                "generic_sft",
                "caps",
                "train_fration",
            ),
            (
                "data/configs/dev/hf_tiny_gpt2_text.yaml",
                "hf_causal_lm_text",
                "causal_lm",
                "stide",
            ),
        ):
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(_load_yaml(source))
                config.setdefault(section, {})[key] = 1
                with self.assertRaises(ValueError) as caught:
                    _validate_generator_keys(config, generator)
                self.assertIn(f"{section}.{key}", str(caught.exception))

    @pytest.mark.fast
    def test_a_misspelled_key_in_a_nested_block_is_refused(self) -> None:
        """A misspelled option_fields would have rendered a prompt with no
        options and generated a dataset that looked plausible."""

        for source, generator, section, block, key in (
            (
                "data/configs/medmcqa_qwen05b_20clients.yaml",
                "generic_sft",
                "generic_sft",
                "choice",
                "option_feilds",
            ),
            (
                "data/configs/oasst1_qwen05b_4clients_hpc.yaml",
                "oasst1_sft",
                "pilot_caps",
                "maximum_generated_windows_per_split",
                "globl_test",
            ),
        ):
            with self.subTest(block=block, key=key):
                config = copy.deepcopy(_load_yaml(source))
                config[section].setdefault(block, {})[key] = 1
                with self.assertRaises(ValueError) as caught:
                    _validate_generator_keys(config, generator)
                self.assertIn(f"{section}.{block}.{key}", str(caught.exception))

    def test_every_shipped_generator_config_still_passes(self) -> None:
        """Checked the way `generate_from_config` checks it: the config's own
        `dataset.extensions` loaded before its generator is looked up. Without
        that, the ten example configs passed only when an earlier test in the
        same process had loaded their extensions -- serially, `RunConfigTest`
        above -- and failed alone or on an xdist worker that had not.
        FINDINGS.csv POST-F17."""

        paths = sorted(glob.glob("data/configs/**/*.yaml", recursive=True))
        paths = [path for path in paths if "/assets/" not in path]
        self.assertGreaterEqual(len(paths), 12)
        for path in paths:
            with self.subTest(path=path):
                config = _load_yaml(path)
                load_extensions(_extensions_of(config["dataset"]))
                _validate_generator_keys(config, str(config["dataset"]["name"]))


if __name__ == "__main__":
    unittest.main()
