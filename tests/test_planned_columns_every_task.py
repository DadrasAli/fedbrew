"""The plan header's column list is what every shipped config writes, for every task.

``_planned_metric_names`` used to assume classification: on every one of the
69 example arms it promised ``central_test_accuracy`` and six
``test_accuracy_*`` columns the run never wrote, and left out every ``fit_*``
column and every ``central_test_<metric>`` the task does write. It is now
built from what the task declares (``TaskAdapter.METRICS``) and what the rule
adds (``RULE_FIT_METRICS``); ``tests/test_planned_columns_are_written.py``
round-trips every rule on classification.

This one covers every shipped config. The column list depends only on what
the planner reads -- the task, the update rule, the two metric lists, which
splits and passes are scheduled, the model scope and the client statistics --
so the 97 configs fall into a few dozen groups that share all of it. One
representative of each group is run for a round on cheap data that keeps
every one of those: an example arm on its own family's generated dataset; a
classification config on a small generated synthetic manifest with an MLP,
with or without a val split to match the group; a causal-LM config on the
tiny corpus with tiny_gpt2 (when the LLM extra is installed). The planned
columns and the written CSV header must agree in both directions.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import tempfile
import textwrap
import unittest
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from fedbrew.core import runner
from fedbrew.core.config import (
    FullConfig,
    is_family_base,
    load_config,
    parse_evaluation_schedule,
    standalone_config_mapping,
)
from fedbrew.core.logging import _planned_metric_names
from fedbrew.data.generate import generate_from_config
from tests.test_planned_columns_are_written import BOOKKEEPING

REPO = Path(__file__).resolve().parent.parent
HAS_LLM = importlib.util.find_spec("transformers") is not None


def _scheduled(every: Any, name: str) -> bool:
    return parse_evaluation_schedule(every, name) is not None


def _signature(config: FullConfig) -> tuple[Any, ...]:
    """Everything `_planned_metric_names` reads from a config."""

    statistics = asdict(config.client_statistics)
    statistics.pop("per_client_csv", None)
    statistics.pop("extra", None)
    evaluation = config.evaluation
    return (
        config.task.name,
        config.client.update_rule,
        config.server.strategy,
        tuple(config.server.metrics),
        tuple(config.client.metrics),
        tuple(_scheduled(getattr(evaluation, s).every, s) for s in ("train", "val", "test")),
        _scheduled(evaluation.central_test.every, "central_test"),
        _scheduled(evaluation.fit.every, "fit"),
        evaluation.model_scope,
        tuple(sorted(statistics.items())),
    )


def _shipped_groups() -> dict[tuple[Any, ...], list[Path]]:
    groups: dict[tuple[Any, ...], list[Path]] = {}
    for path in sorted((REPO / "configs").rglob("*.yaml")):
        if "llm_assets" in path.parts or is_family_base(path):
            continue
        groups.setdefault(_signature(load_config(path)), []).append(path)
    return groups


class _Data:
    """The cheap datasets, generated once for the module."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._made: dict[str, Path] = {}

    def _generate(self, key: str, text: str) -> Path:
        if key not in self._made:
            config = self.root / f"{key}.yaml"
            config.write_text(text, encoding="utf-8")
            self._made[key] = Path(generate_from_config(config))
        return self._made[key]

    def classification(self, with_val: bool) -> Path:
        client_splits = "client_splits: {train_ratio: 0.8, eval_ratio: 0.2}" if with_val else ""
        output = self.root / f"cls-{with_val}"
        return self._generate(
            f"classification-{with_val}",
            textwrap.dedent(f"""
                dataset: {{name: synthetic_classification, output_dir: {output}, seed: 3}}
                synthetic: {{num_samples: 240, input_dim: 5, num_classes: 3}}
                partition: {{strategy: iid, num_clients: 4}}
                splits: {{train_ratio: 0.8, test_ratio: 0.2}}
                {client_splits}
                """),
        )

    def causal_lm(self) -> Path:
        corpus = self.root / "corpus.txt"
        corpus.write_text(
            "\n".join(f"record {i:03d} contains deterministic local text" for i in range(120)),
            encoding="utf-8",
        )
        return self._generate(
            "causal_lm",
            textwrap.dedent(f"""
                dataset: {{name: tiny_causal_lm, output_dir: {self.root / "lm"}, seed: 9}}
                causal_lm: {{corpus_path: {corpus}, sequence_length: 32}}
                splits: {{train_ratio: 0.8, test_ratio: 0.2}}
                client_splits: {{train_ratio: 0.8, eval_ratio: 0.2}}
                partition: {{strategy: iid, num_clients: 2}}
                """),
        )

    def example(self, family: str) -> Path:
        source = REPO / "data" / "configs" / "examples" / f"{family}.yaml"
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["dataset"]["output_dir"] = str(self.root / "examples" / family)
        extensions = config["dataset"].get("extensions") or config.get("extensions")
        if extensions:
            absolute = [str(REPO / path) for path in extensions]
            if "extensions" in config["dataset"]:
                config["dataset"]["extensions"] = absolute
            else:
                config["extensions"] = absolute
        return self._generate(f"example-{family}", yaml.safe_dump(config))


def _runnable(path: Path, data: _Data, output: Path) -> Path:
    """The shipped config at `path`, on cheap data, for one round."""

    raw = standalone_config_mapping(path)
    config = load_config(path)
    raw["experiment"]["output_dir"] = str(output)
    raw["experiment"]["use_run_subdir"] = False
    raw["experiment"]["extensions"] = [
        str(REPO / extension) for extension in raw["experiment"].get("extensions") or []
    ]
    raw["defaults"]["global_rounds"] = 1
    runtime = raw["runtime"]
    runtime["device"] = "cpu"
    runtime.pop("data_staging", None)
    if config.task.name == "classification":
        val = _scheduled(config.evaluation.val.every, "val")
        raw["data"] = {"path": str(data.classification(with_val=val))}
        raw["model"] = {"name": "mlp", "input_dim": 5, "hidden_dim": 8, "num_classes": 3}
    elif config.task.name == "causal_lm":
        tiny = yaml.safe_load((REPO / "configs/dev/tiny_causal_lm.yaml").read_text())
        raw["data"] = {"path": str(data.causal_lm())}
        raw["model"] = tiny["model"]
    else:
        raw["data"] = {"path": str(data.example(path.parent.name))}
    written = output.parent / f"{output.name}.yaml"
    written.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return written


def _written(path: Path) -> set[str]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        runner.run(path, runner.parse_args(["--quiet"]))
    config = load_config(path)
    with (Path(config.experiment.output_dir) / "round_metrics.csv").open(newline="") as handle:
        return set(next(csv.reader(handle))) - BOOKKEEPING


class EveryShippedConfigRoundTripsTest(unittest.TestCase):
    def _representatives(self) -> Iterator[tuple[str, Path]]:
        for paths in _shipped_groups().values():
            yield f"{paths[0].relative_to(REPO)} (+{len(paths) - 1})", paths[0]

    def test_every_group_of_shipped_configs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _Data(root / "data")
            (root / "data").mkdir()
            for index, (label, path) in enumerate(self._representatives()):
                with self.subTest(config=label):
                    if load_config(path).task.name == "causal_lm" and not HAS_LLM:
                        self.skipTest("causal-LM configs need the llm extra")
                    runnable = _runnable(path, data, root / f"run{index}")
                    planned = set(_planned_metric_names(load_config(runnable)))
                    self.assertEqual(
                        planned,
                        set(_planned_metric_names(load_config(path))),
                        "the cheap stand-in plans other columns than the shipped config",
                    )
                    written = _written(runnable)
                    self.assertEqual(planned - written, set(), "planned, not written")
                    self.assertEqual(written - planned, set(), "written, not planned")


class EveryTaskDeclaresItsMetricsTest(unittest.TestCase):
    """Each shipped task says what it reports, so none is planned as classification."""

    def test_every_task_a_shipped_config_uses(self) -> None:
        from fedbrew.core.registry import register_builtin_components, tasks

        register_builtin_components()
        names = {load_config(paths[0]).task.name for paths in _shipped_groups().values()}
        self.assertEqual(len(names), 7)
        for name in sorted(names):
            with self.subTest(task=name):
                self.assertTrue(tasks.metrics(name), f"{name} declares no metrics")


if __name__ == "__main__":
    unittest.main()
