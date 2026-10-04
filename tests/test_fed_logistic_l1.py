"""examples/fed-logistic-l1: its corpora, its problems, and the table of certified optima.

What is held:

- per problem (``problem.PROBLEMS``), on a small synthetic corpus generated
  here (d = 8, 4 clients of 16 rows): the gradient autograd takes of the task's
  ``functional_loss`` is the analytic ``gradient``, at random points, to 1e-14
  relative; a convex problem's looked-up ``F*`` is ``F`` at its ``x*``,
  certified, and a nonconvex one reports no gap; one FedAvg round, batched,
  agrees with the sequential round within the batched executor's ``1e-12``;
- the lookup refuses: a convex problem with no entry, an entry for the corpus
  at another digest, and shards that are not the digest their manifest records;
- the shipped table: every convex arm has its entry, and each entry's ``F*`` is
  the value that setting's own manifest held when every setting was its own
  generated dataset; on the synthetic corpora, generated here, each convex
  arm's task finds that same value by its corpus's digest where this build
  makes the shipped rows, and on any other build the F* ``fedbrew generate``
  certified beside the data on demand, to the same KKT bound;
- certified on demand: a corpus whose digest the shipped table lacks gets the
  shipped table's problems for its name certified beside its data, marked
  with the machine and build; the shipped table is unchanged; the task and the
  planned columns read the shipped table first and then the local one;
- a conditioned design has its condition number, whatever the thread count;
- a LIBSVM source is read as its rows, pinned by its digest, and dealt by a key
  that does not depend on the order a row is summed in.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.core import extensions
from fedbrew.core.config import load_config, load_config_mapping, standalone_config_mapping
from fedbrew.core.logging import _planned_metric_names
from fedbrew.data.writers.torch_shards import load_client_shard, save_client_shard
from tests.test_batched_executor_tolerance import CSVS, TOLERANCE, ExecutorRuns
from tests.test_grad_norm import recorded_tolerances, within_the_rule
from tests.test_planned_columns_are_written import BOOKKEEPING

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTENSION = REPO_ROOT / "examples" / "fed-logistic-l1" / "problem.py"
ARM = (
    REPO_ROOT / "configs" / "examples" / "fed-logistic-l1-synthetic" / "logistic-l1-lambda0.03.yaml"
)
GENERATOR_CONFIGS = REPO_ROOT / "data" / "configs" / "examples"

problem = extensions._import_file(EXTENSION)

#: Each convex setting's F*, as its own manifest held it when every setting
#: was its own generated dataset (measured on 2026-09-28, the step that moved
#: them into the table; the ijcnn1-32 value is its generator's, not the
#: release's 5.6e-17 higher one).
PREVIOUS_F_STAR = {
    ("fed-logistic-l1-a9a", "logistic", "l1", 0.001): 0.3468377180429282,
    ("fed-logistic-l1-a9a", "logistic", "l1", 0.03): 0.5291036251024969,
    ("fed-logistic-l1-a9a", "logistic", "l1", 0.05): 0.5765317125957747,
    ("fed-logistic-l1-a9a", "logistic", "l2sq", 0.001): 0.3330952806480848,
    ("fed-logistic-l1-gisette", "logistic", "l1", 0.001): 0.521996719845196,
    ("fed-logistic-l1-gisette", "logistic", "l1", 0.0005): 0.4220170868705644,
    ("fed-logistic-l1-gisette", "logistic", "l1", 5e-05): 0.16230244614601907,
    ("fed-logistic-l1-gisette", "logistic", "l2sq", 0.001): 0.4580582604022483,
    ("fed-logistic-l1-ijcnn1-32", "logistic", "l1", 0.01): 0.42788364502097165,
    ("fed-logistic-l1-ijcnn1-32", "logistic", "l2sq", 0.01): 0.4154526395889211,
    ("fed-logistic-l1-synthetic", "logistic", "l1", 0.03): 0.5121519380638806,
    ("fed-logistic-l1-synthetic-1000", "logistic", "l1", 0.001): 0.42993436284235004,
    ("fed-logistic-l1-synthetic-1000", "logistic", "l1", 0.03): 0.5121661415511938,
    ("fed-logistic-l1-synthetic-1000", "logistic", "l2sq", 0.001): 0.42893336113321356,
    ("fed-logistic-l1-synthetic-kappa1", "logistic", "l1", 0.01): 0.45512445647395333,
    ("fed-logistic-l1-synthetic-kappa1", "logistic", "l2sq", 0.01): 0.4409334313437286,
    ("fed-logistic-l1-synthetic-kappa10", "logistic", "l1", 0.01): 0.4738248007229769,
    ("fed-logistic-l1-synthetic-kappa10", "logistic", "l2sq", 0.01): 0.4593573490876512,
    ("fed-logistic-l1-synthetic-kappa100", "logistic", "l1", 0.01): 0.48743008220058726,
    ("fed-logistic-l1-synthetic-kappa100", "logistic", "l2sq", 0.01): 0.47750977596974997,
}

#: The corpora cheap enough to generate in a test: seconds, no source file.
CHEAP_CORPORA = (
    "fed-logistic-l1-synthetic",
    "fed-logistic-l1-synthetic-kappa1",
    "fed-logistic-l1-synthetic-kappa10",
    "fed-logistic-l1-synthetic-kappa100",
)

_root = tempfile.TemporaryDirectory()
_small: dict[str, Path] = {}


def _generate(config: dict[str, Any], directory: Path) -> Path:
    from fedbrew.data.generate import generate_from_config

    config = json.loads(json.dumps(config))
    config["dataset"]["output_dir"] = str(directory / "data")
    config["dataset"]["extensions"] = [str(EXTENSION)]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "generator.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return Path(generate_from_config(path))


def small_corpus() -> tuple[Path, Path]:
    """A small synthetic corpus and a table certifying its convex problems, once per process."""

    if not _small:
        directory = Path(_root.name) / "small"
        _small["manifest"] = _generate(
            {
                "dataset": {"name": "fed_logistic_l1", "seed": 42},
                "partition": {"strategy": "analytic", "num_clients": 4},
                "problem": {
                    "corpus": "small",
                    "dim": 8,
                    "rows_per_client": 16,
                    "partition_block": 4,
                },
            },
            directory,
        )
        _small["table"] = directory / "optima.json"
        for loss, penalty in problem.PROBLEMS:
            if problem.convex(loss, penalty):
                problem.write_optimum(_small["table"], problem.certify(_small_spec(loss, penalty)))
    return _small["manifest"], _small["table"]


def _small_spec(loss: str, penalty: str) -> Any:
    return problem.ProblemSpec(
        num_clients=4,
        dim=8,
        rows_per_client=16,
        partition_block=4,
        corpus="small",
        loss=loss,
        penalty=penalty,
        penalty_strength=0.03,
    )


def small_config(loss: str, penalty: str) -> dict[str, Any]:
    """The shipped synthetic FedAvg arm, on the small corpus, for one round."""

    manifest, table = small_corpus()
    config = standalone_config_mapping(ARM)
    config["experiment"]["extensions"] = [str(EXTENSION)]
    config["data"]["path"] = str(manifest)
    config["model"].update(
        input_dim=8, loss=loss, penalty=penalty, penalty_strength=0.03, optima=str(table)
    )
    config["client"].update(learning_rate=0.5, batch_size=16)
    config["schedule"]["rounds"] = 1
    config["evaluation"]["test"] = {"every": 1, "clients": "all"}
    config["runtime"]["quiet"] = True
    config["runtime"]["checkpointing"].update(
        enabled=True, save_last=True, save_every_round=True, keep_last=None
    )
    return config


def _task(manifest: Path, model: dict[str, Any]) -> Any:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["manifest_path"] = str(manifest)
    return problem.FedLogisticL1Task(model_config=model, dataset_metadata=metadata)


def _small_task(loss: str, penalty: str, **model: Any) -> Any:
    manifest, table = small_corpus()
    values = {
        "input_dim": 8,
        "penalty_strength": 0.03,
        "loss": loss,
        "penalty": penalty,
        "optima": str(table),
        **model,
    }
    return _task(manifest, values)


class TheGradientIsAnalyticTest(unittest.TestCase):
    def test_autograd_of_the_task_loss_is_the_gradient(self) -> None:
        spec = _small_spec("logistic", "l1")
        features, labels = spec.design(), spec.labels()
        generator = torch.Generator().manual_seed(7)
        for loss, penalty in problem.PROBLEMS:
            model = problem.LogisticModel(8, 0.03, loss=loss, penalty=penalty)
            task = _small_task(loss, penalty)
            for _ in range(5):
                point = torch.randn(8, generator=generator, dtype=torch.float64)
                with self.subTest(problem=f"{loss}+{penalty}"):
                    params = {"x": point.clone().requires_grad_(True)}
                    value, _ = task.functional_loss(model, params, {}, (features, labels))
                    (measured,) = torch.autograd.grad(value, params["x"])
                    expected = problem.gradient(point, features, labels, 0.03, loss, penalty)
                    scale = float(expected.abs().max())
                    self.assertLessEqual(float((measured - expected).abs().max()), 1e-14 * scale)
                    self.assertEqual(
                        float(value),
                        problem.objective(point, features, labels, 0.03, loss, penalty),
                    )


class TheLookupTest(unittest.TestCase):
    def test_a_convex_problem_finds_its_certified_optimum_and_a_nonconvex_one_none(self) -> None:
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                task = _small_task(loss, penalty)
                if not problem.convex(loss, penalty):
                    self.assertIsNone(task._optimal_objective)
                    self.assertNotIn("optimality_gap", task._central_names)
                    continue
                entry = task.optimum_entry
                self.assertLessEqual(entry["kkt_residual"], problem.CERTIFICATE)
                self.assertEqual(entry["digest"], task.reference["corpus_digest"])
                at = task.pooled_objective(problem._x_star_of(entry, 8))
                self.assertEqual(task._optimal_objective, at)

    def test_a_convex_problem_without_an_entry_is_refused(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "neither the optima table nor the machine-local table"
        ):
            _small_task("logistic", "l1", penalty_strength=0.05)

    def test_an_entry_at_another_digest_is_refused(self) -> None:
        _, table = small_corpus()
        entries = problem.read_optima(table)
        edited = Path(_root.name) / "edited.json"
        for entry in entries:
            problem.write_optimum(edited, {**entry, "digest": "0" * 64})
        with self.assertRaisesRegex(ValueError, "not this data's optimum"):
            _small_task("logistic", "l1", optima=str(edited))

    def test_shards_that_are_not_their_recorded_digest_are_refused(self) -> None:
        manifest, _ = small_corpus()
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)
            for path in manifest.parent.rglob("*"):
                target = copy / path.relative_to(manifest.parent)
                if path.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            shard = copy / "shards" / "global_test.pt"
            rows = load_client_shard(shard)
            features = rows["x"].clone()
            features[0, 0] = torch.nextafter(
                features[0, 0], torch.tensor(1e9, dtype=features.dtype)
            )
            save_client_shard(shard, features, rows["y"])
            with self.assertRaisesRegex(ValueError, "not the corpus it describes"):
                _task(copy / "manifest.json", {"input_dim": 8, "loss": "tanh", "penalty": "l2sq"})


class TheShippedTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = problem.read_optima(problem.OPTIMA_TABLE)

    def _convex_arms(self) -> list[tuple[str, Path, dict[str, Any]]]:
        arms = []
        for corpus_config in sorted(GENERATOR_CONFIGS.glob("fed-logistic-l1-*.yaml")):
            corpus = corpus_config.stem
            for arm in sorted((REPO_ROOT / "configs" / "examples" / corpus).glob("*.yaml")):
                model = load_config_mapping(arm)["model"]
                if problem.convex(model["loss"], model["penalty"]):
                    arms.append((corpus, arm, model))
        return arms

    def test_every_convex_arm_has_its_previous_f_star(self) -> None:
        by_key = {
            (entry["corpus"], entry["loss"], entry["penalty"], float(entry["lam"])): entry
            for entry in self.entries
        }
        arms = self._convex_arms()
        self.assertEqual(len(arms), len(PREVIOUS_F_STAR))
        for corpus, arm, model in arms:
            key = (corpus, model["loss"], model["penalty"], float(model["penalty_strength"]))
            with self.subTest(arm=f"{corpus}/{arm.stem}"):
                self.assertIn(key, by_key)
                self.assertEqual(by_key[key]["f_star"], PREVIOUS_F_STAR[key])
                self.assertLessEqual(by_key[key]["kkt_residual"], problem.CERTIFICATE)

    def test_a_corpus_has_one_digest(self) -> None:
        digests: dict[str, set[str]] = {}
        for entry in self.entries:
            digests.setdefault(entry["corpus"], set()).add(entry["digest"])
        for corpus, found in digests.items():
            with self.subTest(corpus=corpus):
                self.assertEqual(len(found), 1)

    def test_the_cheap_corpora_find_their_f_star_here(self) -> None:
        """Generated here, each convex arm's task finds its F*: shipped, or certified on demand.

        Where this build makes the shipped rows (the corpus's digest is the shipped
        entry's), F* is the shipped value, and nothing was certified beside the data.
        On any other build -- another torch's vectorized kernels, another LAPACK --
        the rows differ in their last bits, ``fedbrew generate`` certified the same
        problem on them, and the task's F* is that local entry's, certified on this
        machine to the same KKT bound, within 1e-9 of the shipped value.
        """

        for corpus in CHEAP_CORPORA:
            config = yaml.safe_load((GENERATOR_CONFIGS / f"{corpus}.yaml").read_text())
            manifest = _generate(config, Path(_root.name) / corpus)
            digest = yaml.safe_load(manifest.read_text())["reference"]["corpus_digest"]
            local = {
                problem._key_of(entry): entry
                for entry in problem.read_optima(manifest.parent / problem.LOCAL_OPTIMA)
            }
            for name, arm, model in self._convex_arms():
                if name != corpus:
                    continue
                key = (corpus, model["loss"], model["penalty"], float(model["penalty_strength"]))
                wanted = (digest, *key[1:])
                with self.subTest(arm=f"{corpus}/{arm.stem}"):
                    task = _task(manifest, model)
                    shipped = [e for e in self.entries if problem._key_of(e) == wanted]
                    if shipped:
                        self.assertEqual(task._optimal_objective, PREVIOUS_F_STAR[key])
                        self.assertNotIn(wanted, local)
                        continue
                    entry = local[wanted]
                    self.assertEqual(entry["certified_on"], problem.CERTIFIED_HERE)
                    self.assertEqual(entry["build"]["torch"], torch.__version__)
                    self.assertLessEqual(entry["kkt_residual"], problem.CERTIFICATE)
                    self.assertEqual(task._optimal_objective, entry["f_star"])
                    self.assertLess(
                        abs(entry["f_star"] - PREVIOUS_F_STAR[key]) / PREVIOUS_F_STAR[key], 1e-9
                    )


class CertifiedOnDemandTest(unittest.TestCase):
    """A corpus the shipped table was not certified on is certified beside its data, on demand.

    Driven on the small corpus with a stand-in shipped table, so the path runs on
    every build, this one included: ``shipped(digest)`` is a table certifying the
    small corpus's convex problems at ``digest``.
    """

    def setUp(self) -> None:
        manifest, _ = small_corpus()
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for item in manifest.parent.iterdir():
            target = self.data / item.name
            if item.is_dir():
                target.symlink_to(item, target_is_directory=True)
            elif item.name != problem.LOCAL_OPTIMA:
                target.write_bytes(item.read_bytes())
        self.manifest = self.data / manifest.name
        self.digest = yaml.safe_load(self.manifest.read_text())["reference"]["corpus_digest"]
        self.spec = _small_spec("logistic", "l1")

    def shipped(self, digest: str, f_star: float | None = None) -> Path:
        path = self.root / f"shipped-{digest[:6]}.json"
        for loss, penalty in problem.PROBLEMS:
            if problem.convex(loss, penalty):
                entry = problem.certify(_small_spec(loss, penalty))
                entry["digest"] = digest
                if f_star is not None:
                    entry["f_star"] = f_star
                problem.write_optimum(path, entry)
        return path

    def model(self, table: Path, loss: str = "logistic", penalty: str = "l1") -> dict[str, Any]:
        return {
            "input_dim": 8,
            "penalty_strength": 0.03,
            "loss": loss,
            "penalty": penalty,
            "optima": str(table),
        }

    def test_another_builds_rows_are_certified_beside_the_data(self) -> None:
        table = self.shipped("0" * 64)
        before = table.read_bytes()
        made = problem.certify_on_demand(self.spec, self.data, self.digest, table=table)
        convex = [
            (loss, penalty) for loss, penalty in problem.PROBLEMS if problem.convex(loss, penalty)
        ]
        self.assertEqual(sorted((e["loss"], e["penalty"]) for e in made), sorted(convex))
        local = problem.read_optima(self.data / problem.LOCAL_OPTIMA)
        self.assertEqual([problem._key_of(e) for e in local], [problem._key_of(e) for e in made])
        for entry in local:
            with self.subTest(problem=(entry["loss"], entry["penalty"])):
                self.assertEqual(entry["digest"], self.digest)
                self.assertEqual(entry["certified_on"], problem.CERTIFIED_HERE)
                self.assertEqual(
                    set(entry["build"]),
                    {"torch", "blas", "lapack", "cpu_capability", "cpu", "qr_threads"},
                )
                self.assertLessEqual(entry["kkt_residual"], problem.CERTIFICATE)
                reference = problem.certify(_small_spec(entry["loss"], entry["penalty"]))
                self.assertEqual(entry["f_star"], reference["f_star"])
        # The shipped table is not changed, and a second generation certifies nothing more.
        self.assertEqual(table.read_bytes(), before)
        self.assertEqual(
            problem.certify_on_demand(self.spec, self.data, self.digest, table=table), []
        )

    def test_nothing_is_certified_where_the_shipped_table_holds_the_digest(self) -> None:
        table = self.shipped(self.digest)
        self.assertEqual(
            problem.certify_on_demand(self.spec, self.data, self.digest, table=table), []
        )
        self.assertFalse((self.data / problem.LOCAL_OPTIMA).exists())

    def test_nor_for_a_corpus_the_shipped_table_does_not_name(self) -> None:
        other = dataclasses.replace(self.spec, corpus="another")
        table = self.shipped("0" * 64)
        self.assertEqual(problem.certify_on_demand(other, self.data, self.digest, table=table), [])

    def test_the_task_reads_the_shipped_table_then_the_local_one(self) -> None:
        elsewhere = self.shipped("0" * 64)
        with self.assertRaisesRegex(ValueError, problem.LOCAL_OPTIMA):
            _task(self.manifest, self.model(elsewhere))
        problem.certify_on_demand(self.spec, self.data, self.digest, table=elsewhere)
        local = {
            problem._key_of(e): e for e in problem.read_optima(self.data / problem.LOCAL_OPTIMA)
        }
        task = _task(self.manifest, self.model(elsewhere))
        key = (self.digest, "logistic", "l1", 0.03)
        self.assertEqual(task._optimal_objective, local[key]["f_star"])
        self.assertEqual(task.optimum_entry["certified_on"], problem.CERTIFIED_HERE)
        # Where both hold the digest, the shipped entry is the one read.
        here = self.shipped(self.digest, f_star=0.125)
        self.assertEqual(_task(self.manifest, self.model(here))._optimal_objective, 0.125)

    def test_the_planned_columns_read_the_local_table_too(self) -> None:
        from types import SimpleNamespace

        elsewhere = self.shipped("0" * 64)
        config = SimpleNamespace(
            data=SimpleNamespace(path=str(self.manifest)),
            model=SimpleNamespace(
                input_dim=8,
                extra={k: v for k, v in self.model(elsewhere).items() if k != "input_dim"},
            ),
        )
        self.assertIsNone(problem.reported_metrics(config))
        problem.certify_on_demand(self.spec, self.data, self.digest, table=elsewhere)
        reported = problem.reported_metrics(config)
        assert reported is not None
        self.assertIn("optimality_gap", reported.central)


class TheConditionNumberDialTest(unittest.TestCase):
    def test_the_pooled_gram_has_the_condition_number_asked_for(self) -> None:
        for kappa in (1.0, 10.0, 100.0):
            with self.subTest(kappa=kappa):
                spec = problem.ProblemSpec(
                    num_clients=4,
                    dim=8,
                    rows_per_client=16,
                    partition_block=4,
                    condition_number=kappa,
                )
                record = problem._gram_record(spec.design())
                self.assertAlmostEqual(record["gram_condition"], kappa, delta=1e-10 * kappa)
                self.assertAlmostEqual(record["gram_lambda_max"], 1.0, delta=1e-12)
                self.assertAlmostEqual(spec.lipschitz(), 0.25, delta=1e-12)

    def test_the_same_rows_on_one_thread_and_on_four(self) -> None:
        design = problem.halton_normal_design(512, 16)
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        built = []
        for count in (1, 4):
            torch.set_num_threads(count)
            built.append(problem.conditioned(design, 10.0))
            self.assertEqual(torch.get_num_threads(), count)
        self.assertTrue(torch.equal(*built))


class ALibsvmSourceTest(unittest.TestCase):
    """The reader, the digest pin and the deal, on a 16-row file written here."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        spec = problem.ProblemSpec(num_clients=4, dim=4, rows_per_client=4, partition_block=2)
        self.features, self.labels = spec.design(), spec.labels()
        lines = []
        for row, label in zip(self.features.tolist(), self.labels.tolist(), strict=True):
            cells = " ".join(f"{column + 1}:{value!r}" for column, value in enumerate(row))
            lines.append(f"{int(label):+d} {cells}")
        self.path = self.root / "rows.libsvm"
        self.path.write_text("\n".join(lines) + "\n", encoding="ascii")
        self.source = problem.SourceSpec(
            path=str(self.path), sha256=problem.digest_of(self.path), row_normalize=False
        )

    def _spec(self) -> Any:
        return problem.ProblemSpec(
            num_clients=4,
            dim=4,
            rows_per_client=4,
            partition_block=2,
            penalty_strength=0.01,
            source=self.source,
            partition_reference_lambda=0.01,
            partition_key="exact",
            corpus="rows",
        )

    def test_the_rows_are_the_file_and_the_deal_a_partition(self) -> None:
        spec = self._spec()
        self.assertTrue(torch.equal(spec.design(), self.features))
        self.assertTrue(torch.equal(spec.labels(), self.labels))
        dealt = torch.cat(spec.client_indices())
        self.assertEqual(sorted(dealt.tolist()), list(range(16)))
        reference = problem.corpus_reference(spec)
        self.assertNotIn("x_true", reference)
        self.assertEqual(reference["problem"]["source"]["sha256"], self.source.sha256)
        for penalty in ("l1", "l2sq"):
            entry = problem.certify(dataclasses.replace(spec, penalty=penalty))
            self.assertEqual(entry["digest"], reference["corpus_digest"])
            self.assertLessEqual(entry["kkt_residual"], problem.CERTIFICATE)

    def test_the_exact_key_does_not_depend_on_the_summation_order(self) -> None:
        weights = torch.tensor([0.3, -1.25, 2.0e-3, 7.5], dtype=torch.float64)
        order = torch.tensor([3, 1, 0, 2])
        self.assertTrue(
            torch.equal(
                problem.exact_margins(self.features, weights),
                problem.exact_margins(self.features[:, order], weights[order]),
            )
        )

    def test_a_file_with_another_digest_or_none_is_refused(self) -> None:
        other = problem.SourceSpec(path=str(self.path), sha256="0" * 64, row_normalize=False)
        with self.assertRaisesRegex(ValueError, "the config pins"):
            problem.load_source(other, 4, 16)
        missing = problem.SourceSpec(
            path=str(self.root / "absent.bz2"), sha256="0" * 64, url="https://example.org/f"
        )
        with self.assertRaisesRegex(FileNotFoundError, "curl -L -o"):
            problem.load_source(missing, 4, 16)


class OneRoundBatchedAgreesTest(ExecutorRuns):
    """One round of each problem, with its gradient norm: batched against sequential.

    Also the plan header's columns against the CSV's, per problem -- the task
    reports the gap and the distance to x* only where F* is certified -- and
    grad_norm_sq against the analytic gradient of F at the round's model, the
    minimum-norm subgradient under the l1 regularizer.
    """

    def test_each_problem_within_the_executor_tolerance(self) -> None:
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                config = small_config(loss, penalty)
                config["evaluation"]["grad_norm"] = {"every": 1}
                batched, sequential = self.both(config)
                record = json.loads((batched / "run.json").read_text())["reproducibility"]
                self.assertEqual(record["executor"]["used"], "batched")
                planned = set(
                    _planned_metric_names(load_config(self.root / f"run{self._count}.yaml"))
                )
                with (sequential / "round_metrics.csv").open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(planned, set(rows[0]) - BOOKKEEPING)
                self.assertEqual(
                    "central_test_optimality_gap" in planned, problem.convex(loss, penalty)
                )
                self._check_grad_norm(sequential, float(rows[-1]["grad_norm_sq"]), loss, penalty)
                for name in CSVS:
                    self._compare_csv(batched / name, sequential / name, False, TOLERANCE)
                self._compare_checkpoint(
                    batched / "checkpoints" / "round_001.pt",
                    sequential / "checkpoints" / "round_001.pt",
                    False,
                    TOLERANCE,
                )

    def _check_grad_norm(self, run: Path, measured: float, loss: str, penalty: str) -> None:
        spec = _small_spec(loss, penalty)
        state = torch.load(run / "checkpoints" / "round_001.pt", weights_only=False)
        x = state["model_state"]["x"].to(torch.float64)
        stacked = torch.cat(spec.client_indices())
        features, labels = spec.design()[stacked], spec.labels()[stacked]
        smooth = problem.smooth_gradient(x, features, labels, loss)
        if penalty == "l1":
            shrunk = smooth.abs().sub(0.03).clamp_min(0.0) * smooth.sign()
            expected = torch.where(x == 0.0, shrunk, smooth + 0.03 * x.sign())
        else:
            expected = problem.gradient(x, features, labels, 0.03, loss, penalty)
        squared = float(expected.square().sum())
        self.assertLessEqual(abs(measured - squared), 1e-12 * squared)


class TheCentralPassIsMeasuredWhereTheRoundTrainsTest(ExecutorRuns):
    """The task's central pass in parts: a resident run measures it on its device.

    ``CentralPassInParts``: the eval steps over every row, the pooled
    objective's terms where F* is certified, and the columns made of both.
    Every column a resident run writes so is the one the host's
    ``evaluate_model`` writes at the flush, bit for bit, for every problem.
    """

    def test_each_problem(self) -> None:
        from fedbrew.core import resident_evaluation
        from fedbrew.tasks.base import CentralPassInParts

        real = resident_evaluation._central_rows
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                self.assertIsInstance(_small_task(loss, penalty), CentralPassInParts)
                config = small_config(loss, penalty)
                config["schedule"]["rounds"] = 4
                config["evaluation"]["central_test"] = {"every": 1}
                taken: list[Any] = []

                def recorded(rounds: Any, _taken: list[Any] = taken) -> Any:
                    _taken.append(real(rounds))
                    return _taken[-1]

                with mock.patch.object(resident_evaluation, "_central_rows", recorded):
                    device = self.run_config(config, "batched")
                with mock.patch.object(resident_evaluation, "_central_rows", lambda rounds: None):
                    host = self.run_config(config, "batched")
                self.assertTrue(taken and taken[0] is not None and taken[0].in_parts)
                for output in (device, host):
                    record = json.loads((output / "run.json").read_text())["reproducibility"]
                    self.assertEqual(record["executor"]["rounds"], {"used": "resident"})
                for name in CSVS:
                    self.assertEqual(_untimed(device / name), _untimed(host / name), name)
                with (device / "round_metrics.csv").open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(
                    "central_test_optimality_gap" in rows[0], problem.convex(loss, penalty)
                )
                self.assertTrue(all(row["central_test_loss"] for row in rows))


class TheStackedEvalIsVmapsTest(unittest.TestCase):
    """``stacked_eval`` is ``torch.func.vmap(functional_eval)``, bit for bit, key for key.

    For every problem, with and without padding, at iterates with zero, tiny and
    large coordinates; and its tables name every penalty and loss.
    """

    def test_every_problem(self) -> None:
        self.assertEqual(set(problem.STACKED_PENALTIES), set(problem.PENALTIES))
        self.assertEqual(set(problem.STACKED_LOSSES), set(problem.LOSSES))
        generator = torch.Generator().manual_seed(11)
        clients, rows, dim = 6, 16, 8
        for loss, penalty in problem.PROBLEMS:
            task = _small_task(loss, penalty)
            model = problem.LogisticModel(dim, 0.03, loss=loss, penalty=penalty)
            for masked, scale in ((False, 1.0), (True, 1.0), (False, 1e-4), (True, 30.0)):
                with self.subTest(problem=f"{loss}+{penalty}", masked=masked, scale=scale):
                    features = torch.randn(
                        clients, rows, dim, generator=generator, dtype=torch.float64
                    )
                    signs = torch.randint(0, 2, (clients, rows), generator=generator)
                    labels = signs.to(torch.float64) * 2.0 - 1.0
                    x = torch.randn(clients, dim, generator=generator, dtype=torch.float64) * scale
                    x[:, :2] = 0.0
                    mask = None
                    if masked:
                        lengths = torch.randint(1, rows + 1, (clients,), generator=generator)
                        mask = (torch.arange(rows) < lengths.unsqueeze(1)).to(torch.float64)
                    params, batch = {"x": x}, (features, labels)

                    def measure(
                        params: Any, batch: Any, mask: Any, _task: Any = task, _model: Any = model
                    ) -> Any:
                        return _task.functional_eval(_model, params, {}, batch, mask)

                    dims = (0, 0, None if mask is None else 0)
                    with torch.no_grad():
                        expected = torch.func.vmap(measure, in_dims=dims)(params, batch, mask)
                        stacked = task.stacked_eval(model, params, {}, batch, mask)
                    self.assertEqual(list(stacked), list(expected))
                    for key, value in expected.items():
                        self.assertTrue(torch.equal(stacked[key], value), key)


class ThePostFitPassTakesTheStackedEvalTest(ExecutorRuns):
    """A resident run's post-fit pass calls no vmap, and writes what the vmapped pass writes."""

    def test_each_problem(self) -> None:
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                config = small_config(loss, penalty)
                config["schedule"]["rounds"] = 3
                # The post-fit pass alone: a client split's, at the shared model, is vmapped.
                config["evaluation"]["fit"] = {"every": 1}
                config["evaluation"]["test"] = {"every": "never"}
                refuse = mock.Mock(side_effect=AssertionError("a vmapped post-fit pass"))
                # The closed form trains, as the task's runs do by default: no vmap there.
                with mock.patch("torch.func.vmap", refuse):
                    stacked = self.run_config(config, "batched", gradient_form="closed_form")
                counted = mock.Mock(wraps=torch.func.vmap)
                with (
                    mock.patch.object(_run_task_class(), "stacked_eval", None),
                    mock.patch("torch.func.vmap", counted),
                ):
                    vmapped = self.run_config(config, "batched", gradient_form="closed_form")
                refuse.assert_not_called()
                self.assertGreater(counted.call_count, 0)
                record = json.loads((stacked / "run.json").read_text())["reproducibility"]
                self.assertEqual(record["executor"]["rounds"], {"used": "resident"})
                for name in CSVS:
                    self.assertEqual(_untimed(stacked / name), _untimed(vmapped / name), name)
                with (stacked / "round_metrics.csv").open(encoding="utf-8") as handle:
                    self.assertTrue(all(row["fit_loss"] for row in csv.DictReader(handle)))


class TheStackedMetricsAreComputeMetricsTest(unittest.TestCase):
    """``stacked_metrics`` folds each split as ``compute_metrics`` does, bit for bit, on the device.

    Splits of no, one and several positions; a split whose totals are all 0
    (the plain mean); NaN and huge values in padding positions, which must not
    reach a split; and values whose sum rounds differently in another order.
    """

    def test_ragged_splits(self) -> None:
        generator = torch.Generator().manual_seed(3)
        for loss, penalty in problem.PROBLEMS:
            task = _small_task(loss, penalty)
            names = list(task._names)
            counts = [0, 1, 4, 2, 3, 4]
            positions = max(counts)
            outputs = []
            for position in range(positions):
                output = {
                    name: torch.randn(len(counts), generator=generator, dtype=torch.float64)
                    * 10.0 ** float(position)
                    for name in names
                }
                output["total"] = torch.tensor([7.0, 3.0, 5.0, 0.0, 1.0, 2.0], dtype=torch.float64)
                for split, count in enumerate(counts):
                    if position >= count:
                        for name in names:
                            output[name][split] = float("nan")
                        output["total"][split] = 1e300
                outputs.append(output)
            with self.subTest(problem=f"{loss}+{penalty}"):
                metrics, examples = task.stacked_metrics(outputs, counts)
                self.assertEqual(list(metrics), names)
                for split, count in enumerate(counts):
                    records = [
                        {key: float(value[split]) for key, value in outputs[p].items()}
                        for p in range(count)
                    ]
                    expected = task.compute_metrics(records)
                    for name in names:
                        self.assertEqual(float(metrics[name][split]), expected[name], (split, name))
                    total = sum(float(outputs[p]["total"][split]) for p in range(count))
                    self.assertEqual(float(examples[split]), total)

    def test_the_host_sums_left_to_right(self) -> None:
        values = [1e16, 1.0, -1e16, 3.0, 0.1, 0.2]
        expected = 0
        for value in values:
            expected = expected + value
        self.assertEqual(problem._left_to_right(values), expected)


class ThePostFitMetricsAreFoldedTest(ExecutorRuns):
    """A resident run folds its post-fit metrics on the device; the rows are the per-client path's.

    Four post-fit batches a client (16 rows, eval batches of 4), so the sums are
    taken over several positions.
    """

    def test_each_problem(self) -> None:
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                config = small_config(loss, penalty)
                config["schedule"]["rounds"] = 3
                config["client"]["eval_batch_size"] = 4
                config["evaluation"]["fit"] = {"every": 1}
                folds = mock.Mock(wraps=_run_task_class().stacked_metrics)

                def folded(task: Any, *args: Any, _folds: Any = folds) -> Any:
                    return _folds(task, *args)

                task_class = _run_task_class()
                with mock.patch.object(task_class, "stacked_metrics", folded):
                    stacked = self.run_config(config, "batched", gradient_form="closed_form")
                with mock.patch.object(task_class, "stacked_metrics", None):
                    per_client = self.run_config(config, "batched", gradient_form="closed_form")
                self.assertGreater(folds.call_count, 0)
                self.assertTrue(any(max(call.args[2]) == 4 for call in folds.call_args_list))
                for name in CSVS:
                    self.assertEqual(_untimed(stacked / name), _untimed(per_client / name), name)


class FAndItsGradientInOnePassTest(ExecutorRuns):
    """``evaluation.grad_norm.fused``, on by default: F's central pass and its gradient in one.

    For every problem, resident, batched per round and sequential: the fused
    run through autograd (``gradient_form: autograd``) has the two passes'
    central columns bit for bit and their grad_norm_sq to rounding; the
    default, the task's closed form (``closed_form_eval``), has those of
    autograd's fused run bit for bit but F's and grad_norm_sq's, which are the
    same to rounding -- within 1e-12 relative, or grad_norm_sq within the
    rounding of the rows' terms (POST-F38, ``within_the_rule``); and run.json
    says which pass and form each run took.
    With the central pass every other round, the rounds between measure
    grad_norm_sq alone, in its own pass, and the run is the same again.
    """

    def test_each_problem_and_path(self) -> None:
        from fedbrew.core.config import evaluates_round
        from fedbrew.core.resident_evaluation import ResidentEvaluation
        from tests.test_resident_round import per_round

        settings = {
            "closed": {"fused": True},
            "autograd": {"fused": True, "gradient_form": "autograd"},
            "separate": {"fused": False},
        }
        for loss, penalty in problem.PROBLEMS:
            for path in ("resident", "per round", "sequential"):
                for central in (1, 2):
                    with self.subTest(problem=f"{loss}+{penalty}", path=path, central=central):
                        config = small_config(loss, penalty)
                        config["schedule"]["rounds"] = 4
                        config["evaluation"]["central_test"] = {"every": central}
                        runs, tolerances = {}, {}
                        for name, setting in settings.items():
                            config["evaluation"]["grad_norm"] = {"every": 1, **setting}
                            spy = mock.patch.object(
                                ResidentEvaluation,
                                "enqueue_fused",
                                autospec=True,
                                side_effect=ResidentEvaluation.enqueue_fused,
                            )
                            executor = "sequential" if path == "sequential" else "batched"
                            context = per_round() if path == "per round" else nullcontext()
                            with (
                                context,
                                spy as fused_rounds,
                                recorded_tolerances() as tolerances[name],
                            ):
                                runs[name] = self.run_config(config, executor)
                            # The rounds the central pass measures: every
                            # other one, pinned at the first and the last.
                            both = sum(evaluates_round(central, r, 4) for r in range(1, 5))
                            self.assertEqual(
                                fused_rounds.call_count,
                                both if name != "separate" and path == "resident" else 0,
                            )
                        self._records(runs)
                        self._same_but(
                            runs["autograd"],
                            runs["separate"],
                            ("grad_norm_sq",),
                            tolerances["autograd"],
                        )
                        self._same_but(
                            runs["closed"],
                            runs["autograd"],
                            ("grad_norm_sq", "central_test_loss"),
                            tolerances["closed"],
                        )

    def _records(self, runs: dict[str, Path]) -> None:
        records = {
            name: json.loads((path / "run.json").read_text())["reproducibility"]["grad_norm"]
            for name, path in runs.items()
        }
        self.assertEqual(
            records["closed"], {"pass": "fused", "asked": "fused", "gradient": "closed_form"}
        )
        self.assertEqual(
            records["autograd"],
            {
                "pass": "fused",
                "asked": "fused",
                "gradient": "autograd",
                "gradient_reason": "asked for",
            },
        )
        apart = {"pass": "separate", "asked": "separate", "reason": "asked for"}
        self.assertEqual(records["separate"], apart)

    def _same_but(
        self, run: Path, other: Path, within: tuple[str, ...], tolerances: dict[float, Any]
    ) -> None:
        """``run``'s CSVs are ``other``'s, but the columns ``within`` names, to rounding."""

        rows = [_untimed(path / "round_metrics.csv") for path in (run, other)]
        assert rows[0] is not None and rows[1] is not None
        self.assertEqual(len(rows[0]), len(rows[1]))
        for row, row_other in zip(rows[0], rows[1], strict=True):
            self.assertEqual(list(row), list(row_other))
            for key, value in row_other.items():
                if not key.startswith(within) or row[key] == value:
                    self.assertEqual(row[key], value, key)
                    continue
                a, b = float(row[key]), float(value)
                self.assertTrue(within_the_rule(key, a, b, tolerances), f"{key}: {a!r} {b!r}")
        for name in CSVS:
            if name != "round_metrics.csv":
                self.assertEqual(_untimed(run / name), _untimed(other / name), name)


class TheFusedPassReadsTheTrainingRowsTest(ExecutorRuns):
    """A resident run's closed-form fused pass reads the stack the round trains on in closed form.

    The prepared rows are held once (``FusedPass.share_rows``), and the run
    writes what it writes with the pass's own copy, for every problem.
    """

    def test_each_problem(self) -> None:
        from fedbrew.core.grad_norm import FusedPass
        from fedbrew.core.resident import ResidentRounds

        real = ResidentRounds._share_rows
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                config = small_config(loss, penalty)
                config["schedule"]["rounds"] = 3
                config["evaluation"]["central_test"] = {"every": 1}
                config["evaluation"]["grad_norm"] = {"every": 1}
                runs, shared = {}, {}
                for sharing in (True, False):
                    seen: list[bool] = []

                    def recording(rounds: Any, fused: Any, _seen: list[bool] = seen) -> bool:
                        _seen.append(real(rounds, fused))
                        return _seen[-1]

                    declined = mock.patch.object(FusedPass, "share_rows", return_value=False)
                    with (
                        mock.patch.object(ResidentRounds, "_share_rows", recording),
                        nullcontext() if sharing else declined,
                    ):
                        runs[sharing] = self.run_config(
                            config, "batched", gradient_form="closed_form"
                        )
                    shared[sharing] = seen
                self.assertEqual(shared, {True: [True], False: [False]})
                self.assertEqual(
                    _untimed(runs[True] / "round_metrics.csv"),
                    _untimed(runs[False] / "round_metrics.csv"),
                )


class TheCentralPassReadsTheTrainingRowsTest(ExecutorRuns):
    """A resident run's central pass reads the stack the round trains on, and writes the same.

    In closed form the task's ``closed_form_central`` measures the steps and
    the gap's terms from the prepared stack (``ResidentRounds._share_central``);
    through autograd the pass reads the stacked rows themselves. Either way,
    for every problem, with the central pass every round and no
    ``grad_norm_sq``, the run writes what it writes reading the pass's own rows.
    """

    def test_each_problem_and_form(self) -> None:
        from fedbrew.core.resident import ResidentRounds

        real = ResidentRounds._share_central
        for loss, penalty in problem.PROBLEMS:
            for form in ("closed_form", "autograd"):
                with self.subTest(problem=f"{loss}+{penalty}", form=form):
                    config = small_config(loss, penalty)
                    config["schedule"]["rounds"] = 3
                    config["evaluation"]["central_test"] = {"every": 1}
                    runs, shared = {}, {}
                    for sharing in (True, False):
                        seen: list[bool] = []

                        def recording(rounds: Any, _seen: list[bool] = seen) -> bool:
                            _seen.append(real(rounds))
                            return _seen[-1]

                        declined = mock.patch.object(
                            ResidentRounds, "_share_central", return_value=False
                        )
                        recorded = mock.patch.object(ResidentRounds, "_share_central", recording)
                        with recorded if sharing else declined:
                            runs[sharing] = self.run_config(config, "batched", gradient_form=form)
                        shared[sharing] = seen
                    self.assertEqual(shared[True], [True])
                    for name in CSVS:
                        self.assertEqual(
                            _untimed(runs[True] / name), _untimed(runs[False] / name), name
                        )


class TheCentralPassInClosedFormTest(unittest.TestCase):
    """``closed_form_central``: ``functional_eval``'s outputs and ``central_terms``' bit for bit.

    For every problem, at iterates with exact zeros, over the task's pooled
    rows prepared as ``closed_form_batch`` prepares them: one step's outputs,
    the step over every row, and the gap's two terms where the optimum is
    certified (none elsewhere), each the same tensor.
    """

    def test_every_problem(self) -> None:
        from fedbrew.tasks.base import closed_form_batch

        generator = torch.Generator().manual_seed(19)
        with_terms = 0
        for loss, penalty in problem.PROBLEMS:
            task = _small_task(loss, penalty)
            model = problem.LogisticModel(8, 0.03, loss=loss, penalty=penalty)
            rows = (task._pooled_features, task._pooled_labels)
            batch = closed_form_batch(task, model, rows)
            for _ in range(3):
                with self.subTest(problem=f"{loss}+{penalty}"):
                    x = torch.randn(8, generator=generator, dtype=torch.float64)
                    x[:2] = 0.0
                    with torch.no_grad():
                        outputs, terms = task.closed_form_central(model, {"x": x}, {}, batch)
                        expected = task.functional_eval(model, {"x": x}, {}, rows, None)
                        expected_terms = task.central_terms(model, {"x": x})
                    self.assertEqual(len(outputs), 1)
                    self.assertEqual(list(outputs[0]), list(expected))
                    for key, value in expected.items():
                        self.assertTrue(torch.equal(outputs[0][key], value), key)
                    with_terms += bool(expected_terms)
                    self.assertEqual(list(terms), list(expected_terms))
                    for key, value in expected_terms.items():
                        self.assertTrue(torch.equal(terms[key], value), key)
        # The convex problems' optima are certified: their gap's terms are held too.
        self.assertGreater(with_terms, 0)


class FAndItsGradientInClosedFormTest(unittest.TestCase):
    """``closed_form_eval``: ``functional_eval``'s outputs and autograd's gradient, in closed form.

    For every problem, at iterates with exact zeros (which the l1 term's
    subgradient reads), with and without padding: each client's loss and
    gradient within 1e-12 relative of autograd's through ``functional_eval``,
    and every other output ``stacked_eval``'s bit for bit.
    """

    def test_every_problem(self) -> None:
        generator = torch.Generator().manual_seed(13)
        clients, rows, dim = 4, 16, 8
        for loss, penalty in problem.PROBLEMS:
            task = _small_task(loss, penalty)
            model = problem.LogisticModel(dim, 0.03, loss=loss, penalty=penalty)
            for masked in (False, True):
                with self.subTest(problem=f"{loss}+{penalty}", masked=masked):
                    features = torch.randn(
                        clients, rows, dim, generator=generator, dtype=torch.float64
                    )
                    signs = torch.randint(0, 2, (clients, rows), generator=generator)
                    labels = signs.to(torch.float64) * 2.0 - 1.0
                    x = torch.randn(clients, dim, generator=generator, dtype=torch.float64)
                    x[:, :2] = 0.0
                    mask = None
                    if masked:
                        lengths = torch.randint(1, rows + 1, (clients,), generator=generator)
                        mask = (torch.arange(rows) < lengths.unsqueeze(1)).to(torch.float64)
                    signed = task.closed_form_rows(model, (features, labels))
                    with torch.no_grad():
                        grads, outputs = task.closed_form_eval(model, {"x": x}, {}, signed, mask)
                        stacked = task.stacked_eval(model, {"x": x}, {}, (features, labels), mask)
                    self.assertEqual(list(outputs), list(stacked))
                    for key, value in stacked.items():
                        if key != "loss":
                            self.assertTrue(torch.equal(outputs[key], value), key)
                    for client in range(clients):
                        keep = None if mask is None else mask[client]
                        leaf = x[client].clone().requires_grad_(True)
                        measured = task.functional_eval(
                            model, {"x": leaf}, {}, (features[client], labels[client]), keep
                        )
                        (gradient,) = torch.autograd.grad(measured["loss"], [leaf])
                        ours, theirs = float(outputs["loss"][client]), float(measured["loss"])
                        self.assertLessEqual(abs(ours - theirs), 1e-12 * abs(theirs))
                        self.assertTrue(
                            torch.allclose(grads["x"][client], gradient, rtol=1e-12, atol=0.0)
                        )


class AtAStationaryPointTest(unittest.TestCase):
    """Where F's gradient is at its own rounding, the closed form is autograd's by the rule.

    At a stationary point of each smooth logistic problem (Newton's steps in
    float64; tanh's loss is not convex enough for them to settle),
    ``grad_norm_sq`` is the square of rounding noise: the two forms' values
    are within ``rounding_tolerance``, which there is wider than 1e-12 of
    either -- the case the relative bound cannot hold (POST-F38).
    """

    def test_every_smooth_problem(self) -> None:
        from fedbrew.core.grad_norm import FusedPass, rounding_tolerance, within_rounding

        generator = torch.Generator().manual_seed(17)
        rows, dim = 64, 8
        for loss, penalty in problem.PROBLEMS:
            if (loss, penalty) not in (("logistic", "l2sq"), ("logistic", "nonconvex")):
                continue
            with self.subTest(problem=f"{loss}+{penalty}"):
                task = _small_task(loss, penalty)
                model = problem.LogisticModel(dim, 0.03, loss=loss, penalty=penalty)
                features = torch.randn(rows, dim, generator=generator, dtype=torch.float64)
                labels = torch.where(torch.rand(rows, generator=generator) < 0.5, -1.0, 1.0)
                pooled = (features, labels.double())

                def objective(
                    x: torch.Tensor, task: Any = task, model: Any = model, rows: Any = pooled
                ) -> torch.Tensor:
                    return task.functional_loss(model, {"x": x}, {}, rows, None)[0]

                x = torch.zeros(dim, dtype=torch.float64)
                for _ in range(40):
                    gradient = torch.autograd.functional.jacobian(objective, x)
                    hessian = torch.autograd.functional.hessian(objective, x)
                    x = x - torch.linalg.solve(hessian, gradient)
                values = {}
                for name, closed in (("closed", model), ("autograd", None)):
                    fused = FusedPass(task, pooled, rows, True, closed=closed)
                    values[name] = float(fused.measure(model, {"x": x}, {})[2])
                tolerance = rounding_tolerance(task, model, {"x": x}, {}, pooled)
                self.assertLess(max(values.values()), 1e-24)
                self.assertGreater(tolerance, 1e-12 * max(values.values()))
                self.assertTrue(within_rounding(values["closed"], values["autograd"], tolerance))


def _run_task_class() -> Any:
    """The task class a run builds: the extension as ``load_extensions`` imports it.

    ``extensions._import_file`` runs the file afresh, so ``problem`` above is
    another module; what a run must see is patched on this one.
    """

    extensions.load_extensions([str(EXTENSION)])
    name = f"{extensions._FILE_MODULE_PREFIX}.{extensions._sha256(EXTENSION)[:16]}"
    return sys.modules[name].FedLogisticL1Task


def _untimed(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return [
            {key: value for key, value in row.items() if not key.endswith("_sec")}
            for row in csv.DictReader(handle)
        ]


if __name__ == "__main__":
    unittest.main()
