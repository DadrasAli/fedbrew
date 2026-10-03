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
import tempfile
import unittest
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
