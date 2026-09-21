"""Focused end-to-end regressions for causal-LM and classification runs."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import textwrap
import unittest
from pathlib import Path

from fedbrew.core import runner
from fedbrew.data.generate import generate_from_config
from fedbrew.data.manifest_dataset import ManifestFederatedDataset

_PROJECT_ROOT = Path(__file__).parents[1]
_CLASSIFICATION_SMOKE = _PROJECT_ROOT / "configs" / "dev" / "smoke_eval.yaml"


class CausalLMEndToEndTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("transformers") is not None,
        "transformers is an optional LLM dependency",
    )
    def test_one_federated_round_writes_metrics_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = _write_corpus(root)
            generated_dir = root / "generated"
            generator_config = root / "generator.yaml"
            generator_config.write_text(
                textwrap.dedent(f"""
                    dataset:
                      name: tiny_causal_lm
                      output_dir: {generated_dir}
                      seed: 9
                    causal_lm:
                      corpus_path: {corpus_path}
                      sequence_length: 32
                    splits:
                      train_ratio: 0.8
                      test_ratio: 0.2
                    client_splits:
                      train_ratio: 0.8
                      eval_ratio: 0.2
                    partition:
                      strategy: iid
                      num_clients: 2
                    """).lstrip(),
                encoding="utf-8",
            )
            manifest_path = generate_from_config(generator_config)

            output_dir = root / "run"
            experiment_config = root / "experiment.yaml"
            experiment_config.write_text(
                textwrap.dedent(f"""
                    experiment:
                      name: causal_lm_test
                      seed: 9
                      output_dir: {output_dir}
                    server:
                      strategy: fedavg
                      participation_rate: 1
                      metrics:
                        - fit_loss
                        - fit_accuracy
                    client:
                      update_rule: local_adamw
                      batch_size: 4
                      learning_rate: 0.0005
                      learning_rate_schedule: constant
                      min_learning_rate: 0.0
                      weight_decay: 0.01
                      beta1: 0.9
                      beta2: 0.999
                      epsilon: 1.0e-8
                      metrics:
                        - fit_loss
                        - fit_accuracy
                    data:
                      name: manifest_dataset
                      path: {manifest_path}
                    model:
                      name: tiny_gpt2
                      vocab_size: 258
                      sequence_length: 32
                      n_embd: 8
                      n_layer: 1
                      n_head: 2
                      dropout: 0.0
                    runtime:
                      deterministic: true
                      deterministic_warn_only: true
                      device: cpu
                      use_amp: false
                      checkpointing:
                        enabled: true
                        interval: 1
                        save_last: true
                        save_best: true
                        best_metric: val_loss_sample_weighted_avg
                        keep_last: 1
                        save_every_round: true
                    evaluation:
                      train:
                        every: 1
                        clients: participating
                      val:
                        every: 1
                        clients: all
                      test:
                        every: never
                      central_test:
                        every: 1
                    client_statistics:
                      per_client_csv: true
                      std: true
                      variance: false
                      min: true
                      max: true
                      worst_percent: 10
                    defaults:
                      global_rounds: 1
                      local_iterations: 1
                    """).lstrip(),
                encoding="utf-8",
            )

            state = runner.run(experiment_config)

            self.assertEqual(len(state.metrics_history), 1)
            metrics = state.metrics_history[0].metrics
            # The exact column set of a real round, written out rather than
            # derived: this is the one place a reader can see what a run of
            # this shape actually produces. The config states every
            # client_statistics toggle, so each aggregate below is a choice
            # the config made and not a default that could move underneath it.
            required_metrics = {
                # The fit phase, aggregated over participating clients.
                "fit_loss",
                "fit_accuracy",
                # One post-aggregation pass per evaluated split, each carrying
                # the same six aggregates for each of the two metric bases.
                *(
                    f"{split}_{base}_{statistic}"
                    for split in ("train", "val")
                    for base in ("loss", "accuracy")
                    for statistic in (
                        "sample_weighted_avg",
                        "avg",
                        "std",
                        "min",
                        "max",
                        "worst10",
                    )
                ),
                # How many clients each split's aggregates are over. Not a
                # client_statistics toggle: every evaluated split writes it,
                # whatever the config says (P07-F06).
                "train_num_clients",
                "val_num_clients",
                # The server's own held-out set: one number, no clients to
                # spread across.
                "central_test_loss",
                "central_test_accuracy",
            }
            self.assertEqual(set(metrics), required_metrics)
            self.assertTrue(all(math.isfinite(metrics[name]) for name in metrics))

            dataset = ManifestFederatedDataset(manifest_path)
            expected_tokens = sum(
                int(dataset.get_client_data(client_id)["train"]["y"].ne(0).sum())
                for client_id in dataset.list_clients()
            )
            self.assertEqual(state.metrics_history[0].num_examples, expected_tokens)

            artifact_names = {
                "run.json",
                "round_metrics.csv",
                "client_metrics.csv",
                "client_update_metrics.csv",
            }
            for name in artifact_names:
                self.assertTrue((output_dir / name).is_file(), name)
            # Retired: the .jsonl twins duplicated the CSVs, and results.json
            # duplicated everything. Assert they stay gone so they cannot be
            # reintroduced without someone noticing the storage cost.
            for name in (
                "metrics.jsonl",
                "client_metrics.jsonl",
                "client_update_metrics.jsonl",
                "results.json",
            ):
                self.assertFalse((output_dir / name).exists(), name)
            self.assertTrue((output_dir / "checkpoints" / "round_001.pt").is_file())
            self.assertTrue((output_dir / "checkpoints" / "best.pt").is_file())


class ClassificationSmokeRegressionTests(unittest.TestCase):
    def test_existing_classification_smoke_finishes_one_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "classification"
            args = runner.parse_args(
                [
                    "--rounds",
                    "1",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                    "--quiet",
                ]
            )

            state = runner.run(_CLASSIFICATION_SMOKE, args)

            self.assertEqual(len(state.metrics_history), 1)
            for value in state.metrics_history[0].metrics.values():
                self.assertTrue(math.isfinite(value))


def _write_corpus(root: Path) -> Path:
    """Write the causal-LM corpus this test trains on, into its own tmpdir.

    It used to read `data/raw/sample_data/tiny_causal_lm.txt`, which
    `.gitignore` excludes, so the file has never existed in a clone and the
    test raised FileNotFoundError on every machine nobody had hand-prepared.
    The generator's own tests have always built their corpus this way.

    120 records at roughly 45 bytes each is about 5,400 byte tokens, which is
    ~168 sequences of 32: enough that the 0.8/0.2 source split and the 0.8/0.2
    client split both leave every one of the two clients a non-empty train and
    eval shard.
    """

    corpus_path = root / "corpus.txt"
    corpus_path.write_text(
        "\n".join(f"record {index:03d} contains deterministic local text" for index in range(120)),
        encoding="utf-8",
    )
    return corpus_path


if __name__ == "__main__":
    unittest.main()
