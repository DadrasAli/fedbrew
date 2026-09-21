"""Evaluate the frozen pretrained base model on a run's global test set.

LoRA initialises B to zero, so the federated model at round 0 is exactly the
base model. The framework logs no round-0 evaluation, so this reports the
reference point that every federated loss curve starts from.

Metrics come from the experiment's own task adapter, so loss and accuracy are
computed identically to the logged ``central_test_loss`` / ``central_test_accuracy``.
"""

from __future__ import annotations

import argparse
import json

from fedbrew.core.config import load_config
from fedbrew.core.factory import _build_dataset, _build_task, _model_config
from fedbrew.core.metrics import json_safe
from fedbrew.core.registry import (
    datasets,
    register_builtin_components,
    tasks,
)
from fedbrew.tasks.base import SupportsDatasetEvaluation

LORA_ONLY_KEYS = (
    "adapter_name",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
)


def main() -> None:
    """Score the frozen pretrained base model on a run's central test set.

    Reads ``--config`` (a training run config, required) and optionally writes
    the result record to ``--output``; without it the record only goes to
    stdout.

    This is the round-0 reference point every federated loss curve starts from.
    LoRA initialises B to zero, so the federated model at round 0 is exactly
    the base model, and the framework logs no round-0 evaluation. The adapter
    is dropped so the base model is measured on its own, and metrics come from
    the experiment's own task adapter -- so the numbers are computed identically
    to the logged ``central_test_loss`` / ``central_test_accuracy`` and are
    directly comparable to them.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    register_builtin_components()

    dataset = _build_dataset(config, datasets.get(config.data.name))
    model_config = _model_config(config, dataset)

    # Drop the adapter so the base model is evaluated on its own.
    model_config["name"] = "hf_causal_lm"
    for key in LORA_ONLY_KEYS:
        model_config.pop(key, None)

    task = _build_task(config, tasks.get(config.task.name), model_config)
    # Both registered tasks implement the optional whole-dataset evaluator, so
    # this holds for every config that reaches here. A task registered outside
    # the package need not, and the two servers answer that case by dropping
    # the central_test_* columns; this command *is* the measurement, so it has
    # nothing to degrade to and says which task cannot make it.
    if not isinstance(task, SupportsDatasetEvaluation):
        raise SystemExit(
            f"task {config.task.name!r} ({type(task).__name__}) does not implement "
            "evaluate_model, so the base-model reference cannot be measured with it"
        )
    model = task.build_model(model_config)
    metrics = task.evaluate_model(model, dataset.get_global_data())

    record = {
        "reference": "pretrained_base_model_round_0",
        "config": args.config,
        "model": config.model.extra.get("asset_manifest"),
        "data": config.data.path,
        "central_test_loss": metrics["loss"],
        "central_test_accuracy": metrics["accuracy"],
    }
    # A measured metric can be inf or nan; json.dumps would write the bare
    # token, which strict readers refuse. Same contract as run.json.
    record = json_safe(record)
    print(json.dumps(record, indent=2, allow_nan=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
