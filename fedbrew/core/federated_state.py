"""Validation helpers for task-defined federated model-state payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

MODEL_STATE_SCOPES = frozenset({"full", "adapter"})
_ADAPTER_COMPATIBILITY_FIELDS = (
    "base_model_identifier",
    "base_model_resolved_revision",
    "adapter_name",
    "lora_config",
)


#: The built-in client rules that federate adapter-only state: each loads,
#: extracts and weights the model through the task's hooks
#: (`TaskAdapter.load_federated_model_state` and the rest), so a LoRA model
#: moves only its adapter. Measured on 2026-09-21: each trains a real
#: adapter-only round on a tiny GPT-2 LoRA model with the `fedavg` server and
#: every FedOpt server, and `centralized` with its own strategy.
ADAPTER_STATE_CLIENT_RULES = frozenset(
    {"fedavg", "centralized", "fedavg_ft", "local_sgd", "local_adamw", "delta_sgd"}
)

#: The built-in client rules that cannot train adapter-only state, and why.
#: Refused at config load for a model the package builds adapter-scoped
#: (`config.ADAPTER_SCOPED_MODELS`), and by the client before its first update
#: for any task that reports adapter scope, an extension's included.
#: FINDINGS.csv POST-F29.
FULL_STATE_ONLY_CLIENT_RULES: Mapping[str, str] = {
    "fedprox": "it loads and returns the whole model's state_dict rather than "
    "the task's federated state, so an adapter-only broadcast does not load",
    "scaffold": "it loads and returns the whole model's state_dict rather than "
    "the task's federated state, and keys its control variates by it",
    "fedlalr": "its local AMSGrad looks its moments up by the model's parameter "
    "names, which under a PEFT adapter carry the adapter name the federated "
    "state's keys do not",
}


def adapter_state_refusal(rule: str) -> str:
    """The reason ``rule`` cannot train adapter-only state, and what can."""

    return (
        f"client.update_rule={rule} cannot train adapter-only (LoRA) state: "
        f"{FULL_STATE_ONLY_CLIENT_RULES[rule]}. Rules that can: "
        f"{', '.join(sorted(ADAPTER_STATE_CLIENT_RULES))} -- with server.strategy "
        "fedavg or a FedOpt strategy, and centralized with its own. Or train the "
        "whole model with a full-state model."
    )


def refuse_adapter_state(client: Any, rule: str, model: Any) -> None:
    """Raise before any update if ``client``'s task federates adapter state.

    The client-side half of POST-F29: config load refuses the adapter models
    the package ships, and this catches a task, an extension's included, that
    reports adapter scope for a model config load could not judge. Asked once
    per client, because the answer follows from the task and model config,
    which a client keeps, and the question copies the federated state.
    """

    if getattr(client, "_adapter_scope_checked", False):
        return
    metadata = client.task.federated_model_state_metadata(model)
    if metadata.get("model_state_scope", "full") != "full":
        raise ValueError(adapter_state_refusal(rule))
    client._adapter_scope_checked = True


#: The built-in client rules whose `fit` asks the task for the client's
#: aggregation weight (`TaskAdapter.federated_aggregation_weight`), and the two
#: that report their post-fit evaluation count directly instead. The causal-LM
#: task answers that question with `model.active_target_weighting`, so under
#: the second set the key has no effect. FINDINGS.csv POST-F30.
AGGREGATION_WEIGHT_HOOK_CLIENT_RULES = frozenset(
    {"fedavg", "centralized", "fedavg_ft", "local_sgd", "local_adamw", "delta_sgd", "fedlalr"}
)
AGGREGATION_WEIGHT_HOOK_BYPASS_RULES = frozenset({"fedprox", "scaffold"})


def active_target_weighting_refusal(
    rule: str,
    task: str,
    model_values: Mapping[str, Any],
    dataset_task: Any = None,
) -> str | None:
    """Why ``rule`` cannot run with active-target weighting on, if it cannot.

    The weighting is on when ``model.active_target_weighting`` says true, or
    when it is unset and the dataset's task is ``causal_lm_sft``, which is the
    task's own default (`TorchCausalLMTask`). Config load sees only the key;
    the default is judged where the manifest is read, on both paths.
    """

    if task != "causal_lm" or rule not in AGGREGATION_WEIGHT_HOOK_BYPASS_RULES:
        return None
    explicit = "active_target_weighting" in model_values
    enabled = model_values.get("active_target_weighting", dataset_task == "causal_lm_sft")
    if enabled is not True:
        return None
    source = (
        "model.active_target_weighting: true"
        if explicit
        else "active-target weighting, which a causal_lm_sft dataset turns on by default"
    )
    return (
        f"client.update_rule={rule} does not honour {source}: {rule} reports the "
        "active target tokens of the whole train split, counted after fitting, and "
        "never asks the task for the tokens its training batches held, so the key "
        "would be recorded and ignored. Set model.active_target_weighting: false to "
        "weight by the train split's tokens, which is what this rule does, or use a "
        "rule that honours it: "
        f"{', '.join(sorted(AGGREGATION_WEIGHT_HOOK_CLIENT_RULES))}."
    )


def model_state_size(state: Mapping[str, Any]) -> tuple[int, int]:
    """Return tensor elements and bytes communicated by a model state."""

    parameters = 0
    num_bytes = 0
    for key, value in state.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"federated model state value for {key!r} is not a tensor")
        parameters += int(value.numel())
        num_bytes += int(value.numel() * value.element_size())
    return parameters, num_bytes


def payload_model_state_scope(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> str:
    """Read and validate a payload state scope, preserving old full-state payloads."""

    metadata = payload.get("model_state_metadata")
    metadata_scope: Any = None
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{context} model_state_metadata must be a mapping")
        metadata_scope = metadata.get("model_state_scope")

    raw_scope = payload.get("model_state_scope", metadata_scope)
    if raw_scope is None:
        raw_scope = "full"
    scope = _normalize_scope(raw_scope, context)
    if metadata_scope is not None and _normalize_scope(metadata_scope, context) != scope:
        raise ValueError(f"{context} contains conflicting model state scopes")
    return scope


def validate_federated_state_metadata(
    expected: Mapping[str, Any],
    received: Mapping[str, Any] | None,
    *,
    received_scope: str,
    context: str,
) -> None:
    """Reject mixed scopes and incompatible adapter/base-model identities."""

    expected_scope = _normalize_scope(expected.get("model_state_scope", "full"), context)
    actual_scope = _normalize_scope(received_scope, context)
    if actual_scope != expected_scope:
        raise ValueError(
            f"{context} has incompatible model state scopes: "
            f"expected {expected_scope!r}, received {actual_scope!r}"
        )

    if actual_scope != "adapter":
        return
    if not isinstance(received, Mapping):
        raise ValueError(f"{context} adapter state requires model_state_metadata")
    for field in _ADAPTER_COMPATIBILITY_FIELDS:
        if field not in expected or field not in received:
            raise ValueError(
                f"{context} adapter state metadata is missing compatibility field {field!r}"
            )
        if received[field] != expected[field]:
            raise ValueError(
                f"{context} has incompatible adapter metadata for {field}: "
                f"expected {expected[field]!r}, received {received[field]!r}"
            )


def validate_state_matches(
    expected: Mapping[str, Any],
    received: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject a state whose keys or tensor shapes differ from the server's.

    The accumulator already holds every client to the first client's keys and
    shapes; this holds the first client to the server's, so a round cannot
    replace the global state with one describing different parameters.
    """

    missing = sorted(set(expected) - set(received))
    unexpected = sorted(set(received) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{context} model_state does not describe the server's parameters: "
            f"missing {missing}, unexpected {unexpected}"
        )
    for key, value in received.items():
        reference = expected[key]
        if isinstance(value, Tensor) and isinstance(reference, Tensor):
            if value.shape != reference.shape:
                raise ValueError(
                    f"{context} model_state tensor {key!r} has shape "
                    f"{tuple(value.shape)}, the server's has {tuple(reference.shape)}"
                )


def _normalize_scope(value: Any, context: str) -> str:
    if not isinstance(value, str) or value not in MODEL_STATE_SCOPES:
        supported = ", ".join(sorted(MODEL_STATE_SCOPES))
        raise ValueError(f"{context} model_state_scope must be one of: {supported}")
    return value
