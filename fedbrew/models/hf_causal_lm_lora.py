"""Strictly local Hugging Face causal LM wrapped with a PEFT LoRA adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from torch import nn

from fedbrew.data.llm_assets.manifest import load_asset_manifest
from fedbrew.models.config_keys import forwarded_model_keys, reject_unknown_model_keys
from fedbrew.models.hf_causal_lm import _KNOWN_KEYS as BASE_MODEL_KEYS
from fedbrew.models.hf_causal_lm import (
    HFCausalLMConfigError,
    build_hf_causal_lm,
)


class HFCausalLMLoRAConfigError(HFCausalLMConfigError):
    """Raised when a LoRA model configuration violates the FL contract."""


#: Every key build_hf_causal_lm_lora below reads, plus the base-model keys it
#: forwards to build_hf_causal_lm.
_KNOWN_KEYS = frozenset(
    {
        "adapter_name",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "bias",
    }
    | BASE_MODEL_KEYS
)


def lora_adapter_name(values: Mapping[str, Any]) -> str:
    """The name this builder attaches the adapter under.

    Read here rather than defaulted a second time by whoever needs to know.
    See :func:`lora_config_from_model_values`.
    """

    return _non_empty_string(values.get("adapter_name", "default"), "adapter_name")


def lora_config_from_model_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """The LoRA configuration these model keys resolve to.

    Every default and every normalisation in one place, because two places is
    what this replaces: `run_metadata.build_hf_causal_lm_trace` wrote
    `run.json`'s `lora_config` block by re-reading the same keys with its own
    copies of `8`, `16`, `0.05` and `"none"`. Changing a default here would
    have left `run.json` describing an adapter that never ran, and nothing
    compared the two. It also recorded `target_modules` unnormalised, so a
    config with a stray space wrote one list and trained on another. P10-F17.

    Args:
        values: The `model` block's keys, as the factory assembles them.

    Returns:
        The keyword arguments for `peft.LoraConfig`, with `task_type` as the
        string PEFT's enum resolves to -- a JSON-writable record of the object
        the builder constructs.

    Raises:
        HFCausalLMLoRAConfigError: If any value is outside its contract. The
            order is the builder's, so a config wrong in two ways is reported
            the same way whichever caller asks first.
    """

    r = _positive_int(values.get("r", 8), "r")
    lora_alpha = _positive_int(values.get("lora_alpha", 16), "lora_alpha")
    lora_dropout = _dropout(values.get("lora_dropout", 0.05))
    target_modules = _target_modules(values.get("target_modules"))
    bias = _bias(values.get("bias", "none"))
    if bias != "none":
        raise HFCausalLMLoRAConfigError(
            "adapter-only federation requires model.bias=none so no frozen "
            "base-model bias tensors become trainable"
        )
    return {
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": list(target_modules),
        "bias": bias,
    }


def build_hf_causal_lm_lora(
    config: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Load a prepared base model and attach an unmerged PEFT LoRA adapter."""

    values = dict(config or {})
    reject_unknown_model_keys(values, _KNOWN_KEYS, "hf_causal_lm_lora")
    adapter_name = lora_adapter_name(values)
    normalized_config = lora_config_from_model_values(values)
    target_modules = tuple(normalized_config["target_modules"])
    bias = str(normalized_config["bias"])

    base_model = build_hf_causal_lm(forwarded_model_keys(values, BASE_MODEL_KEYS))
    _validate_targets(base_model, target_modules)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as error:  # pragma: no cover - install dependent.
        raise ModuleNotFoundError(
            "hf_causal_lm_lora requires the optional LLM dependencies; "
            'install them with pip install -e ".[llm]"'
        ) from error

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=normalized_config["r"],
        lora_alpha=normalized_config["lora_alpha"],
        lora_dropout=normalized_config["lora_dropout"],
        target_modules=list(target_modules),
        bias=bias,
    )
    model = get_peft_model(cast(Any, base_model), peft_config, adapter_name=adapter_name)
    _validate_trainable_parameters(model, bias)
    _validate_unmerged(model)

    manifest = load_asset_manifest(
        values["asset_manifest"],
        preparation_config=values.get("preparation_config"),
        require_assets=True,
    )
    model._fl_model_state_scope = "adapter"
    model._fl_base_model_identifier = manifest.model_identifier
    model._fl_base_model_resolved_revision = (
        manifest.resolved_revision or manifest.requested_revision
    )
    model._fl_adapter_name = adapter_name
    model._fl_lora_config = normalized_config
    return cast(nn.Module, model)


def _validate_targets(model: nn.Module, targets: tuple[str, ...]) -> None:
    module_names = tuple(name for name, _ in model.named_modules() if name)
    unmatched = [
        target
        for target in targets
        if not any(name == target or name.endswith(f".{target}") for name in module_names)
    ]
    if unmatched:
        raise HFCausalLMLoRAConfigError(
            "LoRA target_modules do not match base-model modules: " + ", ".join(unmatched)
        )


def _validate_trainable_parameters(model: nn.Module, bias: str) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise HFCausalLMLoRAConfigError("LoRA construction produced no trainable parameters")
    unexpected = [name for name in trainable if "lora_" not in name]
    if unexpected:
        raise HFCausalLMLoRAConfigError(
            "non-adapter base-model parameters remain trainable: " + ", ".join(unexpected)
        )


def _validate_unmerged(model: nn.Module) -> None:
    merged_modules = [
        name for name, module in model.named_modules() if bool(getattr(module, "merged", False))
    ]
    if merged_modules:
        raise HFCausalLMLoRAConfigError(
            "LoRA adapters must remain unmerged during federated training"
        )


def _target_modules(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, list | tuple)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise HFCausalLMLoRAConfigError(
            "model.target_modules must be a non-empty list of module names"
        )
    normalized = tuple(str(item).strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise HFCausalLMLoRAConfigError("model.target_modules must not contain duplicates")
    return normalized


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HFCausalLMLoRAConfigError(f"model.{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalLMLoRAConfigError(f"model.{name} must be a positive integer")
    return value


def _dropout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HFCausalLMLoRAConfigError("model.lora_dropout must be numeric")
    normalized = float(value)
    if not 0.0 <= normalized < 1.0:
        raise HFCausalLMLoRAConfigError("model.lora_dropout must be in [0, 1)")
    return normalized


def _bias(value: Any) -> Literal["none", "all", "lora_only"]:
    normalized = _non_empty_string(value, "bias")
    if normalized not in {"none", "all", "lora_only"}:
        raise HFCausalLMLoRAConfigError("model.bias must be one of: all, lora_only, none")
    return cast(Literal["none", "all", "lora_only"], normalized)
