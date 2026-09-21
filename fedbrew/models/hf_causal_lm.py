"""Strict offline factory for prepared Hugging Face causal language models."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from torch import Tensor, nn

from fedbrew.data.llm_assets.manifest import (
    AssetManifest,
    load_asset_manifest,
    preparation_command_for,
)
from fedbrew.models.config_keys import reject_unknown_model_keys


class HFCausalLMConfigError(ValueError):
    """Raised when cached model, tokenizer, and dataset metadata are incompatible."""


class HFCausalLMLoadError(RuntimeError):
    """Raised when a prepared model cannot be loaded strictly offline."""


#: Every key build_hf_causal_lm below reads. The dataset_* names it also
#: consults are injected by the factory from the manifest, not written by hand.
_KNOWN_KEYS = frozenset(
    {
        "asset_manifest",
        "preparation_config",
        "local_files_only",
        "trust_remote_code",
        "sequence_length",
        "vocab_size",
        "pad_token_id",
    }
)


def build_hf_causal_lm(config: Mapping[str, Any] | None = None) -> nn.Module:
    """Load pretrained causal-LM weights from explicitly prepared local assets."""

    values = dict(config or {})
    reject_unknown_model_keys(values, _KNOWN_KEYS, "hf_causal_lm")
    manifest_path = _required_path(values, "asset_manifest")
    preparation_config = _optional_path(values.get("preparation_config"))
    _require_strict_offline_flags(values)

    manifest = load_asset_manifest(
        manifest_path,
        preparation_config=preparation_config,
        require_assets=True,
    )
    _validate_dataset_asset_trace(values, manifest)
    expected_vocab_size = _expected_vocab_size(values, manifest)
    expected_sequence_length = _expected_sequence_length(values)
    _validate_padding_token(values, expected_vocab_size)

    try:
        from transformers import AutoModelForCausalLM
    except ModuleNotFoundError as error:  # pragma: no cover - install dependent.
        raise ModuleNotFoundError(
            "hf_causal_lm requires the optional LLM dependencies; "
            'install them with pip install -e ".[llm]"'
        ) from error

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(manifest.model_asset_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        command = preparation_command_for(manifest_path, preparation_config)
        raise HFCausalLMLoadError(
            "Could not load the prepared Hugging Face causal LM strictly locally "
            f"from {manifest.model_asset_path}. Run: {command}"
        ) from error

    _validate_loaded_vocabulary(model, expected_vocab_size)
    _validate_sequence_capacity(model, expected_sequence_length)
    _preserve_tied_embeddings(model)
    return cast(nn.Module, model)


def _require_strict_offline_flags(values: Mapping[str, Any]) -> None:
    local_files_only = values.get("local_files_only", True)
    if local_files_only is not True:
        raise HFCausalLMConfigError("hf_causal_lm requires model.local_files_only=true")
    trust_remote_code = values.get("trust_remote_code", False)
    if trust_remote_code is not False:
        raise HFCausalLMConfigError("hf_causal_lm requires model.trust_remote_code=false")


def _expected_vocab_size(
    values: Mapping[str, Any],
    manifest: AssetManifest,
) -> int:
    declared: dict[str, int] = {
        "asset manifest": manifest.vocabulary_size,
    }
    for key, label in (
        ("vocab_size", "model config"),
        ("dataset_vocab_size", "generated dataset"),
        ("dataset_tokenizer_vocabulary_size", "generated tokenizer"),
    ):
        raw_value = values.get(key)
        if raw_value is not None:
            declared[label] = _positive_int(raw_value, key)

    if len(set(declared.values())) != 1:
        details = ", ".join(f"{label}={value}" for label, value in declared.items())
        raise HFCausalLMConfigError(f"incompatible tokenizer/model vocabulary sizes: {details}")
    return manifest.vocabulary_size


def _expected_sequence_length(values: Mapping[str, Any]) -> int:
    configured = values.get("sequence_length")
    generated = values.get("dataset_sequence_length")
    if configured is not None and generated is not None:
        configured_value = _positive_int(configured, "sequence_length")
        generated_value = _positive_int(generated, "dataset_sequence_length")
        if configured_value != generated_value:
            raise HFCausalLMConfigError(
                "model.sequence_length does not match the generated dataset: "
                f"{configured_value} != {generated_value}"
            )
    raw_value = generated if generated is not None else configured
    if raw_value is None:
        raise HFCausalLMConfigError(
            "hf_causal_lm requires sequence_length from the generated manifest "
            "or model configuration"
        )
    return _positive_int(raw_value, "sequence_length")


def _validate_padding_token(
    values: Mapping[str, Any],
    vocabulary_size: int,
) -> None:
    raw_value = values.get("dataset_padding_token_id", values.get("pad_token_id"))
    if raw_value is None:
        return
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise HFCausalLMConfigError("generated padding_token_id must be an integer or null")
    if not 0 <= raw_value < vocabulary_size:
        raise HFCausalLMConfigError(
            "generated padding_token_id is outside the tokenizer vocabulary"
        )


def _validate_dataset_asset_trace(
    values: Mapping[str, Any],
    manifest: AssetManifest,
) -> None:
    dataset_manifest_path = values.get(
        "dataset_asset_manifest",
        values.get("dataset_tokenizer_asset_manifest"),
    )
    if dataset_manifest_path is not None:
        if not isinstance(dataset_manifest_path, str | Path):
            raise HFCausalLMConfigError("generated tokenizer asset-manifest path must be a string")
        if _normalized_path(dataset_manifest_path) != _normalized_path(manifest.manifest_path):
            raise HFCausalLMConfigError(
                "model asset_manifest does not match the tokenizer assets used "
                "to generate the dataset"
            )

    tokenizer_identifier = values.get("dataset_tokenizer_identifier")
    if tokenizer_identifier is not None and tokenizer_identifier != manifest.tokenizer_identifier:
        raise HFCausalLMConfigError(
            "prepared tokenizer identifier does not match the generated dataset"
        )

    requested_revision = values.get("dataset_tokenizer_requested_revision")
    if requested_revision is not None and requested_revision != manifest.requested_revision:
        raise HFCausalLMConfigError(
            "prepared tokenizer requested revision does not match the dataset"
        )

    resolved_revision = values.get("dataset_tokenizer_resolved_revision")
    if resolved_revision is not None and resolved_revision != manifest.resolved_revision:
        raise HFCausalLMConfigError(
            "prepared tokenizer resolved revision does not match the dataset"
        )

    tokenizer_revision = values.get("dataset_tokenizer_revision")
    accepted_revisions = {
        manifest.requested_revision,
        manifest.resolved_revision,
    }
    if tokenizer_revision is not None and tokenizer_revision not in accepted_revisions:
        raise HFCausalLMConfigError(
            "prepared tokenizer revision does not match the generated dataset"
        )


def _validate_loaded_vocabulary(model: Any, expected_size: int) -> None:
    declared_size = getattr(getattr(model, "config", None), "vocab_size", None)
    sizes: dict[str, int] = {}
    if isinstance(declared_size, int):
        sizes["model config"] = declared_size

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_size = _embedding_size(input_embeddings)
    output_size = _embedding_size(output_embeddings)
    if input_size is not None:
        sizes["input embeddings"] = input_size
    if output_size is not None:
        sizes["output embeddings"] = output_size
    if not sizes:
        raise HFCausalLMConfigError("loaded causal LM does not expose vocabulary-sized embeddings")

    undersized = {label: size for label, size in sizes.items() if size < expected_size}
    if undersized:
        details = ", ".join(f"{label}={size}" for label, size in undersized.items())
        raise HFCausalLMConfigError(
            "loaded model vocabulary is smaller than the prepared tokenizer "
            f"({expected_size}): {details}"
        )
    if len(set(sizes.values())) != 1:
        details = ", ".join(f"{label}={size}" for label, size in sizes.items())
        raise HFCausalLMConfigError(
            "loaded model exposes inconsistent vocabulary capacities: " + details
        )


def _embedding_size(module: Any) -> int | None:
    weight = getattr(module, "weight", None)
    if isinstance(weight, Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])
    return None


def _validate_sequence_capacity(model: Any, sequence_length: int) -> None:
    model_config = getattr(model, "config", None)
    capacities: list[int] = []
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(model_config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            capacities.append(value)

    transformer = getattr(model, "transformer", None)
    position_embeddings = getattr(transformer, "wpe", None)
    num_embeddings = getattr(position_embeddings, "num_embeddings", None)
    if isinstance(num_embeddings, int) and num_embeddings > 0:
        capacities.append(num_embeddings)

    if not capacities:
        raise HFCausalLMConfigError("cannot determine the loaded model's supported sequence length")
    capacity = min(capacities)
    if sequence_length > capacity:
        raise HFCausalLMConfigError(
            "generated sequence_length exceeds the loaded model capacity: "
            f"{sequence_length} > {capacity}"
        )


def _preserve_tied_embeddings(model: Any) -> None:
    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()

    model_config = getattr(model, "config", None)
    if not bool(getattr(model_config, "tie_word_embeddings", False)):
        return

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if not isinstance(input_weight, Tensor) or not isinstance(output_weight, Tensor):
        raise HFCausalLMConfigError(
            "loaded model declares tied embeddings but does not expose both weights"
        )
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise HFCausalLMConfigError("loaded model failed to preserve tied input/output embeddings")


def _required_path(values: Mapping[str, Any], key: str) -> str | Path:
    value = values.get(key)
    if not isinstance(value, str | Path) or not str(value).strip():
        raise HFCausalLMConfigError(f"hf_causal_lm requires a non-empty model.{key}")
    return value


def _optional_path(value: Any) -> str | Path | None:
    if value is None:
        return None
    if not isinstance(value, str | Path) or not str(value).strip():
        raise HFCausalLMConfigError("model.preparation_config must be a non-empty path when set")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalLMConfigError(f"{name} must be a positive integer")
    return value


def _normalized_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded).absolute()
