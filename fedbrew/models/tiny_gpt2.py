"""Randomly initialized, compact GPT-2 causal language model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from fedbrew.models.config_keys import reject_unknown_model_keys

#: Every key build_tiny_gpt2 below reads.
_KNOWN_KEYS = frozenset(
    {
        "vocab_size",
        "sequence_length",
        "n_embd",
        "n_layer",
        "n_head",
        "n_positions",
        "dropout",
        "resid_pdrop",
        "embd_pdrop",
        "attn_pdrop",
        "layer_norm_epsilon",
        "initializer_range",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    }
)


def build_tiny_gpt2(config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build a tiny GPT-2 LM from configuration without downloading weights."""

    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent.
        raise ModuleNotFoundError(
            "tiny_gpt2 requires the optional LLM dependencies; "
            'install them with pip install -e ".[llm]"'
        ) from error

    values = dict(config or {})
    reject_unknown_model_keys(values, _KNOWN_KEYS, "tiny_gpt2")
    vocab_size = _positive_int(values, "vocab_size", 258)
    sequence_length = _positive_int(values, "sequence_length", 32)
    n_embd = _positive_int(values, "n_embd", 64)
    n_layer = _positive_int(values, "n_layer", 2)
    n_head = _positive_int(values, "n_head", 2)
    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    configured_positions = _positive_int(values, "n_positions", sequence_length)
    n_positions = max(sequence_length, configured_positions)
    dropout = _probability(values, "dropout", 0.0)

    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=_probability(values, "resid_pdrop", dropout),
        embd_pdrop=_probability(values, "embd_pdrop", dropout),
        attn_pdrop=_probability(values, "attn_pdrop", dropout),
        layer_norm_epsilon=float(values.get("layer_norm_epsilon", 1e-5)),
        initializer_range=float(values.get("initializer_range", 0.02)),
        bos_token_id=int(values.get("bos_token_id", 1)),
        eos_token_id=int(values.get("eos_token_id", 1)),
        pad_token_id=_optional_token_id(values, "pad_token_id", 0),
        use_cache=False,
    )
    return GPT2LMHeadModel(model_config)


def _optional_token_id(values: Mapping[str, Any], name: str, default: int) -> int | None:
    """Return the configured token id, or None when the dataset declares none.

    `pad_token_id` is the one key here the factory writes rather than the
    author: `_add_causal_manifest_metadata` copies the manifest's
    `padding_token_id` over it, and both SFT generators write that as `null`.
    So an SFT manifest hands this builder an explicit None, which `int()` used
    to turn into a bare `TypeError` from inside a model factory -- the pairing
    is legal, and nothing shipped happened to make it. `hf_causal_lm` has
    always accepted None for the same key (`_validate_padding_token`), and
    GPT2Config takes None as "no padding token".

    An absent key still takes the default; only an explicit None is passed on.
    """

    if name not in values:
        return default
    value = values[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null")
    return int(value)


def _positive_int(values: Mapping[str, Any], name: str, default: int) -> int:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _probability(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be in [0, 1)")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be in [0, 1)") from error
    if not 0.0 <= parsed < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return parsed
