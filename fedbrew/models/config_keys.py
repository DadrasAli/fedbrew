"""Reject a model config key no builder reads.

Every builder resolves its parameters with ``values.get(name, default)``, so a
misspelled key used to be dropped by the dict and replaced by the default:
``lora_alph: 32`` ran at 16 with nothing to say so. The builders are the only
place that knows which names are real, so each one declares its set and calls
this.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: Keys fedbrew.core.factory._model_config injects into every builder's
#: mapping: the registry name it dispatched on, the three shared dimensions,
#: and -- for causal_lm -- the manifest metadata, which is namespaced.
_INJECTED_KEYS = frozenset({"name", "input_dim", "hidden_dim", "num_classes"})
_INJECTED_PREFIX = "dataset_"


def forwarded_model_keys(
    values: Mapping[str, Any],
    known: Iterable[str],
) -> dict[str, Any]:
    """Narrow a builder's mapping to what an inner builder may be handed.

    hf_causal_lm_lora builds its base model with hf_causal_lm, which runs the
    same check: passing the whole mapping down would have the inner builder
    reject the outer one's own keys. The inner builder still has to be strict
    when it is the registered builder, so the boundary is narrowed here
    instead, by the same rules reject_unknown_model_keys applies.
    """

    accepted = frozenset(known) | _INJECTED_KEYS
    return {
        name: value
        for name, value in values.items()
        if name in accepted or name.startswith(_INJECTED_PREFIX)
    }


def reject_unknown_model_keys(
    values: Mapping[str, Any],
    known: Iterable[str],
    builder: str,
) -> None:
    """Raise if ``values`` carries a key ``builder`` will never read."""

    accepted = frozenset(known) | _INJECTED_KEYS
    unknown = sorted(
        name for name in values if name not in accepted and not name.startswith(_INJECTED_PREFIX)
    )
    if not unknown:
        return
    raise ValueError(
        f"model {builder!r} does not read: "
        + ", ".join(f"model.{name}" for name in unknown)
        + ". It reads: "
        + ", ".join(sorted(accepted - _INJECTED_KEYS))
        + ". An unread key takes the builder's default instead, so a "
        "misspelling silently builds a different model."
    )
