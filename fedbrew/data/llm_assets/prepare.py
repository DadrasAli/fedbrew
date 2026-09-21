"""Explicitly prepare self-contained Hugging Face model/tokenizer assets."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fedbrew.core.console import Rail, add_output_arguments, silent_rail, surface_from_args
from fedbrew.core.download_progress import redirect_huggingface_hub_progress
from fedbrew.core.logging import print_download_progress
from fedbrew.data.llm_assets.config import (
    AssetConfigError,
    AssetPreparationConfig,
    load_asset_config,
)
from fedbrew.data.llm_assets.manifest import (
    ASSET_MANIFEST_FILENAME,
    ASSET_MANIFEST_SCHEMA_VERSION,
    AssetManifest,
    AssetManifestError,
    load_asset_manifest,
)

_COMMIT_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


class AssetPreparationError(RuntimeError):
    """Raised when a configured Hugging Face revision cannot be prepared."""


#: The stages worth a line. Config load, the imports and the three mkdirs are
#: absent: none of them takes real time or reports anything new.
STAGES = ("cache", "model", "tokenizer", "revision")


def prepare_llm_assets(config_path: str | Path, rail: Rail | None = None) -> Path:
    """Download pinned assets once and save verified local copies.

    Skipped entirely when a cache already matches -- see
    ``_has_verified_matching_cache``.

    `rail` is where the stages report, defaulting to one that renders nothing
    so every existing caller keeps its old signature and its old silence.
    """

    rail = silent_rail() if rail is None else rail
    config = load_asset_config(config_path)
    cache_root = config.cache_dir.resolve()
    manifest_path = cache_root / ASSET_MANIFEST_FILENAME
    rail.detail("model id", config.model_identifier, note=f"revision {config.revision}")
    rail.detail("tokenizer id", config.tokenizer_identifier)
    rail.detail("cache root", str(cache_root))
    with rail.stage("cache") as stage:
        hit = _has_verified_matching_cache(manifest_path, config)
        stage.done("verified, nothing to fetch" if hit else "no usable cache, preparing")
    if hit:
        return manifest_path

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install mode.
        raise AssetPreparationError(
            'LLM asset preparation requires Transformers. Install with: pip install -e ".[llm]"'
        ) from exc

    download_cache = cache_root / "downloads"
    model_path = cache_root / "model"
    tokenizer_path = cache_root / "tokenizer"
    download_cache.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path.mkdir(parents=True, exist_ok=True)

    common_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "cache_dir": str(download_cache),
        "trust_remote_code": False,
    }
    # The two slow, network-bound, failure-prone stages, and the two that
    # cost nothing and were never reported: a model's type and a tokenizer's
    # vocabulary size are both checked here already and both discarded.
    with rail.stage("model") as stage:
        model = _load_pretrained(
            AutoModelForCausalLM,
            kind="model",
            identifier=config.model_identifier,
            config=config,
            kwargs=common_kwargs,
        )
        model.save_pretrained(model_path)
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if not isinstance(model_type, str) or not model_type:
            raise AssetPreparationError("prepared model does not declare a model_type")
        stage.done(model_type, note=_parameter_count(model))

    with rail.stage("tokenizer") as stage:
        tokenizer = _load_pretrained(
            AutoTokenizer,
            kind="tokenizer",
            identifier=config.tokenizer_identifier,
            config=config,
            kwargs=common_kwargs,
        )
        tokenizer.save_pretrained(tokenizer_path)
        vocabulary_size = len(tokenizer)
        if vocabulary_size <= 0:
            raise AssetPreparationError("prepared tokenizer has an empty vocabulary")
        stage.done(f"vocabulary {vocabulary_size:,}")

    # The most actionable fact this command has, and the one it never showed:
    # the requested revision need not be a SHA, and model and tokenizer can
    # resolve to different commits. A pin that has drifted is invisible
    # everywhere else until a run's numbers change.
    with rail.stage("revision") as stage:
        resolved_revision = _resolved_revision(model, tokenizer, config)
        stage.done(resolved_revision)
    manifest = AssetManifest(
        manifest_path=manifest_path,
        schema_version=ASSET_MANIFEST_SCHEMA_VERSION,
        model_identifier=config.model_identifier,
        tokenizer_identifier=config.tokenizer_identifier,
        requested_revision=config.revision,
        resolved_revision=resolved_revision,
        cache_path=str(cache_root),
        model_path=model_path.relative_to(cache_root).as_posix(),
        tokenizer_path=tokenizer_path.relative_to(cache_root).as_posix(),
        vocabulary_size=vocabulary_size,
        model_type=model_type,
        preparation_timestamp=datetime.now(timezone.utc).isoformat(),
        trust_remote_code=False,
    )
    _write_manifest(manifest)
    return manifest_path


def _parameter_count(model: Any) -> str | None:
    """ "494M parameters", or None if the model will not say.

    Guarded because a header must never be the thing that fails a preparation
    that otherwise succeeded.
    """

    try:
        total = sum(int(parameter.numel()) for parameter in model.parameters())
    except Exception:  # pragma: no cover - defensive around third-party models.
        return None
    for scale, unit in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if total >= scale:
            return f"{total / scale:.3g}{unit} parameters"
    return f"{total} parameters"


def _has_verified_matching_cache(
    manifest_path: Path,
    config: AssetPreparationConfig,
) -> bool:
    """Whether an already-prepared cache can stand in without re-fetching.

    Only trusted when ``config.revision`` is a full 40-character commit SHA.
    A floating ref (a branch or tag) can move on the Hub between runs, and
    checking that would mean the network round trip this shortcut exists to
    avoid -- so a non-SHA revision always falls through to full preparation,
    matching how ``_resolved_revision`` already treats the two cases
    differently. A SHA-pinned config cannot drift, so a manifest that already
    resolved to it is a complete answer without asking the Hub again.
    """

    if not _COMMIT_REVISION.fullmatch(config.revision):
        return False
    requested_commit = config.revision.lower()
    try:
        manifest = load_asset_manifest(manifest_path, preparation_config=config.config_path)
    except AssetManifestError:
        return False
    return (
        manifest.model_identifier == config.model_identifier
        and manifest.tokenizer_identifier == config.tokenizer_identifier
        and manifest.requested_revision.lower() == requested_commit
        and manifest.resolved_revision == requested_commit
        and Path(manifest.cache_path) == config.cache_dir.resolve()
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit LLM asset preparation command."""

    parser = argparse.ArgumentParser(
        description="Prepare a pinned Hugging Face causal-LM for offline use."
    )
    parser.add_argument("--config", required=True, help="Asset preparation YAML.")
    add_output_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the asset preparation CLI."""

    args = parse_args(argv)
    surface = surface_from_args(args)
    surface.rule("PREPARE LLM ASSETS")
    rail = surface.rail(STAGES)
    try:
        manifest_path = prepare_llm_assets(args.config, rail)
    except (AssetConfigError, AssetPreparationError) as exc:
        raise SystemExit(f"fedbrew prepare-llm: error: {exc}") from exc
    surface.final(f"Prepared Hugging Face causal-LM assets: {manifest_path}")


def _load_pretrained(
    factory: Any,
    *,
    kind: str,
    identifier: str,
    config: AssetPreparationConfig,
    kwargs: dict[str, Any],
) -> Any:
    try:
        with redirect_huggingface_hub_progress(print_download_progress):
            return factory.from_pretrained(identifier, **kwargs)
    except Exception as exc:
        # Not "unable to resolve requested revision": that diagnosis is only
        # one of the ways from_pretrained can fail, and it used to be printed
        # unconditionally. A tiny-gpt2 config with a revision confirmed valid
        # on the Hub still hit this branch because transformers refuses
        # torch.load on a pre-2.6 torch (CVE-2025-32434) -- the message named
        # the revision while the real cause was the local torch version.
        # str(exc) carries whatever transformers/huggingface_hub actually
        # raised; `from exc` keeps the original traceback for anyone
        # inspecting the chain.
        raise AssetPreparationError(
            f"unable to load {kind} {identifier!r} at revision {config.revision!r}: {exc}"
        ) from exc


def _resolved_revision(
    model: Any,
    tokenizer: Any,
    config: AssetPreparationConfig,
) -> str | None:
    requested_commit = (
        config.revision.lower() if _COMMIT_REVISION.fullmatch(config.revision) else None
    )
    candidates = {
        "model": getattr(getattr(model, "config", None), "_commit_hash", None),
        "tokenizer": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    }
    resolved: dict[str, str] = {}
    identifiers = {
        "model": config.model_identifier,
        "tokenizer": config.tokenizer_identifier,
    }
    for kind, candidate in candidates.items():
        if candidate is not None:
            if not isinstance(candidate, str) or not _COMMIT_REVISION.fullmatch(candidate):
                raise AssetPreparationError(
                    f"resolved {kind} revision is not a full commit SHA: {candidate!r}"
                )
            resolved[kind] = candidate.lower()
        elif requested_commit is not None:
            resolved[kind] = requested_commit
        elif not Path(identifiers[kind]).expanduser().exists():
            raise AssetPreparationError(
                f"unable to determine the resolved commit revision for {kind} "
                f"{identifiers[kind]!r} at requested revision "
                f"{config.revision!r}"
            )

    if requested_commit is not None:
        mismatched = {
            kind: revision for kind, revision in resolved.items() if revision != requested_commit
        }
        if mismatched:
            details = ", ".join(f"{kind}={revision}" for kind, revision in mismatched.items())
            raise AssetPreparationError(
                "prepared assets did not resolve to the requested commit "
                f"{requested_commit}: {details}"
            )

    unique_revisions = set(resolved.values())
    if len(unique_revisions) > 1:
        details = ", ".join(f"{kind}={revision}" for kind, revision in resolved.items())
        raise AssetPreparationError(
            "model and tokenizer resolved to different commits, which cannot be "
            f"represented by one asset revision: {details}"
        )
    return next(iter(unique_revisions), None)


def _write_manifest(manifest: AssetManifest) -> None:
    manifest.manifest_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
