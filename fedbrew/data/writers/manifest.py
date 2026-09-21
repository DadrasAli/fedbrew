"""Manifest persistence helpers for generated federated data."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def save_manifest(output_dir: str | Path, manifest_dict: Mapping[str, Any]) -> Path:
    """Save manifest.json under an output directory."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(dict(manifest_dict), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def save_clients_jsonl(
    output_dir: str | Path,
    clients_metadata: Sequence[Mapping[str, Any]],
) -> Path:
    """Save one JSON client metadata record per line."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clients_path = output_path / "clients.jsonl"
    with clients_path.open("w", encoding="utf-8") as file:
        for record in clients_metadata:
            file.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return clients_path
