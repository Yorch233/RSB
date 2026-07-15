"""Download and normalize pretrained RSB checkpoints from Hugging Face Hub."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from RSB.utils.config import BaseConfiguer, migrate_config, read_yml

DEFAULT_GENERATIVE_MODEL_ID = "Yorch233/RSB"
DEFAULT_GENERATIVE_RUN_NAME = "Yorch233--RSB"
_REQUIRED_FILES = ("config.yml", "model.safetensors")


def migrate_generative_config(
    values: Mapping[str, Any],
    *,
    model_id: str,
    revision: str,
    run_path: Path,
) -> dict[str, Any]:
    """Convert a legacy Hub configuration into the current run format."""
    migrated = migrate_config(values)
    source_run_name = migrated.get("run_name")
    migrated.pop("datasets", None)
    migrated.pop("inherit", None)

    migrated.update(
        {
            "run_type": "generative",
            "run_name": DEFAULT_GENERATIVE_RUN_NAME,
            "source_run_name": source_run_name,
            "run_dir": str(run_path.parent),
            "run_path": str(run_path),
            "output_path": str(run_path),
            "hub_model_id": model_id,
            "hub_revision": revision,
        }
    )
    return migrated


def _materialized_root(cache_dir: str | Path | None) -> Path:
    hub_cache = Path(cache_dir).expanduser() if cache_dir is not None else Path(HF_HUB_CACHE)
    return hub_cache / "rsb-runs"


def _link_or_copy(source: Path, destination: Path) -> None:
    resolved_source = source.resolve(strict=True)
    if destination.is_file() and destination.stat().st_size == resolved_source.stat().st_size:
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(resolved_source, destination)
    except FileExistsError:
        return
    except OSError:
        shutil.copy2(resolved_source, destination)


def materialize_generative_run(
    model_id: str = DEFAULT_GENERATIVE_MODEL_ID,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> Path:
    """Download a Hub snapshot and expose it as a current-format local run."""
    try:
        downloaded = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=list(_REQUIRED_FILES),
        )
    except OSError as error:
        raise RuntimeError(f"Unable to download Hugging Face checkpoint {model_id!r}: {error}") from error
    snapshot = Path(downloaded).resolve()
    missing = [name for name in _REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise ValueError(f"Hub checkpoint {model_id!r} is missing required files: {missing}")

    resolved_revision = snapshot.name
    destination = _materialized_root(cache_dir) / model_id.replace("/", "--") / resolved_revision
    destination.mkdir(parents=True, exist_ok=True)
    _link_or_copy(snapshot / "model.safetensors", destination / "model.safetensors")
    migrated = migrate_generative_config(
        read_yml(snapshot / "config.yml"),
        model_id=model_id,
        revision=resolved_revision,
        run_path=destination,
    )
    BaseConfiguer.dump(migrated, destination / "config.yml")
    return destination
