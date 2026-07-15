"""Run, dataset, result-directory, and provenance artifact helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from RSB.utils.config import Config, read_config_from_yaml, read_yml
from RSB.utils.paths import PROJECT_ROOT, RESULTS_DIR, RUNS_DIR, USER_CONFIG_PATH
from RSB.workflows.hub import DEFAULT_GENERATIVE_MODEL_ID, materialize_generative_run

RunType = Literal["predictive", "generative"]
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class RunArtifact:
    """Resolved trained-model run and its persisted configuration."""

    path: Path
    config: Config
    run_type: RunType
    model_path: Path
    model_sha256: str

    @property
    def name(self) -> str:
        """Return the persisted run name, falling back to the directory name."""
        return str(self.config.get("run_name") or self.path.name)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _local_settings() -> dict[str, Any]:
    if not USER_CONFIG_PATH.is_file():
        return {}
    return dict(read_yml(USER_CONFIG_PATH))


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_run_type(config: Config) -> RunType:
    """Return an explicit run type or infer it for backward-compatible runs."""
    run_type = config.get("run_type")
    if run_type in {"predictive", "generative"}:
        return run_type
    if config.get("predictive_method") is not None:
        return "predictive"
    if config.get("training_method") is not None:
        return "generative"
    raise ValueError("Run config does not identify a predictive or generative model")


def resolve_run(run: str | Path | None, expected_type: RunType | None = None) -> RunArtifact:
    """Resolve a local run or materialize the default generative Hub checkpoint."""
    if run is None:
        if expected_type != "generative":
            raise ValueError("A local run name or directory is required")
        candidate = materialize_generative_run()
    elif str(run) == DEFAULT_GENERATIVE_MODEL_ID:
        candidate = materialize_generative_run()
    else:
        candidate = Path(run).expanduser()
        if not candidate.is_dir():
            settings = _local_settings()
            run_root = _project_path(settings.get("run_dir", RUNS_DIR))
            candidate = run_root / str(run)
    path = candidate.resolve()
    if not path.is_dir():
        raise ValueError(f"Run directory does not exist: {path}")
    config_path = path / "config.yml"
    model_path = path / "model.safetensors"
    if not config_path.is_file():
        raise ValueError(f"Run is missing config.yml: {path}")
    if not model_path.is_file():
        raise ValueError(f"Run is missing model.safetensors: {path}")
    config = read_config_from_yaml(config_path)
    run_type = infer_run_type(config)
    if expected_type is not None and run_type != expected_type:
        raise ValueError(f"Expected a {expected_type} run, but {path.name!r} is {run_type}")
    return RunArtifact(path, config, run_type, model_path, file_sha256(model_path))


def resolve_dataset(run_config: Config, dataset_id: str | None = None) -> tuple[str, Path]:
    """Resolve a registered dataset, preferring the current project registry over its run snapshot."""
    settings = _local_settings()
    current = settings.get("datasets", {})
    snapshot = run_config.get("datasets", {})
    current = current if isinstance(current, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    selected = dataset_id or settings.get("dataset") or run_config.get("dataset")
    datasets = current if isinstance(selected, str) and selected in current else snapshot
    if not isinstance(selected, str) or selected not in datasets:
        available = ", ".join(dict.fromkeys([*current, *snapshot])) or "none"
        raise ValueError(f"Dataset ID {selected!r} is not registered; available IDs: {available}")
    return selected, _project_path(str(datasets[selected]))


def resolve_results_root(run_config: Config) -> Path:
    """Resolve the configured project results directory."""
    settings = _local_settings()
    return _project_path(settings.get("results_dir", run_config.get("results_dir", RESULTS_DIR)))


def result_directory(run: RunArtifact, variant: str) -> Path:
    """Return the canonical result directory for a run and inference variant."""
    return resolve_results_root(run.config) / run.name / variant


def paired_wavs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Return exactly aligned clean/noisy WAV pairs for a dataset split."""
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}; choose from {SPLITS}")
    clean_dir = dataset_root / split / "clean"
    noisy_dir = dataset_root / split / "noisy"
    clean = {path.name: path for path in clean_dir.glob("*.wav")}
    noisy = {path.name: path for path in noisy_dir.glob("*.wav")}
    if not clean or not noisy:
        raise ValueError(f"Dataset split has no paired WAV files: {dataset_root / split}")
    if clean.keys() != noisy.keys():
        missing_noisy = sorted(clean.keys() - noisy.keys())
        missing_clean = sorted(noisy.keys() - clean.keys())
        raise ValueError(f"Unpaired WAV files; missing noisy={missing_noisy}, missing clean={missing_clean}")
    return [(clean[name], noisy[name]) for name in sorted(clean)]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON mapping from disk."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def prepare_artifact_directory(output: Path, signature: dict[str, Any], *, overwrite: bool) -> set[str]:
    """Validate a resumable artifact directory and return already generated WAV names."""
    manifest_path = output / "inference.json"
    if output.exists() and overwrite:
        for path in output.glob("*.wav"):
            path.unlink()
        for path in (manifest_path, output / "metrics.csv", output / "metrics.json"):
            path.unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.glob("*.wav")}
    if manifest_path.is_file():
        current = read_json(manifest_path)
        mismatches = [key for key, value in signature.items() if current.get(key) != value]
        if mismatches:
            names = ", ".join(mismatches)
            raise ValueError(f"Existing inference manifest conflicts on {names}; use --overwrite")
    elif existing:
        raise ValueError(f"Result directory contains WAV files without inference.json: {output}")
    return existing


def finish_manifest(output: Path, signature: dict[str, Any], expected_files: list[str]) -> Path:
    """Write a completed inference manifest and verify every expected result exists."""
    missing = [name for name in expected_files if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Inference did not produce {len(missing)} expected files: {missing[:5]}")
    manifest = {
        **signature,
        "generated_at": datetime.now(UTC).isoformat(),
        "num_files": len(expected_files),
        "files": expected_files,
    }
    path = output / "inference.json"
    write_json(path, manifest)
    return path
