"""Offline posterior-mean generation and training-input validation."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torchaudio

from RSB.data import STFTUtil
from RSB.utils.config import Config
from RSB.workflows.artifacts import (
    SPLITS,
    paired_wavs,
    read_json,
    resolve_dataset,
    resolve_run,
    result_directory,
    write_json,
)
from RSB.workflows.inference import (
    _audio_iterator,
    _initialize_stft,
    _predictive_function,
    load_predictive_model,
    resolve_device,
)


def _prepare_mean_directory(output: Path, signature: dict[str, Any], *, overwrite: bool) -> set[str]:
    manifest_path = output / "manifest.json"
    if output.exists() and overwrite:
        for path in output.glob("*.wav"):
            path.unlink()
        manifest_path.unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.glob("*.wav")}
    if manifest_path.is_file():
        current = read_json(manifest_path)
        mismatches = [key for key, value in signature.items() if current.get(key) != value]
        if mismatches:
            raise ValueError(f"Existing posterior-mean manifest conflicts on {', '.join(mismatches)}; use --overwrite")
    elif existing:
        raise ValueError(f"Posterior-mean directory contains WAV files without manifest.json: {output}")
    return existing


def _matching_test_results(run_name: str, model_sha256: str, dataset_id: str, output: Path) -> Path | None:
    manifest_path = output / "inference.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    expected = {
        "artifact_type": "predictive_inference",
        "run_name": run_name,
        "model_sha256": model_sha256,
        "dataset_id": dataset_id,
        "split": "test",
    }
    return output if all(manifest.get(key) == value for key, value in expected.items()) else None


def generate_posterior_means(
    run_reference: str | Path,
    *,
    dataset_id: str | None = None,
    splits: tuple[str, ...] = SPLITS,
    device: str = "auto",
    num_workers: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> list[Path]:
    """Generate predictive posterior means inside every requested registered dataset split."""
    run = resolve_run(run_reference, "predictive")
    selected_dataset, dataset_root = resolve_dataset(run.config, dataset_id)
    invalid = [split for split in splits if split not in SPLITS]
    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}; choose from {SPLITS}")
    requested = tuple(dict.fromkeys(splits))
    source = str(run.config.get("predictive_method", "NCSN++M"))
    target_device = resolve_device(device)
    sample_rate = int(run.config.get("sample_rate", 16_000))
    model = None
    predict = None
    manifests: list[Path] = []
    reusable_test = _matching_test_results(
        run.name,
        run.model_sha256,
        selected_dataset,
        result_directory(run, "predictive"),
    )

    for split in requested:
        pairs = paired_wavs(dataset_root, split)
        output = dataset_root / split / "mean" / source
        signature = {
            "artifact_type": "posterior_mean",
            "dataset_id": selected_dataset,
            "split": split,
            "source": source,
            "predictive_run_name": run.name,
            "predictive_run_id": run.config.get("run_id"),
            "model_sha256": run.model_sha256,
            "sample_rate": sample_rate,
        }
        existing = _prepare_mean_directory(output, signature, overwrite=overwrite)
        expected = [noisy.name for _, noisy in pairs]
        pending = [noisy for _, noisy in pairs if noisy.name not in existing]
        if split == "test" and reusable_test is not None:
            still_pending = []
            for noisy in pending:
                reusable = reusable_test / noisy.name
                if reusable.is_file():
                    shutil.copy2(reusable, output / noisy.name)
                else:
                    still_pending.append(noisy)
            pending = still_pending
        if pending:
            if model is None:
                _initialize_stft(run)
                model = load_predictive_model(run, target_device)
                predict = _predictive_function(model, target_device)
            assert predict is not None
            for name, audio in _audio_iterator(
                pending,
                sample_rate=sample_rate,
                num_workers=num_workers,
                progress=progress,
                description=f"Posterior means ({split})",
            ):
                spectrum, invert = STFTUtil.to_stft(audio, device=target_device)
                enhanced = invert(predict(spectrum))
                torchaudio.save(output / name, enhanced.float().reshape(1, -1), sample_rate)
        missing = [name for name in expected if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"Posterior-mean generation missed {len(missing)} files: {missing[:5]}")
        manifest_path = output / "manifest.json"
        write_json(
            manifest_path,
            {
                **signature,
                "generated_at": datetime.now(UTC).isoformat(),
                "num_files": len(expected),
                "files": expected,
            },
        )
        manifests.append(manifest_path)
    return manifests


def validate_posterior_means(config: Config) -> dict[str, Any]:
    """Validate all split posterior means before generative training starts."""
    if config.training_method == "none":
        return {}
    _, dataset_root = resolve_dataset(config, config.dataset)
    source = str(config.get("posterior_mean_from", "NCSN++M"))
    source_path = Path(source)
    if not source or source in {".", ".."} or source_path.name != source:
        raise ValueError("--posterior-mean-from must name one directory directly below each split's mean directory")
    splits: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        clean_dir = dataset_root / split / "clean"
        output = dataset_root / split / "mean" / source_path
        if not clean_dir.is_dir():
            raise ValueError(f"Missing clean-audio directory for {split}: {clean_dir}")
        if not output.is_dir():
            raise ValueError(f"Missing posterior-mean directory for {split}: {output}")
        clean_files = {path.name for path in clean_dir.glob("*.wav")}
        mean_files = {path.name for path in output.glob("*.wav")}
        if clean_files != mean_files:
            missing = sorted(clean_files - mean_files)
            unexpected = sorted(mean_files - clean_files)
            raise ValueError(
                f"Posterior means do not match clean audio for {split}: "
                f"clean={len(clean_files)}, mean={len(mean_files)}, "
                f"missing={missing}, unexpected={unexpected}"
            )
        splits[split] = {"directory": str(output), "num_files": len(mean_files)}
    return {"source": source, "splits": splits}
