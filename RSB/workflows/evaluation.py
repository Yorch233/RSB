"""Run-linked speech-enhancement metric evaluation."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from RSB.metrics import MetricRegister
from RSB.workflows.artifacts import paired_wavs, read_json, resolve_dataset, resolve_run, result_directory, write_json

DEFAULT_METRICS = ("pesq", "estoi", "si_sdr")
SUPPORTED_METRICS = ("pesq", "estoi", "si_sdr", "composite", "dnsmos", "si_sir", "si_sar")
_CALCULATORS = {
    "pesq": "pesq",
    "estoi": "estoi",
    "si_sdr": "si_sdr",
    "composite": "composite",
    "dnsmos": "dnsmos",
    "si_sir": "energy_ratios",
    "si_sar": "energy_ratios",
}
_OUTPUTS = {
    "pesq": ("PESQ",),
    "estoi": ("ESTOI",),
    "si_sdr": ("SI_SDR",),
    "composite": ("CSIG", "CBAK", "COVL", "LLR", "WSS"),
    "dnsmos": ("DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_OVRL", "DNSMOS_P808"),
    "si_sir": ("SI_SIR",),
    "si_sar": ("SI_SAR",),
}


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def _evaluate_file(
    clean_path: Path,
    noisy_path: Path,
    enhanced_path: Path,
    calculators: dict[str, Any],
    output_names: set[str],
    sample_rate: int,
) -> dict[str, Any]:
    clean = _load_audio(clean_path, sample_rate)
    noisy = _load_audio(noisy_path, sample_rate)
    enhanced = _load_audio(enhanced_path, sample_rate)
    length = min(clean.size, noisy.size, enhanced.size)
    clean, noisy, enhanced = clean[:length], noisy[:length], enhanced[:length]
    values: dict[str, Any] = {"filename": clean_path.name}
    for calculator in calculators.values():
        calculated = calculator.calculate(
            ref_wav=clean,
            deg_wav=enhanced,
            noise_wav=noisy - clean,
            sample_rate=sample_rate,
            wav_path=str(enhanced_path),
        )
        values.update({key: float(value) for key, value in calculated.items() if key in output_names})
    return values


def _calculate_metrics(
    pairs: list[tuple[Path, Path]],
    output: Path,
    metadata: dict[str, Any],
    *,
    sample_rate: int,
    metrics: tuple[str, ...],
    max_workers: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    invalid = [metric for metric in metrics if metric not in SUPPORTED_METRICS]
    if invalid:
        raise ValueError(f"Unsupported metrics: {invalid}; choose from {SUPPORTED_METRICS}")
    selected_metrics = tuple(dict.fromkeys(metrics or DEFAULT_METRICS))
    csv_path = output / "metrics.csv"
    json_path = output / "metrics.json"
    if (csv_path.exists() or json_path.exists()) and not overwrite:
        if csv_path.is_file() and json_path.is_file():
            return csv_path, json_path
        raise ValueError(f"Partial metric artifacts exist in {output}; use --overwrite")

    calculator_names = tuple(dict.fromkeys(_CALCULATORS[metric] for metric in selected_metrics))
    calculator_types = MetricRegister.fetch(list(calculator_names))
    calculators = {name: calculator_types[name]() for name in calculator_names}
    output_names = {name for metric in selected_metrics for name in _OUTPUTS[metric]}
    workers = None if max_workers == 0 else max_workers
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _evaluate_file,
                clean,
                noisy,
                output / noisy.name,
                calculators,
                output_names,
                sample_rate,
            )
            for clean, noisy in pairs
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: str(item["filename"]))
    frame = pd.DataFrame(results)
    frame.to_csv(csv_path, index=False)
    summary = {
        column: {
            "mean": float(np.nanmean(frame[column].to_numpy(dtype=np.float64))),
            "std": float(np.nanstd(frame[column].to_numpy(dtype=np.float64))),
        }
        for column in frame.columns
        if column != "filename"
    }
    write_json(
        json_path,
        {
            "artifact_type": "metrics",
            **metadata,
            "sample_rate": sample_rate,
            "metrics": list(selected_metrics),
            "num_files": len(results),
            "summary": summary,
        },
    )
    return csv_path, json_path


def evaluate_results(
    run_reference: str | Path,
    variant: str,
    *,
    dataset_id: str | None = None,
    split: str = "test",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    max_workers: int = 0,
    overwrite: bool = False,
    result_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Evaluate a canonical run result directory and save per-file and summary metrics."""
    run = resolve_run(run_reference)
    selected_dataset, dataset_root = resolve_dataset(run.config, dataset_id)
    pairs = paired_wavs(dataset_root, split)
    output = result_dir.expanduser().resolve() if result_dir is not None else result_directory(run, variant)
    inference_manifest_path = output / "inference.json"
    if not inference_manifest_path.is_file():
        raise ValueError(f"Inference results are missing inference.json: {output}")
    inference_manifest = read_json(inference_manifest_path)
    expected_manifest = {
        "dataset_id": selected_dataset,
        "split": split,
        "run_name": run.name,
        "model_sha256": run.model_sha256,
    }
    mismatches = [key for key, value in expected_manifest.items() if inference_manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Inference manifest conflicts on {', '.join(mismatches)}")
    expected_files = [noisy.name for _, noisy in pairs]
    available = {path.name for path in output.glob("*.wav")}
    if set(expected_files) != available:
        raise ValueError(f"Inference results are incomplete or contain unexpected WAV files: {output}")

    sample_rate = int(inference_manifest.get("sample_rate", run.config.get("sample_rate", 16_000)))
    return _calculate_metrics(
        pairs,
        output,
        {
            "source_type": "run",
            "run_name": run.name,
            "run_id": run.config.get("run_id"),
            "model_sha256": run.model_sha256,
            "dataset_id": selected_dataset,
            "split": split,
            "variant": variant,
        },
        sample_rate=sample_rate,
        metrics=metrics,
        max_workers=max_workers,
        overwrite=overwrite,
    )


def evaluate_directory(
    directory: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    max_workers: int = 0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Evaluate an inference result directory using its provenance manifest."""
    output = directory.expanduser().resolve()
    manifest_path = output / "inference.json"
    if not manifest_path.is_file():
        raise ValueError(f"Result directory is missing inference.json: {output}")
    manifest = read_json(manifest_path)
    run_reference = manifest.get("run_path") or manifest.get("run_name")
    dataset_id = manifest.get("dataset_id")
    split = manifest.get("split")
    if not isinstance(run_reference, str) or not isinstance(dataset_id, str) or not isinstance(split, str):
        raise ValueError(f"Inference manifest lacks run, dataset, or split provenance: {manifest_path}")
    return evaluate_results(
        run_reference,
        output.name,
        dataset_id=dataset_id,
        split=split,
        metrics=metrics,
        max_workers=max_workers,
        overwrite=overwrite,
        result_dir=output,
    )


def evaluate_external(
    clean_dir: Path,
    noisy_dir: Path,
    enhanced_dir: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    sample_rate: int = 16_000,
    max_workers: int = 0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Evaluate third-party enhanced WAVs against explicit clean and noisy directories."""
    clean_dir = clean_dir.expanduser().resolve()
    noisy_dir = noisy_dir.expanduser().resolve()
    enhanced_dir = enhanced_dir.expanduser().resolve()
    clean = {path.name: path for path in clean_dir.glob("*.wav")}
    noisy = {path.name: path for path in noisy_dir.glob("*.wav")}
    enhanced = {path.name: path for path in enhanced_dir.glob("*.wav")}
    if not clean or clean.keys() != noisy.keys() or clean.keys() != enhanced.keys():
        raise ValueError("Clean, noisy, and enhanced directories must contain the same non-empty WAV filename set")
    pairs = [(clean[name], noisy[name]) for name in sorted(clean)]
    return _calculate_metrics(
        pairs,
        enhanced_dir,
        {
            "source_type": "external",
            "clean_dir": str(clean_dir),
            "noisy_dir": str(noisy_dir),
            "enhanced_dir": str(enhanced_dir),
        },
        sample_rate=sample_rate,
        metrics=metrics,
        max_workers=max_workers,
        overwrite=overwrite,
    )
