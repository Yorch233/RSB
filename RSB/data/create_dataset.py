import json
import math
import shutil
from collections.abc import Iterable, Sequence
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from scipy.signal import fftconvolve

STORM_CREATE_DATA_URL = "https://github.com/sp-uhh/storm/blob/master/preprocessing/create_data.py"
SUPPORTED_AUDIO_SUFFIXES = {".flac", ".ogg", ".wav"}
SPLITS = ("train", "valid", "test")


class CleanDataset(StrEnum):
    """Supported clean-speech dataset labels."""

    VCTK = "vctk"
    WSJ0 = "wsj0"
    TIMIT = "timit"


class NoiseDataset(StrEnum):
    """Supported noise dataset labels."""

    NONE = "none"
    CHIME = "chime"
    QUT = "qut"
    WHAM = "wham"


def discover_audio_files(inputs: Sequence[Path]) -> list[Path]:
    """Return sorted, unique audio files found in files or directories."""
    files: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
            files.add(path)
        elif path.is_dir():
            files.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
            )
    return sorted(files)


def infer_split(path: Path) -> str | None:
    """Infer a fixed dataset split from a path component."""
    return next((part for part in reversed(path.parts) if part in SPLITS), None)


def discover_presplit_audio_files(inputs: Sequence[Path]) -> dict[str, list[Path]]:
    """Discover clean files while preserving existing train/valid/test splits."""
    split_files: dict[str, set[Path]] = {split: set() for split in SPLITS}
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if path.is_file():
            split = infer_split(path)
            if split is None:
                raise ValueError(f"Clean audio path has no train/valid/test component: {path}")
            split_files[split].add(path)
            continue

        discovered_from_root = False
        for split in SPLITS:
            split_dir = path / split
            clean_split_dir = split_dir / "clean"
            source_dir = clean_split_dir if clean_split_dir.is_dir() else split_dir
            if source_dir.is_dir():
                split_files[split].update(discover_audio_files([source_dir]))
                discovered_from_root = True
        if discovered_from_root:
            continue

        inferred_split = infer_split(path)
        if inferred_split is not None:
            split_files[inferred_split].update(discover_audio_files([path]))
            continue

        for audio_path in discover_audio_files([path]):
            split = infer_split(audio_path)
            if split is None:
                raise ValueError(f"Clean audio path has no train/valid/test component: {audio_path}")
            split_files[split].add(audio_path)

    result = {split: sorted(split_files[split]) for split in SPLITS}
    if not any(result.values()):
        raise ValueError("No supported clean audio files were found in fixed train/valid/test splits.")
    return result


def find_named_directory(root: Path, name: str) -> Path | None:
    """Find the shallowest directory below ROOT with a case-insensitive name."""
    if root.is_dir() and root.name.lower() == name.lower():
        return root
    matches = sorted(
        (path for path in root.rglob("*") if path.is_dir() and path.name.lower() == name.lower()),
        key=lambda path: (len(path.parts), str(path)),
    )
    return matches[0] if matches else None


def discover_wsj0_splits(root: Path) -> dict[str, list[Path]] | None:
    """Discover StoRM's fixed WSJ0 train, validation, and test directories."""
    directory_names = {"train": "si_tr_s", "valid": "si_et_05", "test": "si_dt_05"}
    directories = {split: find_named_directory(root, name) for split, name in directory_names.items()}
    if not all(directories.values()):
        return None
    return {
        split: discover_audio_files([directory]) for split, directory in directories.items() if directory is not None
    }


def discover_vctk_splits(root: Path) -> dict[str, list[Path]] | None:
    """Discover StoRM's fixed VCTK speaker-index partitions."""
    speech_root = find_named_directory(root, "wav48")
    if speech_root is None:
        speech_root = root if any(path.is_dir() and path.name.startswith("p") for path in root.iterdir()) else None
    if speech_root is None:
        return None
    speakers = sorted(path for path in speech_root.iterdir() if path.is_dir() and path.name not in {"p280", "p315"})
    ranges = {"train": (0, 99), "valid": (97, 99), "test": (99, 107)}
    return {split: discover_audio_files(speakers[start:stop]) for split, (start, stop) in ranges.items()}


def discover_timit_splits(root: Path) -> dict[str, list[Path]] | None:
    """Discover StoRM's fixed TIMIT dialect-region partitions."""
    train_root = find_named_directory(root, "train")
    test_root = find_named_directory(root, "test")
    if train_root is None or test_root is None:
        return None

    def dialect_regions(parent: Path, start: int, stop: int) -> list[Path]:
        directories: list[Path] = []
        for index in range(start, stop):
            directory = find_named_directory(parent, f"dr{index}")
            if directory is not None:
                directories.append(directory)
        return directories

    return {
        "train": discover_audio_files(dialect_regions(train_root, 1, 7)),
        "valid": discover_audio_files(dialect_regions(train_root, 7, 8)),
        "test": discover_audio_files(dialect_regions(test_root, 1, 8)),
    }


def discover_clean_dataset_splits(dataset: CleanDataset, inputs: Sequence[Path]) -> dict[str, list[Path]]:
    """Resolve fixed clean-speech partitions for a supported dataset.

    The WSJ0, VCTK, and TIMIT rules follow StoRM's fixed partition logic.
    Thanks to the StoRM authors; source:
    https://github.com/sp-uhh/storm/blob/master/preprocessing/create_data.py
    """
    roots = [path.expanduser().resolve() for path in inputs]
    missing_roots = [root for root in roots if not root.exists()]
    if missing_roots:
        raise ValueError(f"Clean dataset path does not exist: {missing_roots[0]}")
    if all(any((root / split).is_dir() for root in roots) for split in SPLITS):
        return discover_presplit_audio_files(roots)

    discoverers = {
        CleanDataset.WSJ0: discover_wsj0_splits,
        CleanDataset.VCTK: discover_vctk_splits,
        CleanDataset.TIMIT: discover_timit_splits,
    }
    merged: dict[str, set[Path]] = {split: set() for split in SPLITS}
    for root in roots:
        discovered = discoverers[dataset](root)
        if discovered is None:
            return discover_presplit_audio_files(roots)
        for split in SPLITS:
            merged[split].update(discovered[split])
    result = {split: sorted(merged[split]) for split in SPLITS}
    if not any(result.values()):
        raise ValueError(f"No audio files matched the fixed {dataset.value} partition rules.")
    return result


def discover_noise_pools(inputs: Sequence[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    """Discover split-specific noise pools or one shared fallback pool."""
    shared_pool = discover_audio_files(inputs)
    split_pools: dict[str, set[Path]] = {split: set() for split in SPLITS}
    for input_path in inputs:
        path = input_path.expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Noise dataset path does not exist: {path}")
        if path.is_file():
            split = infer_split(path)
            if split is not None:
                split_pools[split].add(path)
            continue
        aliases = {"train": ("train", "tr"), "valid": ("valid", "cv"), "test": ("test", "tt")}
        for split, directory_names in aliases.items():
            for directory_name in directory_names:
                split_dir = path / directory_name
                noise_split_dir = split_dir / "noise"
                source_dir = noise_split_dir if noise_split_dir.is_dir() else split_dir
                if source_dir.is_dir():
                    split_pools[split].update(discover_audio_files([source_dir]))
                    break
        inferred_split = infer_split(path)
        if inferred_split is not None:
            split_pools[inferred_split].update(discover_audio_files([path]))
    return {split: sorted(split_pools[split]) for split in SPLITS}, shared_pool


def load_mono_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load an audio file as mono floating-point samples at SAMPLE_RATE."""
    audio, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1)
    if source_rate != sample_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=sample_rate)
    if mono.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return np.asarray(mono, dtype=np.float32)


def normalize_tasks(tasks: Sequence[str]) -> set[str]:
    """Normalize enhancement and dereverberation task aliases."""
    aliases = {
        "derev": "dereverberation",
        "dereverberation": "dereverberation",
        "enh": "enhancement",
        "enhancement": "enhancement",
        "noise": "enhancement",
    }
    normalized: set[str] = set()
    for task_group in tasks:
        for task in task_group.lower().replace("+", ",").split(","):
            value = task.strip()
            if value not in aliases:
                choices = ", ".join(sorted(aliases))
                raise ValueError(f"Unsupported task '{value}'. Choose from: {choices}")
            normalized.add(aliases[value])
    if not normalized:
        raise ValueError("At least one task is required.")
    return normalized


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale and add noise at a requested SNR.

    This function is inspired by the signal/noise power scaling in StoRM's
    ``preprocessing/create_data.py``. Thanks to the StoRM authors; source:
    https://github.com/sp-uhh/storm/blob/master/preprocessing/create_data.py
    """
    clean_power = float(np.mean(np.square(clean, dtype=np.float64)))
    noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    if clean_power <= 0 or noise_power <= 0:
        raise ValueError("Clean speech and noise must both contain non-zero energy.")
    target_noise_power = clean_power * 10 ** (-snr_db / 10)
    scaled_noise = noise * math.sqrt(target_noise_power / noise_power)
    return clean + scaled_noise, scaled_noise


def create_synthetic_rir(
    sample_rate: int,
    t60_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a direct-path plus exponentially decaying diffuse RIR.

    StoRM's room-simulation workflow inspired this corruption stage. Thanks to
    the StoRM authors; source:
    https://github.com/sp-uhh/storm/blob/master/preprocessing/create_data.py
    """
    rir_length = max(int(math.ceil(t60_s * sample_rate)), int(0.05 * sample_rate))
    direct_window = max(1, int(round(0.0025 * sample_rate)))
    times = np.arange(rir_length, dtype=np.float64) / sample_rate
    diffuse = rng.standard_normal(rir_length) * np.exp(-math.log(1000) * times / t60_s)
    diffuse[:direct_window] = 0
    diffuse_energy = float(np.sum(np.square(diffuse)))
    target_diffuse_energy = max(0.1, t60_s / 0.4) * float(rng.lognormal(mean=0.0, sigma=0.35))
    if diffuse_energy > 0:
        diffuse *= math.sqrt(target_diffuse_energy / diffuse_energy)
    rir = diffuse
    rir[0] = 1.0
    return np.asarray(rir, dtype=np.float32)


def measure_t60(rir: np.ndarray, sample_rate: int) -> float | None:
    """Estimate T60 from a room impulse response using Schroeder decay."""
    energy_decay = np.cumsum(np.square(rir, dtype=np.float64)[::-1])[::-1]
    if energy_decay[0] <= 0:
        return None
    decay_db = 10 * np.log10(np.maximum(energy_decay / energy_decay[0], np.finfo(float).tiny))
    indices = np.flatnonzero((decay_db <= -5) & (decay_db >= -35))
    if indices.size < 2:
        return None
    slope, _ = np.polyfit(indices / sample_rate, decay_db[indices], 1)
    if slope >= 0:
        return None
    return float(-60 / slope)


def measure_direct_to_diffuse_ratio(rir: np.ndarray, sample_rate: int) -> float | None:
    """Measure direct-to-diffuse energy ratio using a 2.5 ms direct window."""
    direct_window = max(1, int(round(0.0025 * sample_rate)))
    direct_energy = float(np.sum(np.square(rir[:direct_window], dtype=np.float64)))
    diffuse_energy = float(np.sum(np.square(rir[direct_window:], dtype=np.float64)))
    if direct_energy <= 0 or diffuse_energy <= 0:
        return None
    return float(10 * np.log10(direct_energy / diffuse_energy))


def scale_pair_to_prevent_clipping(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a shared gain when either member of an audio pair would clip."""
    peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))))
    if peak <= 0.99:
        return clean, noisy
    gain = 0.99 / peak
    return clean * gain, noisy * gain


def measured_snr(clean: np.ndarray, noisy: np.ndarray) -> float | None:
    """Measure signal-to-degradation ratio between aligned audio arrays."""
    signal_energy = float(np.sum(np.square(clean, dtype=np.float64)))
    error_energy = float(np.sum(np.square(noisy - clean, dtype=np.float64)))
    if signal_energy <= 0 or error_energy <= 0:
        return None
    return float(10 * np.log10(signal_energy / error_energy))


def si_sdr(reference: np.ndarray, degraded: np.ndarray) -> float | None:
    """Calculate scale-invariant signal-to-distortion ratio."""
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= 0:
        return None
    target = reference * (float(np.dot(degraded, reference)) / reference_energy)
    residual = degraded - target
    residual_energy = float(np.dot(residual, residual))
    if residual_energy <= 0:
        return None
    return float(10 * np.log10(float(np.dot(target, target)) / residual_energy))


def calculate_pair_metrics(clean: np.ndarray, noisy: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    """Calculate PESQ, ESTOI, and SI-SDR for one aligned test pair."""
    metrics: dict[str, float | None] = {"pesq": None, "estoi": None, "si_sdr_db": si_sdr(clean, noisy)}
    try:
        pesq_rate = sample_rate if sample_rate in {8_000, 16_000} else 16_000
        pesq_clean = clean
        pesq_noisy = noisy
        if sample_rate != pesq_rate:
            pesq_clean = librosa.resample(clean, orig_sr=sample_rate, target_sr=pesq_rate)
            pesq_noisy = librosa.resample(noisy, orig_sr=sample_rate, target_sr=pesq_rate)
        mode = "wb" if pesq_rate == 16_000 else "nb"
        metrics["pesq"] = float(pesq(pesq_rate, pesq_clean, pesq_noisy, mode))
    except (ValueError, RuntimeError):
        pass
    with suppress(ValueError, RuntimeError):
        metrics["estoi"] = float(stoi(clean, noisy, sample_rate, extended=True))
    return metrics


def mean_optional(values: Iterable[float | None]) -> float | None:
    """Return the finite mean of optional values."""
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else None


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create an empty output directory without silently deleting data."""
    resolved = output_dir.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"Output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {resolved}. Pass --overwrite to replace it.")
        if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
            raise ValueError(f"Refusing to remove unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (resolved / split / "clean").mkdir(parents=True, exist_ok=True)
        (resolved / split / "noisy").mkdir(parents=True, exist_ok=True)


def select_noise(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Tile or crop noise to exactly LENGTH samples."""
    if noise.size < length:
        repeats = math.ceil(length / noise.size)
        return np.tile(noise, repeats)[:length]
    start = int(rng.integers(0, noise.size - length + 1))
    return noise[start : start + length]


def create_dataset(
    *,
    tasks: Sequence[str],
    clean_dataset: CleanDataset,
    clean_inputs: Sequence[Path],
    noise_dataset: NoiseDataset,
    noise_inputs: Sequence[Path],
    output_dir: Path,
    sample_rate: int = 16_000,
    snr_range_db: tuple[float, float] = (-6.0, 14.0),
    t60_range_s: tuple[float, float] = (0.4, 1.0),
    seed: int = 100,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create paired speech data and export synthesis statistics.

    The split/corrupt/export workflow is inspired by StoRM's dataset creation
    script. Thanks to the StoRM authors; source:
    https://github.com/sp-uhh/storm/blob/master/preprocessing/create_data.py
    """
    normalized_tasks = normalize_tasks(tasks)
    splits = discover_clean_dataset_splits(clean_dataset, clean_inputs)
    noise_pools, shared_noise_pool = discover_noise_pools(noise_inputs)
    if "enhancement" in normalized_tasks and noise_dataset is NoiseDataset.NONE:
        raise ValueError("The enhancement task cannot use the 'none' noise dataset.")
    if noise_dataset is not NoiseDataset.NONE and not shared_noise_pool:
        raise ValueError("No supported noise audio files were found.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    for name, value_range in {
        "SNR": snr_range_db,
        "T60": t60_range_s,
    }.items():
        if value_range[0] > value_range[1]:
            raise ValueError(f"Invalid {name} range: minimum exceeds maximum.")

    prepare_output_directory(output_dir, overwrite)
    output_dir = output_dir.expanduser().resolve()
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    test_metrics: list[dict[str, float | None]] = []

    for split, split_clean_files in splits.items():
        for index, clean_path in enumerate(split_clean_files):
            clean = load_mono_audio(clean_path, sample_rate)
            noisy = clean.copy()
            record: dict[str, Any] = {
                "split": split,
                "source_clean": str(clean_path),
                "source_noise": None,
                "requested_snr_db": None,
                "measured_snr_db": None,
                "requested_t60_s": None,
                "measured_t60_s": None,
                "measured_direct_to_diffuse_ratio_db": None,
            }

            if "dereverberation" in normalized_tasks:
                requested_t60 = float(rng.uniform(*t60_range_s))
                rir = create_synthetic_rir(sample_rate, requested_t60, rng)
                noisy = fftconvolve(noisy, rir, mode="full")[: clean.size].astype(np.float32)
                record.update(
                    {
                        "requested_t60_s": requested_t60,
                        "measured_t60_s": measure_t60(rir, sample_rate),
                        "measured_direct_to_diffuse_ratio_db": measure_direct_to_diffuse_ratio(rir, sample_rate),
                    }
                )

            if "enhancement" in normalized_tasks:
                noise_pool = noise_pools[split] or shared_noise_pool
                noise_path = noise_pool[int(rng.integers(0, len(noise_pool)))]
                noise = select_noise(load_mono_audio(noise_path, sample_rate), clean.size, rng)
                requested_snr = float(rng.uniform(*snr_range_db))
                _, scaled_noise = mix_at_snr(clean, noise, requested_snr)
                noisy = noisy + scaled_noise
                record["source_noise"] = str(noise_path)
                record["requested_snr_db"] = requested_snr

            clean, noisy = scale_pair_to_prevent_clipping(clean, noisy)
            record["measured_snr_db"] = measured_snr(clean, noisy)
            filename = f"{index:06d}_{clean_path.stem}.wav"
            sf.write(output_dir / split / "clean" / filename, clean, sample_rate)
            sf.write(output_dir / split / "noisy" / filename, noisy, sample_rate)
            record["output_file"] = filename
            records.append(record)
            if split == "test":
                test_metrics.append(calculate_pair_metrics(clean, noisy, sample_rate))

    configuration: dict[str, Any] = {
        "schema_version": 1,
        "attribution": {
            "project": "StoRM",
            "source": STORM_CREATE_DATA_URL,
            "note": "Dataset corruption and export workflow inspired by StoRM's MIT-licensed create_data.py.",
        },
        "export": {
            "output_dir": str(output_dir),
            "clean_dataset": {"type": clean_dataset.value, "paths": [str(path) for path in clean_inputs]},
            "noise_dataset": {"type": noise_dataset.value, "paths": [str(path) for path in noise_inputs]},
            "tasks": sorted(normalized_tasks),
            "sample_rate": sample_rate,
            "seed": seed,
            "split_policy": "preserve train/valid/test components from clean input paths",
            "snr_range_db": list(snr_range_db),
            "t60_range_s": list(t60_range_s),
        },
        "summary": {
            "num_files": len(records),
            "split_counts": {split: len(splits[split]) for split in SPLITS},
            "average_measured_snr_db": mean_optional(record["measured_snr_db"] for record in records),
            "average_direct_to_diffuse_ratio_db": mean_optional(
                record["measured_direct_to_diffuse_ratio_db"] for record in records
            ),
            "average_measured_t60_s": mean_optional(record["measured_t60_s"] for record in records),
            "test_metrics": {
                "num_pairs": len(test_metrics),
                "pesq": mean_optional(metric["pesq"] for metric in test_metrics),
                "estoi": mean_optional(metric["estoi"] for metric in test_metrics),
                "si_sdr_db": mean_optional(metric["si_sdr_db"] for metric in test_metrics),
            },
        },
        "files": records,
    }
    with (output_dir / "create_configuraton.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, ensure_ascii=False, allow_nan=False)
    return configuration
