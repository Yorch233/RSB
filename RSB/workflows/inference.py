"""Predictive and generative test-set inference workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import torch
import torchaudio
from safetensors.torch import load_model
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from RSB.data import STFTUtil
from RSB.modeling_rsb import RSB
from RSB.workflows.artifacts import (
    RunArtifact,
    finish_manifest,
    paired_wavs,
    prepare_artifact_directory,
    resolve_dataset,
    resolve_run,
    result_directory,
)

DeviceName = str | torch.device


class _InferenceAudioDataset(Dataset[tuple[str, Tensor]]):
    """Lazily load and resample named noisy WAV files."""

    def __init__(self, paths: list[Path], sample_rate: int) -> None:
        self.paths = paths
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        path = self.paths[index]
        audio, source_rate = torchaudio.load(path)
        if source_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, self.sample_rate)
        return path.name, audio


def resolve_device(value: DeviceName = "auto") -> torch.device:
    """Resolve an automatic, CUDA-index, or explicit torch device selection."""
    if isinstance(value, torch.device):
        return value
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    return device


def _initialize_stft(run: RunArtifact) -> None:
    config = run.config
    STFTUtil.initial(
        n_fft=int(config.get("n_fft", 510)),
        num_frames=int(config.get("num_frames", 256)),
        hop_length=int(config.get("hop_length", 128)),
        spec_abs_exponent=float(config.get("spec_abs_exponent", 0.5)),
        spec_factor=float(config.get("spec_factor", 0.33)),
        window=str(config.get("window", "sqrthann")),
    )


def load_predictive_model(run: RunArtifact, device: DeviceName = "auto") -> nn.Module:
    """Instantiate a predictive backbone and load its best safetensors weights."""
    if run.run_type != "predictive":
        raise ValueError(f"Run {run.name!r} is not predictive")
    from RSB.backbone import BackboneRegister

    model = BackboneRegister.fetch(run.config.predictive_backbone)(discriminative=True)
    target = resolve_device(device)
    load_model(model, run.model_path, device=str(target))
    return model.to(target).eval()


def _audio_iterator(
    paths: list[Path],
    *,
    sample_rate: int,
    num_workers: int,
    progress: bool,
    description: str,
) -> Iterator[tuple[str, Tensor]]:
    loader = DataLoader(
        _InferenceAudioDataset(paths, sample_rate),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    items: Any = tqdm(loader, desc=description, total=len(paths)) if progress else loader
    yield from items


def _predictive_function(model: nn.Module, device: torch.device) -> Callable[[Tensor], Tensor]:
    @torch.no_grad()
    def predict(observation: Tensor) -> Tensor:
        return model(observation.to(device))

    return predict


def _set_file_seed(seed: int, filename: str) -> None:
    digest = hashlib.sha256(filename.encode(), usedforsecurity=False).digest()
    file_seed = (seed + int.from_bytes(digest[:4], "big")) % (2**31)
    torch.manual_seed(file_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(file_seed)


def run_predictive_inference(
    run_reference: str | Path,
    *,
    dataset_id: str | None = None,
    split: str = "test",
    device: DeviceName = "auto",
    num_workers: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Enhance a registered dataset split with a predictive model run."""
    run = resolve_run(run_reference, "predictive")
    selected_dataset, dataset_root = resolve_dataset(run.config, dataset_id)
    pairs = paired_wavs(dataset_root, split)
    target_device = resolve_device(device)
    sample_rate = int(run.config.get("sample_rate", 16_000))
    output = result_directory(run, "predictive")
    signature = {
        "artifact_type": "predictive_inference",
        "dataset_id": selected_dataset,
        "split": split,
        "run_name": run.name,
        "run_path": str(run.path),
        "run_id": run.config.get("run_id"),
        "model_sha256": run.model_sha256,
        "sample_rate": sample_rate,
        "device": str(target_device),
        "method": run.config.get("predictive_method", "NCSN++M"),
    }
    existing = prepare_artifact_directory(output, signature, overwrite=overwrite)
    expected = [noisy.name for _, noisy in pairs]
    pending = [noisy for _, noisy in pairs if noisy.name not in existing]
    if pending:
        _initialize_stft(run)
        model = load_predictive_model(run, target_device)
        predict = _predictive_function(model, target_device)
        for name, audio in _audio_iterator(
            pending,
            sample_rate=sample_rate,
            num_workers=num_workers,
            progress=progress,
            description="Predictive inference",
        ):
            spectrum, invert = STFTUtil.to_stft(audio, device=target_device)
            enhanced = invert(predict(spectrum))
            torchaudio.save(output / name, enhanced.float().reshape(1, -1), sample_rate)
    finish_manifest(output, signature, expected)
    return output


def run_generative_inference(
    run_reference: str | Path | None = None,
    *,
    dataset_id: str | None = None,
    split: str = "test",
    sampler: Literal["SDE", "ODE"] = "SDE",
    num_steps: int = 5,
    skip_type: Literal["time_uniform", "time_quadratic"] = "time_uniform",
    seed: int = 10,
    device: DeviceName = "auto",
    num_workers: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Enhance a registered dataset split with a trained generative RSB run."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    run = resolve_run(run_reference, "generative")
    selected_dataset, dataset_root = resolve_dataset(run.config, dataset_id)
    pairs = paired_wavs(dataset_root, split)
    target_device = resolve_device(device)
    sample_rate = int(run.config.get("sample_rate", 16_000))
    variant = f"{sampler}_N={num_steps}"
    output = result_directory(run, variant)
    signature = {
        "artifact_type": "generative_inference",
        "dataset_id": selected_dataset,
        "split": split,
        "run_name": run.name,
        "run_path": str(run.path),
        "run_id": run.config.get("run_id"),
        "model_sha256": run.model_sha256,
        "sample_rate": sample_rate,
        "device": str(target_device),
        "sampler": sampler,
        "num_steps": num_steps,
        "skip_type": skip_type,
        "seed": seed,
    }
    if run.config.get("hub_model_id") is not None:
        signature["hub_model_id"] = run.config.hub_model_id
        signature["hub_revision"] = run.config.get("hub_revision")
    existing = prepare_artifact_directory(output, signature, overwrite=overwrite)
    expected = [noisy.name for _, noisy in pairs]
    pending = [noisy for _, noisy in pairs if noisy.name not in existing]
    if pending:
        _initialize_stft(run)
        bridge = RSB.from_pretrained(run.path, map_location=target_device).to(target_device).eval()
        for name, audio in _audio_iterator(
            pending,
            sample_rate=sample_rate,
            num_workers=num_workers,
            progress=progress,
            description="RSB inference",
        ):
            _set_file_seed(seed, name)
            enhanced, _, _ = bridge.enhance(
                audio,
                num_steps=num_steps,
                solver=sampler,
                skip_type=skip_type,
            )
            torchaudio.save(output / name, enhanced.float().reshape(1, -1), sample_rate)
    finish_manifest(output, signature, expected)
    return output
