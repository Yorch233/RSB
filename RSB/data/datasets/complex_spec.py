"""Paired clean, noisy, and posterior-mean spectrogram dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from RSB.data.audio_folder import AudioFolder
from RSB.data.spectral import STFTUtil

if TYPE_CHECKING:
    from RSB.utils.config import Config


class ComplexSpecDataset(Dataset[tuple[Tensor, ...]]):
    """Load aligned clean, noisy, and optional offline posterior-mean spectra."""

    def __init__(
        self,
        config: Config,
        dataset: str = "voicebank",
        subset: Literal["train", "valid", "test"] = "train",
        shuffle_spec: bool | None = None,
        normalize_audio: bool | None = None,
        return_raw: bool = False,
        return_spec: bool = True,
        posterior_mean_from: str | None = None,
        dummy: bool = False,
    ) -> None:
        """Initialize aligned audio folders and spectrum transforms."""
        if dataset not in config.datasets:
            raise ValueError(f"Dataset {dataset!r} is not configured")
        self.sample_rate = int(config.sample_rate)
        self.audio_length = int(config.audio_length)
        self.data_dir = config.datasets[dataset]
        self.subset = subset
        self.spatial_channels = int(config.spatial_channels)
        self.num_frames = int(config.num_frames)
        self.hop_length = int(config.hop_length)
        self.clean_files = AudioFolder(
            audio_path=f"{self.data_dir}/{subset}/clean",
            sample_rate=self.sample_rate,
        )
        self.noisy_files = AudioFolder(
            audio_path=f"{self.data_dir}/{subset}/noisy",
            sample_rate=self.sample_rate,
        )
        self.mean_files: AudioFolder | None = None
        if posterior_mean_from is not None:
            self.mean_files = AudioFolder(
                audio_path=f"{self.data_dir}/{subset}/mean/{posterior_mean_from}",
                sample_rate=self.sample_rate,
            )

        self.shuffle_spec = bool(shuffle_spec)
        self.normalize_audio = config.normalize_audio if normalize_audio is None else normalize_audio
        self.return_spec = return_spec
        self.return_raw = return_raw
        self.dummy = dummy
        STFTUtil.initial(
            n_fft=int(config.n_fft),
            num_frames=self.num_frames,
            hop_length=self.hop_length,
            spec_abs_exponent=float(config.spec_abs_exponent),
            spec_factor=float(config.spec_factor),
            window=config.window,
        )

    def __len__(self) -> int:
        """Return the number of noisy examples, or the dummy training size."""
        return 200 if self.dummy else len(self.noisy_files)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        """Load, align, crop, normalize, and transform one paired example."""
        clean = self.clean_files[index]
        noisy = self.noisy_files[index]
        posterior_mean = self.mean_files[index] if self.mean_files is not None else None
        tensors = [clean, noisy, *([posterior_mean] if posterior_mean is not None else [])]
        minimum_length = min(tensor.size(-1) for tensor in tensors)
        tensors = [tensor[..., :minimum_length] for tensor in tensors]
        clean, noisy = tensors[:2]
        posterior_mean = tensors[2] if len(tensors) == 3 else None

        if clean.ndim == 2 and self.spatial_channels == 1:
            clean, noisy = clean[0].unsqueeze(0), noisy[0].unsqueeze(0)
            if posterior_mean is not None:
                posterior_mean = posterior_mean[0].unsqueeze(0)
        if self.spatial_channels > clean.size(0):
            raise ValueError(
                f"Requested {self.spatial_channels} channels from audio containing {clean.size(0)} channels"
            )
        clean, noisy = clean[: self.spatial_channels], noisy[: self.spatial_channels]
        if posterior_mean is not None:
            posterior_mean = posterior_mean[: self.spatial_channels]
        if self.return_raw:
            return (clean, noisy, posterior_mean) if posterior_mean is not None else (clean, noisy)

        normalization = noisy.abs().max().clamp_min(torch.finfo(noisy.dtype).eps)
        target_length = self.audio_length if not self.return_spec else (self.num_frames - 1) * self.hop_length
        clean, noisy, posterior_mean = self._fit_length(clean, noisy, posterior_mean, target_length)
        if self.normalize_audio:
            clean, noisy = clean / normalization, noisy / normalization
            if posterior_mean is not None:
                posterior_mean = posterior_mean / normalization

        if not self.return_spec:
            return (clean, noisy, posterior_mean) if posterior_mean is not None else (clean, noisy)
        clean_spec, noisy_spec = STFTUtil.stft(clean), STFTUtil.stft(noisy)
        if posterior_mean is None:
            return clean_spec, noisy_spec
        return clean_spec, noisy_spec, STFTUtil.stft(posterior_mean)

    def _fit_length(
        self,
        clean: Tensor,
        noisy: Tensor,
        posterior_mean: Tensor | None,
        target_length: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        current_length = clean.size(-1)
        padding = max(target_length - current_length, 0)
        if padding:
            pad = (padding // 2, padding // 2 + padding % 2)
            clean, noisy = F.pad(clean, pad), F.pad(noisy, pad)
            if posterior_mean is not None:
                posterior_mean = F.pad(posterior_mean, pad)
            return clean, noisy, posterior_mean

        maximum_start = current_length - target_length
        start = int(np.random.uniform(0, maximum_start)) if self.shuffle_spec and maximum_start else maximum_start // 2
        stop = start + target_length
        clean, noisy = clean[..., start:stop], noisy[..., start:stop]
        if posterior_mean is not None:
            posterior_mean = posterior_mean[..., start:stop]
        return clean, noisy, posterior_mean
