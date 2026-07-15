"""Intrusive metrics that compare enhanced and reference speech."""

from __future__ import annotations

from typing import Any

import numpy as np
from pesq import pesq
from pystoi import stoi

from RSB.metrics.base import AudioMetric
from RSB.metrics.composite_metric import eval_composite
from RSB.metrics.registry import MetricRegister


@MetricRegister.register("pesq")
class PESQMetric(AudioMetric):
    """Perceptual Evaluation of Speech Quality."""

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate wideband or narrowband PESQ."""
        del noise_wav, wav_path, kwargs
        mode = "wb" if sample_rate == 16_000 else "nb"
        return {"PESQ": float(pesq(sample_rate, ref_wav, deg_wav, mode))}


@MetricRegister.register("estoi")
class ESTOIMetric(AudioMetric):
    """Extended Short-Time Objective Intelligibility."""

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate ESTOI."""
        del noise_wav, wav_path, kwargs
        return {"ESTOI": float(stoi(ref_wav, deg_wav, sample_rate, extended=True))}


@MetricRegister.register("distortion")
class DistortionMetric(AudioMetric):
    """LLR and weighted-spectral-slope distortion metrics."""

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate LLR and WSS through the composite metric implementation."""
        del noise_wav, wav_path, kwargs
        result = eval_composite(ref_wav, deg_wav, sample_rate)
        return {"LLR": float(result["llr"]), "WSS": float(result["wss_dist"])}


@MetricRegister.register("si_sdr")
class SiSdrMetric(AudioMetric):
    """Scale-Invariant Signal-to-Distortion Ratio."""

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate SI-SDR in decibels."""
        del noise_wav, wav_path, sample_rate, kwargs
        alpha = np.dot(deg_wav, ref_wav) / np.linalg.norm(ref_wav) ** 2
        target = alpha * ref_wav
        value = 10 * np.log10(np.linalg.norm(target) ** 2 / np.linalg.norm(target - deg_wav) ** 2)
        return {"SI_SDR": float(value)}
