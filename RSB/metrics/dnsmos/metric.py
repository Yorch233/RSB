"""DNSMOS metric adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from RSB.metrics.base import AudioMetric
from RSB.metrics.dnsmos.dnsmos_local import ComputeScore
from RSB.metrics.registry import MetricRegister
from RSB.utils.paths import DNSMOS_MODEL_DIR


@MetricRegister.register("dnsmos")
class DNSMOSMetric(AudioMetric):
    """Microsoft DNSMOS speech, background, and overall quality scores."""

    def __init__(self) -> None:
        """Load the bundled DNSMOS ONNX models."""
        self._compute_score = ComputeScore(
            str(DNSMOS_MODEL_DIR / "sig_bak_ovr.onnx"),
            str(DNSMOS_MODEL_DIR / "model_v8.onnx"),
        )

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate DNSMOS for an enhanced WAV file."""
        del ref_wav, deg_wav, noise_wav, kwargs
        if wav_path is None:
            raise ValueError("wav_path is required for DNSMOS")
        result = self._compute_score(wav_path, sampling_rate=sample_rate, is_personalized_MOS=False)
        return {
            "DNSMOS_SIG": float(result["SIG"]),
            "DNSMOS_BAK": float(result["BAK"]),
            "DNSMOS_OVRL": float(result["OVRL"]),
            "DNSMOS_P808": float(result["P808_MOS"]),
        }
