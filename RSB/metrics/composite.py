"""Composite intrusive speech-quality metric."""

from __future__ import annotations

from typing import Any

import numpy as np

from RSB.metrics.base import AudioMetric
from RSB.metrics.composite_metric import eval_composite
from RSB.metrics.registry import MetricRegister


@MetricRegister.register("composite")
class CompositeMetric(AudioMetric):
    """CSIG, CBAK, COVL, LLR, and WSS speech-quality metrics."""

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate the composite speech-quality metrics."""
        del noise_wav, wav_path, kwargs
        result = eval_composite(ref_wav, deg_wav, sample_rate)
        return {
            "CSIG": float(result["csig"]),
            "CBAK": float(result["cbak"]),
            "COVL": float(result["covl"]),
            "LLR": float(result["llr"]),
            "WSS": float(result["wss_dist"]),
        }
