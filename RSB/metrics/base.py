"""Base protocol for registered speech-quality metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AudioMetric(ABC):
    """Base class for waveform-level audio metrics."""

    @abstractmethod
    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate metric values for one waveform pair."""

    @classmethod
    def compute(cls, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Create a metric instance and calculate one result."""
        return cls().calculate(*args, **kwargs)
