"""Energy-decomposition metrics for enhanced speech."""

from __future__ import annotations

from typing import Any

import numpy as np

from RSB.metrics.base import AudioMetric
from RSB.metrics.registry import MetricRegister


@MetricRegister.register("energy_ratios")
class EnergyRatiosMetric(AudioMetric):
    """SI-SDR, SI-SIR, and SI-SAR from target/noise decomposition."""

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        noise_wav: np.ndarray | None = None,
        wav_path: str | None = None,
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate the three scale-invariant energy ratios."""
        del wav_path, sample_rate, kwargs
        if noise_wav is None:
            raise ValueError("noise_wav is required for energy-ratio metrics")
        sdr, sir, sar = self.energy_ratios(deg_wav, ref_wav, noise_wav)
        return {"SI_SDR": sdr, "SI_SIR": sir, "SI_SAR": sar}

    @staticmethod
    def energy_ratios(
        estimate: np.ndarray,
        target: np.ndarray,
        noise: np.ndarray,
        eps: float = 1e-10,
    ) -> tuple[float, float, float]:
        """Calculate SI-SDR, SI-SIR, and SI-SAR."""
        target_component, noise_component, artifact_component = EnergyRatiosMetric.si_sdr_components(
            estimate,
            target,
            noise,
            eps,
        )
        si_sdr = 10 * np.log10(
            eps
            + np.linalg.norm(target_component) ** 2 / (eps + np.linalg.norm(noise_component + artifact_component) ** 2)
        )
        si_sir = 10 * np.log10(
            eps + np.linalg.norm(target_component) ** 2 / (eps + np.linalg.norm(noise_component) ** 2)
        )
        si_sar = 10 * np.log10(
            eps
            + np.linalg.norm(target_component + noise_component) ** 2 / (eps + np.linalg.norm(artifact_component) ** 2)
        )
        return float(si_sdr), float(si_sir), float(si_sar)

    @staticmethod
    def si_sdr_components(
        estimate: np.ndarray,
        target: np.ndarray,
        noise: np.ndarray,
        eps: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose an estimate into target, noise, and artifact components."""
        target_component = np.dot(estimate, target) / (eps + np.linalg.norm(target) ** 2) * target
        noise_component = np.dot(estimate, noise) / (eps + np.linalg.norm(noise) ** 2) * noise
        artifact_component = estimate - target_component - noise_component
        return target_component, noise_component, artifact_component
