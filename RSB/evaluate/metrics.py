# RSB/evaluate/metrics.py
from typing import Dict
import os

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz

from RSB.evaluate.composite_metric import eval_composite
from RSB.evaluate.DNSMOS.dnsmos_local import ComputeScore
from RSB.evaluate.registry import MetricRegister


class AudioMetric:
    """
    Base class for audio quality metrics calculation.
    Subclasses should implement the calculate method.
    """

    def __init__(self, **kwargs):
        """Initialize the audio metric."""
        pass

    def calculate(self,
                  ref_wav,
                  deg_wav,
                  noise_wav=None,
                  wav_path=None,
                  sample_rate=16000,
                  **kwargs) -> dict:
        """
        Calculate the metric value(s) between reference and degraded audio.

        Args:
            ref_wav (np.ndarray): Reference (clean) audio waveform.
            deg_wav (np.ndarray): Degraded (processed) audio waveform.
            noise_wav (np.ndarray, optional): Noise audio waveform. Defaults to None.
            wav_path (str, optional): Path to the audio file. Defaults to None.
            sample_rate (int): Sampling rate of the audio. Defaults to 16000.

        Returns:
            dict: Dictionary containing metric names and their values.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def compute(cls, *args, **kwargs) -> dict:
        """
        Class method to create an instance and compute the metric.

        Returns:
            dict: Dictionary containing metric results.
        """
        instance = cls()
        return instance.calculate(*args, **kwargs)


@MetricRegister.register('pesq')
class PESQMetric(AudioMetric):
    """
    Perceptual Evaluation of Speech Quality (PESQ) metric.
    Measures the perceived quality of speech.
    """

    def calculate(self, ref_wav, deg_wav, sample_rate=16000, **kwargs):
        """
        Calculate PESQ score.

        Args:
            ref_wav (np.ndarray): Reference audio.
            deg_wav (np.ndarray): Degraded audio.
            sample_rate (int): Audio sampling rate.

        Returns:
            dict: Dictionary with PESQ score.
        """
        psq_mode = "wb" if sample_rate == 16000 else "nb"  # Wideband or Narrowband
        return {'PESQ': pesq(sample_rate, ref_wav, deg_wav, psq_mode)}


@MetricRegister.register('estoi')
class ESTOIMetric(AudioMetric):
    """
    Extended Short-Time Objective Intelligibility (ESTOI) metric.
    Measures speech intelligibility.
    """

    def calculate(self, ref_wav, deg_wav, sample_rate=16000, **kwargs):
        """
        Calculate ESTOI score.

        Args:
            ref_wav (np.ndarray): Reference audio.
            deg_wav (np.ndarray): Degraded audio.
            sample_rate (int): Audio sampling rate.

        Returns:
            dict: Dictionary with ESTOI score.
        """
        return {"ESTOI": stoi(ref_wav, deg_wav, sample_rate, extended=True)}


@MetricRegister.register('distortion')
class CompositeMetric(AudioMetric):
    """
    Distortion metrics (LLR and WSS) from composite evaluation.
    """

    def calculate(self, ref_wav, deg_wav, sample_rate=16000, **kwargs):
        """
        Calculate distortion metrics.

        Args:
            ref_wav (np.ndarray): Reference audio.
            deg_wav (np.ndarray): Degraded audio.
            sample_rate (int): Audio sampling rate.

        Returns:
            dict: Dictionary with LLR and WSS distortion metrics.
        """
        result = eval_composite_distortion(ref_wav, deg_wav, sample_rate)
        return {"LLR": result['llr'], "WSS": result['wss_dist']}


@MetricRegister.register('composite')
class CompositeMetric(AudioMetric):
    """
    Composite speech quality metrics including CSIG, CBAK, COVL, LLR, and WSS.
    """

    def calculate(self, ref_wav, deg_wav, sample_rate=16000, **kwargs):
        """
        Calculate composite speech quality metrics.

        Args:
            ref_wav (np.ndarray): Reference audio.
            deg_wav (np.ndarray): Degraded audio.
            sample_rate (int): Audio sampling rate.

        Returns:
            dict: Dictionary with composite metrics (CSIG, CBAK, COVL, LLR, WSS).
        """
        result = eval_composite(ref_wav, deg_wav, sample_rate)
        return {
            "CSIG": result['csig'],  # Signal distortion
            "CBAK": result['cbak'],  # Background noise distortion
            "COVL": result['covl'],  # Overall quality
            "LLR": result['llr'],  # Log-likelihood ratio
            "WSS": result['wss_dist']  # Weighted spectral slope
        }


@MetricRegister.register("si_sdr")
class SiSdrMetric(AudioMetric):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric.
    Measures the quality of separated speech signals.
    """

    def calculate(self, ref_wav, deg_wav, **kwargs):
        """
        Calculate SI-SDR score.

        Args:
            ref_wav (np.ndarray): Reference audio.
            deg_wav (np.ndarray): Degraded audio.

        Returns:
            dict: Dictionary with SI-SDR score.
        """
        # Compute optimal scaling factor
        alpha = np.dot(deg_wav, ref_wav) / np.linalg.norm(ref_wav)**2
        # Calculate SI-SDR in dB
        sdr = 10 * np.log10(
            np.linalg.norm(alpha * ref_wav)**2 /
            np.linalg.norm(alpha * ref_wav - deg_wav)**2)
        return {'SI_SDR': sdr.astype(float)}


@MetricRegister.register("energy_ratios")
class EnergyRatiosMetric(AudioMetric):
    """
    Energy-based ratios including SI-SDR, SI-SIR, and SI-SAR.
    Used for evaluating source separation performance.
    """

    def calculate(self, ref_wav, deg_wav, noise_wav, **kwargs):
        """
        Calculate energy ratio metrics.

        Args:
            ref_wav (np.ndarray): Reference (target) audio.
            deg_wav (np.ndarray): Degraded (estimated) audio.
            noise_wav (np.ndarray): Noise (interference) audio.

        Returns:
            dict: Dictionary with SI-SDR, SI-SIR, and SI-SAR scores.
        """
        sdr, sir, sar = self.energy_ratios(deg_wav, ref_wav, noise_wav)
        return {
            'SI_SDR': sdr.astype(float),
            'SI_SIR': sir.astype(float),
            'SI_SAR': sar.astype(float)
        }

    def energy_ratios(self, s_hat, s, n, eps=1e-10):
        """
        Compute SI-SDR, SI-SIR, and SI-SAR energy ratios.

        Args:
            s_hat (np.ndarray): Estimated signal.
            s (np.ndarray): Target signal.
            n (np.ndarray): Noise/interference signal.
            eps (float): Small epsilon to avoid division by zero.

        Returns:
            tuple: SI-SDR, SI-SIR, SI-SAR values.
        """
        s_target, e_noise, e_art = self.si_sdr_components(s_hat, s, n)

        # SI-SDR: Signal to (noise + artifacts) ratio
        si_sdr = 10 * np.log10(eps + np.linalg.norm(s_target)**2 /
                               (eps + np.linalg.norm(e_noise + e_art)**2))

        # SI-SIR: Signal to noise ratio
        si_sir = 10 * np.log10(eps + np.linalg.norm(s_target)**2 /
                               (eps + np.linalg.norm(e_noise)**2))

        # SI-SAR: Signal + noise to artifacts ratio
        si_sar = 10 * np.log10(eps + np.linalg.norm(s_target + e_noise)**2 /
                               (eps + np.linalg.norm(e_art)**2))

        return si_sdr, si_sir, si_sar

    def si_sdr_components(self, s_hat, s, n, eps=1e-10):
        """
        Decompose the estimated signal into target, noise, and artifact components.

        Args:
            s_hat (np.ndarray): Estimated signal.
            s (np.ndarray): Target signal.
            n (np.ndarray): Noise signal.
            eps (float): Small epsilon to avoid division by zero.

        Returns:
            tuple: Target signal, noise component, artifact component.
        """
        # Target scaling factor
        alpha_s = np.dot(s_hat, s) / (eps + np.linalg.norm(s)**2)
        s_target = alpha_s * s

        # Noise scaling factor
        alpha_n = np.dot(s_hat, n) / (eps + np.linalg.norm(n)**2)
        e_noise = alpha_n * n

        # Artifacts (everything else)
        e_art = s_hat - s_target - e_noise

        return s_target, e_noise, e_art


@MetricRegister.register("dnsmos")
class DNSMOSMetric(AudioMetric):
    """
    DNSMOS: Microsoft's deep learning-based audio quality assessment.
    Provides MOS (Mean Opinion Score) predictions for speech quality.
    """

    def __init__(self, **kwargs):
        """
        Initialize DNSMOS with pre-trained models.
        """
        super().__init__(**kwargs)
        # Define paths to ONNX models
        p808_model_path = os.path.join('RSB', 'evaluate', 'DNSMOS',
                                       'DNSMOS', 'model_v8.onnx')
        primary_model_path = os.path.join('RSB', 'evaluate', 'DNSMOS',
                                          'DNSMOS', 'sig_bak_ovr.onnx')
        # Initialize the DNSMOS computation class
        self.compute = ComputeScore(primary_model_path, p808_model_path)

    def calculate(self, wav_path, sample_rate=16000, **kwargs):
        """
        Calculate DNSMOS scores for an audio file.

        Args:
            wav_path (str): Path to the audio file.
            sample_rate (int): Sampling rate (default: 16000).

        Returns:
            dict: Dictionary with DNSMOS scores (SIG, BAK, OVRL, P808).
        """
        result = self.compute(wav_path,
                              sampling_rate=sample_rate,
                              is_personalized_MOS=False)
        return {
            # 'DNSMOS_SIG_RAW': result['SIG_raw'],
            # 'DNSMOS_BAK_RAW': result['BAK_raw'],
            # 'DNSMOS_OVRL_RAW': result['OVRL_raw'],
            'DNSMOS_SIG': result['SIG'],  # Signal quality
            'DNSMOS_BAK': result['BAK'],  # Background quality
            'DNSMOS_OVRL': result['OVRL'],  # Overall quality
            'DNSMOS_P808': result['P808_MOS']  # P.808 MOS prediction
        }
