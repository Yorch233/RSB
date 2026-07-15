"""Complex-spectrogram transforms shared by training and inference."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def get_window(window_type: str, window_length: int) -> Tensor:
    """Create a Hann or square-root Hann window."""
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    if window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    raise NotImplementedError(f"Window type {window_type!r} is not implemented")


class STFTUtil:
    """Apply STFT transforms with the magnitude warping used by RSB."""

    n_fft: int | None = None
    num_frames: int | None = None
    hop_length: int | None = None
    spec_abs_exponent: float | None = None
    spec_factor: float | None = None
    window: Tensor | None = None
    windows: dict[torch.device, Tensor] = {}
    initialized = False

    @classmethod
    def initial(
        cls,
        n_fft: int = 510,
        num_frames: int = 256,
        hop_length: int = 128,
        spec_abs_exponent: float = 0.5,
        spec_factor: float = 0.33,
        window: str = "sqrthann",
    ) -> None:
        """Initialize transform parameters and reset the device-window cache."""
        parameters = (n_fft, num_frames, hop_length, spec_abs_exponent, spec_factor)
        current = (cls.n_fft, cls.num_frames, cls.hop_length, cls.spec_abs_exponent, cls.spec_factor)
        if cls.initialized and parameters == current:
            return
        cls.n_fft = n_fft
        cls.num_frames = num_frames
        cls.hop_length = hop_length
        cls.spec_abs_exponent = spec_abs_exponent
        cls.spec_factor = spec_factor
        cls.window = get_window(window, n_fft)
        cls.windows = {}
        cls.initialized = True

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls.initialized:
            cls.initial()

    @classmethod
    def _get_window(cls, tensor: Tensor) -> Tensor:
        """Return a cached transform window on the tensor's device."""
        cls._ensure_initialized()
        if cls.window is None:
            raise RuntimeError("STFT window was not initialized")
        if tensor.device not in cls.windows:
            cls.windows[tensor.device] = cls.window.to(tensor.device)
        return cls.windows[tensor.device]

    @classmethod
    def stft(cls, audio: Tensor, transform: bool = True) -> Tensor:
        """Convert audio to a complex spectrum."""
        cls._ensure_initialized()
        if cls.n_fft is None or cls.hop_length is None:
            raise RuntimeError("STFT parameters were not initialized")
        spectrum = torch.stft(
            audio,
            n_fft=cls.n_fft,
            hop_length=cls.hop_length,
            window=cls._get_window(audio),
            center=True,
            return_complex=True,
        )
        return cls.magnitude_warping(spectrum) if transform else spectrum

    @classmethod
    def istft(cls, spectrum: Tensor, transform: bool = True, length: int | None = None) -> Tensor:
        """Convert a complex spectrum back to audio."""
        cls._ensure_initialized()
        if cls.n_fft is None or cls.hop_length is None:
            raise RuntimeError("STFT parameters were not initialized")
        restored = cls.invert_magnitude_warping(spectrum) if transform else spectrum
        return torch.istft(
            restored,
            n_fft=cls.n_fft,
            hop_length=cls.hop_length,
            window=cls._get_window(restored),
            center=True,
            length=length,
        )

    @classmethod
    def magnitude_warping(cls, spectrum: Tensor) -> Tensor:
        """Compress spectrum magnitudes while retaining phase."""
        cls._ensure_initialized()
        if cls.spec_abs_exponent is None or cls.spec_factor is None:
            raise RuntimeError("Magnitude-warping parameters were not initialized")
        if cls.spec_abs_exponent != 1:
            spectrum = spectrum.abs().pow(cls.spec_abs_exponent) * torch.exp(1j * spectrum.angle())
        return spectrum * cls.spec_factor

    @classmethod
    def invert_magnitude_warping(cls, spectrum: Tensor) -> Tensor:
        """Undo spectrum magnitude compression."""
        cls._ensure_initialized()
        if cls.spec_abs_exponent is None or cls.spec_factor is None:
            raise RuntimeError("Magnitude-warping parameters were not initialized")
        spectrum = spectrum / cls.spec_factor
        if cls.spec_abs_exponent != 1:
            spectrum = spectrum.abs().pow(1 / cls.spec_abs_exponent) * torch.exp(1j * spectrum.angle())
        return spectrum

    @classmethod
    def to_stft(
        cls,
        audio: Tensor,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        """Normalize and transform audio, returning an inverse closure."""
        audio_length = audio.size(-1)
        audio = audio.reshape(1, -1)
        normalization = max(float(audio.abs().max()), torch.finfo(audio.dtype).eps)
        normalized = audio.to(device) / normalization
        spectrum = pad_spec(cls.stft(normalized).unsqueeze(0))

        def invert(value: Tensor) -> Tensor:
            restored = cls.istft(value.squeeze(), length=audio_length)
            return restored.squeeze().cpu() * normalization

        return spectrum, invert


def pad_spec(spectrum: Tensor) -> Tensor:
    """Pad the time axis to a multiple of 64."""
    frames = spectrum.size(3)
    padding = 64 - frames % 64 if frames % 64 else 0
    return torch.nn.ZeroPad2d((0, padding, 0, 0))(spectrum)
