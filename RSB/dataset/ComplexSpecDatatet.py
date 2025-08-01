# RSB/dataset/ComplexSpecDatatet.py
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from RSB.backbone.registry import BackboneRegister
from RSB.dataset.AudioFolder import AudioFolder


def get_window(window_type, window_length):
    """Generate window functions for STFT
    
    Args:
        window_type (str): Window type, supports 'sqrthann' and 'hann'
        window_length (int): Length of the window
    
    Returns:
        torch.Tensor: Generated window tensor
    """
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(
            f"Window type {window_type} not implemented!")


class STFTUtil:
    """Utility class for STFT operations with magnitude warping"""

    n_fft = None
    num_frames = None
    hop_length = None
    spec_abs_exponent = None
    spec_factor = None
    window = None
    windows = None  # cached windows for different devices
    initialized = False

    @classmethod
    def initial(cls,
                n_fft=510,
                num_frames=256,
                hop_length=128,
                spec_abs_exponent=0.5,
                spec_factor=0.33,
                window="sqrthann"):
        """Initialize STFT parameters"""
        if cls.initialized:
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
    def _get_window(cls, x):
        """Get window for specific device"""
        device = x.device
        if device not in cls.windows:
            cls.windows[device] = cls.window.to(device)
        return cls.windows[device]

    @classmethod
    def stft(cls, x, transform=True):
        """Compute STFT with magnitude warping
        
        Args:
            x (torch.Tensor): Input audio tensor
            transform (bool): Apply magnitude warping if True
            
        Returns:
            torch.Tensor: STFT result
        """
        if cls.num_frames is None:
            cls.initial()
        window = cls._get_window(x)
        X = torch.stft(x,
                       n_fft=cls.n_fft,
                       hop_length=cls.hop_length,
                       window=window,
                       center=True,
                       return_complex=True)
        if transform:
            X = cls.magnitude_warping(X)
        return X

    @classmethod
    def istft(cls, X, transform=True, length=None):
        """Inverse STFT with inverse magnitude warping
        
        Args:
            X (torch.Tensor): STFT tensor
            transform (bool): Apply inverse warping if True
            length (int): Output audio length
            
        Returns:
            torch.Tensor: Reconstructed audio
        """
        if cls.num_frames is None:
            cls.initial()
        window = cls._get_window(X)
        if transform:
            X = cls.invert_magnitude_warping(X)
        return torch.istft(X,
                           n_fft=cls.n_fft,
                           hop_length=cls.hop_length,
                           window=window,
                           center=True,
                           length=length)

    @classmethod
    def magnitude_warping(cls, spec):
        """Apply magnitude warping to STFT spectrum"""
        if cls.spec_abs_exponent != 1:
            e = cls.spec_abs_exponent
            spec = spec.abs()**e * torch.exp(1j * spec.angle())
        return spec * cls.spec_factor

    @classmethod
    def invert_magnitude_warping(cls, spec):
        """Inverse magnitude warping for STFT spectrum"""
        spec = spec / cls.spec_factor
        if cls.spec_abs_exponent != 1:
            e = cls.spec_abs_exponent
            spec = spec.abs()**(1 / e) * torch.exp(1j * spec.angle())
        return spec

    @classmethod
    def to_stft(cls, audio, device="cpu"):
        """Convert audio to padded STFT format with inversion function
        
        Returns tuple (stft_tensor, inversion_function)
        """
        audio_length = audio.size(-1)
        audio = audio.view(1, -1)
        normfac = audio.abs().max().item()
        audio = audio.to(device)
        norm_audio = audio / normfac
        x = cls.stft(norm_audio)
        x = pad_spec(x.unsqueeze(0))

        def invert(x_):
            x_ = cls.istft(x_.squeeze(), length=audio_length)
            return x_.squeeze().cpu() * normfac

        return x, invert


def pad_spec(Y):
    """Pad spectrogram to make time dimension divisible by 64"""
    T = Y.size(3)
    num_pad = 64 - T % 64 if T % 64 != 0 else 0
    return torch.nn.ZeroPad2d((0, num_pad, 0, 0))(Y)


class ComplexSpecDataset(Dataset):

    def __init__(self,
                 config,
                 dataset='voicebank',
                 subset='train',
                 shuffle_spec=None,
                 normalize_audio=None,
                 return_raw=False,
                 return_spec=True,
                 load_posterior_mean=False,
                 dummy=False):
        self.sample_rate = config.sample_rate
        self.audio_length = config.audio_length
        assert subset in ['train', 'valid', 'test']

        assert dataset in config.datasets.keys(
        ), f'Dataset {dataset} is not supported yet.'
        self.data_dir = config.datasets[dataset]
        self.subset = subset
        self.spatial_channels = config.spatial_channels
        self.num_frames = config.num_frames
        self.hop_length = config.hop_length

        self.clean_files = AudioFolder(audio_path=os.path.join(
            self.data_dir, subset, 'clean'),
                                       sample_rate=self.sample_rate)
        self.noisy_files = AudioFolder(audio_path=os.path.join(
            self.data_dir, subset, 'noisy'),
                                       sample_rate=self.sample_rate)

        self.load_posterior_mean = load_posterior_mean
        if load_posterior_mean:
            assert os.path.exists(
                os.path.join(self.data_dir, subset, 'mean')
            ), f"Fail to load posterior mean from local files. Dictionary  {os.path.join(self.data_dir, subset, 'mean')} doesn't exisit"
            self.mean_files = AudioFolder(audio_path=os.path.join(
                self.data_dir, subset, 'mean'),
                                          sample_rate=self.sample_rate)

        self.shuffle_spec = shuffle_spec
        self.normalize_audio = config.normalize_audio if normalize_audio is None else normalize_audio
        self.return_spec = return_spec
        self.return_raw = return_raw
        self.dummy = dummy

        STFTUtil.initial(n_fft=config.n_fft,
                         num_frames=config.num_frames,
                         hop_length=config.hop_length,
                         spec_abs_exponent=config.spec_abs_exponent,
                         spec_factor=config.spec_factor,
                         window=config.window)

    def stft(self, x):
        return STFTUtil.stft(x)

    def istft(self, x, length=None):
        return STFTUtil.istft(x, length)

    def __getitem__(self, i):
        x = self.clean_files[i]
        y = self.noisy_files[i]
        if self.load_posterior_mean:
            x_star = self.mean_files[i]

        min_len = min(x.size(-1), y.size(-1))
        x, y = x[..., :min_len], y[..., :min_len]
        if self.load_posterior_mean:
            x_star = x_star[..., :min_len]

        if x.ndimension() == 2 and self.spatial_channels == 1:
            x, y = x[0].unsqueeze(0), y[0].unsqueeze(0)  # Select first channel
            if self.load_posterior_mean:
                x_star = x_star[0].unsqueeze(0)

        # Select channels
        assert self.spatial_channels <= x.size(
            0
        ), f"You asked too many channels ({self.spatial_channels}) for the given dataset ({x.size(0)})"
        x, y = x[:self.spatial_channels], y[:self.spatial_channels]
        if self.load_posterior_mean:
            x_star = x_star[:self.spatial_channels]

        if self.return_raw:
            return x, y

        normfac = y.abs().max()
        target_len = self.audio_length if not self.return_spec else (
            self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            x = x[..., start:start + target_len]
            y = y[..., start:start + target_len]
            if self.load_posterior_mean:
                x_star = x_star[..., start:start + target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
            y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
            if self.load_posterior_mean:
                x_star = F.pad(x_star, (pad // 2, pad // 2 + (pad % 2)),
                               mode='constant')

        if self.normalize_audio:
            # normalize both based on noisy speech, to ensure same clean signal power in x and y.
            x = x / normfac
            y = y / normfac
            if self.load_posterior_mean:
                x_star = x_star / normfac

        if self.return_spec:
            X, Y = self.stft(x), self.stft(y)
            if self.load_posterior_mean:
                X_star = self.stft(x_star)
                return X, Y, X_star
            return X, Y

        if self.load_posterior_mean:
            return x, y, x_star
        return x, y

    def __len__(self):
        if self.dummy:
            return 200
        return len(self.noisy_files)
