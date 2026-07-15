import os
from glob import glob

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioFolder(Dataset):
    """Audio dataset loading WAV files or preprocessed .pt files"""

    def __init__(self, audio_path, sample_rate=16000, return_path=False, reverse=False):
        assert os.path.exists(audio_path), f"Path not found: {audio_path}"

        self.reverse = reverse
        self.return_path = return_path

        self.audios = []
        if os.path.isdir(audio_path):
            audio_paths = sorted(glob(f"{audio_path}/*.wav"))
            self.sample_rate = sample_rate
            for path in audio_paths:
                audio, source_sample_rate = torchaudio.load(path)
                if source_sample_rate != self.sample_rate:
                    audio = torchaudio.transforms.Resample(source_sample_rate, self.sample_rate)(audio)
                item = (audio, os.path.basename(path)[:-4]) if self.return_path else audio
                self.audios.append(item)
        elif audio_path.endswith(".pt"):
            data = torch.load(audio_path, weights_only=True)
            self.sample_rate = data["sample_rate"]
            assert self.sample_rate == sample_rate, "Sample rate mismatch"
            self.audios = list(data["audio"])
        else:
            raise NotImplementedError("Unsupported file type")

        if self.reverse:
            self.audios.reverse()

    def __len__(self):
        """Get dataset size"""
        return len(self.audios)

    def __getitem__(self, index):
        """Get audio data by index"""
        return self.audios[index]

    def get_data_loader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, reverse=None):
        """Create dataloader with optional reverse option"""
        if reverse is not None:
            self.reverse = reverse
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
        )
