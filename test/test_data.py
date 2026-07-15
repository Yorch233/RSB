import os
from pathlib import Path

import pytest
import torch
import torchaudio

from RSB.data import AudioFolder, STFTUtil


def test_audio_folder_loads_tensor_archive(tmp_path: Path) -> None:
    archive = tmp_path / "audio.pt"
    expected = [torch.arange(8, dtype=torch.float32), torch.arange(4, dtype=torch.float32)]
    torch.save({"sample_rate": 16_000, "audio": expected}, archive)

    dataset = AudioFolder(str(archive), sample_rate=16_000)

    assert len(dataset) == 2
    assert torch.equal(dataset[0], expected[0])


def test_stft_round_trip() -> None:
    STFTUtil.initialized = False
    STFTUtil.initial(n_fft=510, hop_length=128)
    audio = torch.randn(1, 16_000)

    reconstructed = STFTUtil.istft(STFTUtil.stft(audio), length=audio.shape[-1])

    assert reconstructed.shape == audio.shape
    assert torch.allclose(reconstructed, audio, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif("RSB_TEST_DATASET" not in os.environ, reason="read-only integration dataset not configured")
def test_remote_dataset_is_readable() -> None:
    dataset_root = Path(os.environ["RSB_TEST_DATASET"])
    noisy_dir = dataset_root / "test" / "noisy"
    audio_path = next(noisy_dir.glob("*.wav"))

    audio, sample_rate = torchaudio.load(audio_path)

    assert audio.numel() > 0
    assert sample_rate > 0
