import json
from pathlib import Path

import numpy as np
import soundfile as sf

from RSB.data.create_dataset import CleanDataset, NoiseDataset, create_dataset


def write_tone(path: Path, frequency: float, sample_rate: int = 16_000) -> None:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sf.write(path, 0.2 * np.sin(2 * np.pi * frequency * time), sample_rate)


def test_dereverberation_exports_acoustic_statistics(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean_source"
    noise_dir = tmp_path / "noise_source"
    output_dir = tmp_path / "output"
    (clean_dir / "train").mkdir(parents=True)
    noise_dir.mkdir()
    write_tone(clean_dir / "train" / "speech.wav", 220)
    write_tone(noise_dir / "noise.wav", 997)

    configuration = create_dataset(
        tasks=["dereverberation"],
        clean_dataset=CleanDataset.WSJ0,
        clean_inputs=[clean_dir],
        noise_dataset=NoiseDataset.CHIME,
        noise_inputs=[noise_dir],
        output_dir=output_dir,
        t60_range_s=(0.5, 0.5),
        seed=7,
    )

    assert configuration["summary"]["average_measured_t60_s"] is not None
    assert configuration["summary"]["average_direct_to_diffuse_ratio_db"] is not None
    assert (output_dir / "train" / "clean" / "000000_speech.wav").is_file()
    assert (output_dir / "train" / "noisy" / "000000_speech.wav").is_file()
    saved = json.loads((output_dir / "create_configuraton.json").read_text())
    assert saved["files"][0]["requested_t60_s"] == 0.5
    assert "requested_direct_to_diffuse_ratio_db" not in saved["files"][0]


def test_combined_task_records_noise_and_room_parameters(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean_source"
    noise_dir = tmp_path / "noise_source"
    (clean_dir / "train").mkdir(parents=True)
    noise_dir.mkdir()
    write_tone(clean_dir / "train" / "speech.wav", 220)
    write_tone(noise_dir / "noise.wav", 997)

    configuration = create_dataset(
        tasks=["derev+enh"],
        clean_dataset=CleanDataset.WSJ0,
        clean_inputs=[clean_dir],
        noise_dataset=NoiseDataset.CHIME,
        noise_inputs=[noise_dir],
        output_dir=tmp_path / "output",
        snr_range_db=(5.0, 5.0),
        t60_range_s=(0.5, 0.5),
        seed=7,
    )

    record = configuration["files"][0]
    assert record["requested_snr_db"] == 5.0
    assert record["requested_t60_s"] == 0.5
    assert record["source_noise"].endswith("noise.wav")


def test_wsj0_uses_fixed_source_partitions(tmp_path: Path) -> None:
    clean_dir = tmp_path / "wsj0"
    noise_dir = tmp_path / "chime"
    noise_dir.mkdir()
    write_tone(noise_dir / "noise.wav", 997)
    for directory_name, frequency in (("si_tr_s", 220), ("si_dt_05", 330), ("si_et_05", 440)):
        directory = clean_dir / directory_name
        directory.mkdir(parents=True)
        write_tone(directory / "speech.wav", frequency)

    configuration = create_dataset(
        tasks=["dereverberation"],
        clean_dataset=CleanDataset.WSJ0,
        clean_inputs=[clean_dir],
        noise_dataset=NoiseDataset.CHIME,
        noise_inputs=[noise_dir],
        output_dir=tmp_path / "output",
        t60_range_s=(0.5, 0.5),
        seed=7,
    )

    assert configuration["summary"]["split_counts"] == {"train": 1, "valid": 1, "test": 1}
    sources = {record["split"]: Path(record["source_clean"]).parent.name for record in configuration["files"]}
    assert sources == {"train": "si_tr_s", "valid": "si_et_05", "test": "si_dt_05"}
