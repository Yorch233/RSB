"""Sampled speech-quality evaluation during generative training."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchaudio
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from tqdm import tqdm

from RSB.metrics import MetricRegister
from RSB.utils.config import Config
from RSB.workflows.artifacts import paired_wavs, resolve_dataset

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

    from RSB.modeling_rsb import RSB


def _load_audio(path: Path, sample_rate: int) -> Tensor:
    """Load one waveform and resample it when necessary."""
    audio, source_rate = torchaudio.load(path)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def evaluate_sampled_split(
    model: RSB,
    config: Config,
    split: str,
    *,
    max_samples: int,
    num_steps: int,
    progress: bool = True,
) -> dict[str, float]:
    """Sample a deterministic prefix of a split and average PESQ and SI-SDR."""
    _, dataset_root = resolve_dataset(config, config.dataset)
    pairs = paired_wavs(dataset_root, split)[:max_samples]
    metrics = MetricRegister.fetch(["pesq", "si_sdr"])
    totals: dict[str, float] = {}
    items: Iterable[tuple[Path, Path]] = (
        tqdm(pairs, desc=f"Sampling {split} metrics", leave=False) if progress else pairs
    )

    for clean_path, noisy_path in items:
        clean = _load_audio(clean_path, int(config.sample_rate))
        noisy = _load_audio(noisy_path, int(config.sample_rate))
        enhanced, _, _ = model.enhance(
            noisy,
            num_steps=num_steps,
            solver="SDE",
        )
        reference = clean.detach().float().cpu().reshape(-1).numpy()
        estimate = enhanced.detach().float().cpu().reshape(-1).numpy()
        minimum = min(reference.size, estimate.size)
        reference, estimate = reference[:minimum], estimate[:minimum]
        for metric_name, metric_type in metrics.items():
            try:
                values = metric_type.compute(ref_wav=reference, deg_wav=estimate, sample_rate=int(config.sample_rate))
            except Exception as error:  # Metric libraries can reject individual malformed utterances.
                warnings.warn(f"Failed to compute {metric_name} for {noisy_path.name}: {error}", stacklevel=2)
                continue
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value)

    count = len(pairs)
    return {name: value / count for name, value in totals.items()} if count else {}


class GenerativeSampleMetrics(Callback):
    """Log sampled valid/test metrics for generative model selection."""

    metric_names = (
        "valid/PESQ_per_epoch",
        "valid/SI_SDR_per_epoch",
        "test/PESQ_per_epoch",
        "test/SI_SDR_per_epoch",
    )

    def __init__(
        self,
        config: Config,
        *,
        valid_samples: int = 50,
        test_samples: int = 5,
        num_steps: int = 20,
    ) -> None:
        """Configure sampled evaluation sizes and solver steps."""
        super().__init__()
        if valid_samples < 1 or test_samples < 1 or num_steps < 1:
            raise ValueError("Sample counts and num_steps must be positive")
        self.config = config
        self.valid_samples = valid_samples
        self.test_samples = test_samples
        self.num_steps = num_steps

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate on rank zero, broadcast scores, and expose them to later callbacks."""
        if trainer.sanity_checking:
            return
        scores = torch.zeros(len(self.metric_names), device=pl_module.device, dtype=torch.float64)
        if trainer.is_global_zero:
            model = getattr(pl_module, "model", pl_module)
            valid = evaluate_sampled_split(
                model,
                self.config,
                "valid",
                max_samples=self.valid_samples,
                num_steps=self.num_steps,
            )
            test = evaluate_sampled_split(
                model,
                self.config,
                "test",
                max_samples=self.test_samples,
                num_steps=self.num_steps,
            )
            scores = torch.tensor(
                [
                    valid.get("PESQ", 0.0),
                    valid.get("SI_SDR", -100.0),
                    test.get("PESQ", 0.0),
                    test.get("SI_SDR", -100.0),
                ],
                device=pl_module.device,
                dtype=torch.float64,
            )
        scores = trainer.strategy.broadcast(scores, src=0)
        logged = dict(zip(self.metric_names, scores, strict=True))
        for name, score in logged.items():
            pl_module.log(name, score, on_step=False, on_epoch=True, prog_bar=name.startswith("valid/"))
        trainer.callback_metrics.update(logged)
