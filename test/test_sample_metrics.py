from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch
from lightning.pytorch.loggers import Logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from RSB.callbacks import BestModelExport, ValidationEarlyStopping, sample_metrics
from RSB.callbacks.sample_metrics import GenerativeSampleMetrics, evaluate_sampled_split
from RSB.utils.config import Config, read_config_from_yaml


class IdentityRSB:
    training_method = "regularization"

    def __init__(self) -> None:
        self.calls = 0

    def enhance(self, audio, **kwargs):
        assert kwargs["num_steps"] == 20
        assert kwargs["solver"] == "SDE"
        self.calls += 1
        return audio, torch.empty(0), torch.empty(0)


class ConstantPESQ:
    @classmethod
    def compute(cls, **kwargs):
        del kwargs
        return {"PESQ": 3.0}


class ConstantSiSdr:
    @classmethod
    def compute(cls, **kwargs):
        del kwargs
        return {"SI_SDR": 10.0}


def test_evaluate_sampled_split_uses_fixed_prefix(monkeypatch, tmp_path: Path) -> None:
    pairs = [(tmp_path / "clean" / f"{index:03}.wav", tmp_path / "noisy" / f"{index:03}.wav") for index in range(60)]
    monkeypatch.setattr(sample_metrics, "resolve_dataset", lambda config, dataset: (dataset, tmp_path))
    monkeypatch.setattr(sample_metrics, "paired_wavs", lambda root, split: pairs)
    monkeypatch.setattr(sample_metrics, "_load_audio", lambda path, sample_rate: torch.ones(1, sample_rate))
    monkeypatch.setattr(
        sample_metrics.MetricRegister,
        "fetch",
        lambda names: {"pesq": ConstantPESQ, "si_sdr": ConstantSiSdr},
    )
    model = IdentityRSB()

    result = evaluate_sampled_split(
        model,
        Config({"dataset": "voicebank", "sample_rate": 16_000}),
        "valid",
        max_samples=50,
        num_steps=20,
        progress=False,
    )

    assert result == {"PESQ": 3.0, "SI_SDR": 10.0}
    assert model.calls == 50


def test_sample_metric_callback_logs_valid_and_test_without_using_test_for_selection(monkeypatch) -> None:
    calls = []

    def evaluate(model, config, split, **kwargs):
        del model, config
        calls.append((split, kwargs["max_samples"], kwargs["num_steps"]))
        return {"PESQ": 3.0 if split == "valid" else 2.0, "SI_SDR": 10.0 if split == "valid" else 5.0}

    monkeypatch.setattr(sample_metrics, "evaluate_sampled_split", evaluate)
    logged = {}
    module = SimpleNamespace(
        device=torch.device("cpu"),
        model=IdentityRSB(),
        log=lambda name, value, **kwargs: logged.update({name: float(value)}),
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        is_global_zero=True,
        strategy=SimpleNamespace(broadcast=lambda tensor, src: tensor),
        callback_metrics={},
    )

    GenerativeSampleMetrics(Config({}), valid_samples=50, test_samples=5, num_steps=20).on_validation_epoch_end(
        trainer, module
    )

    assert calls == [("valid", 50, 20), ("test", 5, 20)]
    assert logged == {
        "valid/PESQ_per_epoch": 3.0,
        "valid/SI_SDR_per_epoch": 10.0,
        "test/PESQ_per_epoch": 2.0,
        "test/SI_SDR_per_epoch": 5.0,
    }
    assert set(trainer.callback_metrics) == set(logged)


def test_sample_metrics_drive_lightning_selection_callbacks(monkeypatch, tmp_path: Path) -> None:
    class CaptureLogger(Logger):
        def __init__(self) -> None:
            super().__init__()
            self.logged_metrics = []

        @property
        def name(self) -> str:
            return "capture"

        @property
        def version(self) -> str:
            return "0"

        def log_hyperparams(self, params) -> None:
            del params

        def log_metrics(self, metrics, step=None) -> None:
            self.logged_metrics.append((dict(metrics), step))

    class TinyPipeline(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            del batch_idx
            inputs, targets = batch
            return torch.nn.functional.mse_loss(self.model(inputs), targets)

        def validation_step(self, batch, batch_idx):
            del batch_idx
            inputs, targets = batch
            loss = torch.nn.functional.mse_loss(self.model(inputs), targets)
            self.log("valid/loss_per_epoch", loss, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    monkeypatch.setattr(
        sample_metrics,
        "evaluate_sampled_split",
        lambda model, config, split, **kwargs: {
            "PESQ": 3.0 if split == "valid" else 2.0,
            "SI_SDR": 10.0 if split == "valid" else 5.0,
        },
    )
    config = Config({})
    early_stopping = ValidationEarlyStopping(
        patience=2,
        monitor="valid/SI_SDR_per_epoch",
        mode="max",
    )
    callbacks = [
        GenerativeSampleMetrics(config),
        BestModelExport(
            tmp_path,
            config,
            monitor="valid/PESQ_per_epoch",
            mode="max",
            config_field="best_pesq",
        ),
        early_stopping,
    ]
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1)
    logger = CaptureLogger()
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(TinyPipeline(), train_dataloaders=loader, val_dataloaders=loader)

    persisted = read_config_from_yaml(tmp_path / "config.yml")
    assert persisted.best_pesq == 3.0
    assert (tmp_path / "model.safetensors").is_file()
    early_stopping_metrics = next(
        metrics for metrics, _ in logger.logged_metrics if "valid/early_stopping_wait_count_per_epoch" in metrics
    )
    assert early_stopping_metrics == {
        "valid/early_stopping_wait_count_per_epoch": 0,
        "valid/early_stopping_best_value_per_epoch": 10.0,
        "valid/early_stopping_best_epoch_per_epoch": 0,
    }
    assert early_stopping.best_epoch == 0
    assert early_stopping.state_dict()["best_epoch"] == 0
