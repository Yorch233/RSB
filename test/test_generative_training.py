from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from RSB.callbacks import BestModelExport, EMACallback, GenerativeSampleMetrics, ValidationEarlyStopping
from RSB.training import generative
from RSB.utils.config import CURRENT_CONFIG_VERSION, Config, read_config_from_yaml


def make_config(run_dir: Path, **overrides) -> Config:
    values = {
        "run_dir": str(run_dir),
        "logger": "none",
        "ema": True,
        "ema_rate": 0.999,
        "patience": 20,
        "training_method": "regularization",
        "posterior_mean_from": "NCSN++M",
        "dataset": "voicebank",
        "batch_size": 2,
        "ncsnpp_operator_backend": "cuda_jit",
        "ncsnpp_cuda_jit_status": "available",
    }
    values.update(overrides)
    return Config(values)


def test_prepare_generative_run_uses_default_name_and_persists_config(tmp_path: Path) -> None:
    run = generative.prepare_generative_run(
        make_config(tmp_path),
        now=datetime(2026, 7, 18, 9, 8),
    )

    assert run.run_path.name == "rsb_generative_07180908"
    persisted = read_config_from_yaml(run.run_path / "config.yml")
    assert persisted.run_name == "rsb_generative_07180908"
    assert persisted.run_type == "generative"
    assert persisted.run_id is None
    assert persisted.version == CURRENT_CONFIG_VERSION
    assert persisted.ncsnpp_operator_backend == "cuda_jit"
    assert persisted.ncsnpp_cuda_jit_status == "available"


def test_prepare_generative_run_reuses_rank_zero_run_for_distributed_worker(monkeypatch, tmp_path: Path) -> None:
    config = make_config(tmp_path, run_name="rsb_generative_distributed")
    primary = generative.prepare_generative_run(config, run_id="wandb-id")

    monkeypatch.setenv("LOCAL_RANK", "1")
    worker = generative.prepare_generative_run(
        make_config(tmp_path, run_name="rsb_generative_distributed"),
    )

    assert worker.run_path == primary.run_path
    assert worker.config.run_id == "wandb-id"
    assert worker.checkpoint_path is None


def test_generative_callbacks_respect_ema_switch(tmp_path: Path) -> None:
    enabled = generative.build_generative_callbacks(make_config(tmp_path, ema=True), tmp_path)
    disabled = generative.build_generative_callbacks(make_config(tmp_path, ema=False), tmp_path)

    assert any(isinstance(callback, EMACallback) for callback in enabled)
    assert not any(isinstance(callback, EMACallback) for callback in disabled)


def test_generative_callbacks_use_reference_metric_selection(tmp_path: Path) -> None:
    callbacks = generative.build_generative_callbacks(make_config(tmp_path, patience=20), tmp_path)
    sampled = next(callback for callback in callbacks if isinstance(callback, GenerativeSampleMetrics))
    best = next(callback for callback in callbacks if isinstance(callback, BestModelExport))
    early_stopping = next(callback for callback in callbacks if isinstance(callback, ValidationEarlyStopping))

    assert sampled.valid_samples == 50
    assert sampled.test_samples == 5
    assert sampled.num_steps == 20
    assert best.monitor == "valid/PESQ_per_epoch"
    assert best.mode == "max"
    assert best.config_field == "best_pesq"
    assert early_stopping.monitor == "valid/SI_SDR_per_epoch"
    assert early_stopping.mode == "max"
    assert early_stopping.patience == 20


def test_best_model_export_maximizes_valid_pesq(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    callback = BestModelExport(
        tmp_path,
        config,
        monitor="valid/PESQ_per_epoch",
        mode="max",
        config_field="best_pesq",
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        is_global_zero=True,
        current_epoch=0,
        callback_metrics={"valid/PESQ_per_epoch": torch.tensor(2.8)},
    )
    module = SimpleNamespace(model=nn.Linear(2, 2))

    callback.on_validation_epoch_end(trainer, module)
    trainer.callback_metrics["valid/PESQ_per_epoch"] = torch.tensor(2.7)
    callback.on_validation_epoch_end(trainer, module)
    trainer.current_epoch = 2
    trainer.callback_metrics["valid/PESQ_per_epoch"] = torch.tensor(3.0)
    callback.on_validation_epoch_end(trainer, module)

    persisted = read_config_from_yaml(tmp_path / "config.yml")
    assert callback.best_score == pytest.approx(3.0)
    assert persisted.best_pesq == pytest.approx(3.0)
    assert persisted.best_model_epoch == 2


def test_generative_dataloader_requests_offline_posterior_mean(monkeypatch, tmp_path: Path) -> None:
    captured = []

    class DummyDataset(Dataset):
        def __init__(self, *args, **kwargs) -> None:
            del args
            captured.append(kwargs)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            del index
            return torch.zeros(1), torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(generative, "ComplexSpecDataset", DummyDataset)
    generative.build_generative_dataloader(make_config(tmp_path), "train")
    generative.build_generative_dataloader(make_config(tmp_path, training_method="none"), "valid")

    assert captured[0]["posterior_mean_from"] == "NCSN++M"
    assert captured[1]["posterior_mean_from"] is None


def test_generative_training_rejects_unsupported_method_before_creating_run(monkeypatch, tmp_path: Path) -> None:
    config = make_config(tmp_path, training_method="unsupported")
    monkeypatch.setattr(
        generative,
        "prepare_generative_run",
        lambda *args, **kwargs: pytest.fail("a run must not be created for an unsupported training method"),
    )

    with pytest.raises(ValueError, match="training_method must be one of"):
        generative.start_generative_training(config)


def test_generative_dataloader_rejects_unsupported_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training_method must be one of"):
        generative.build_generative_dataloader(make_config(tmp_path, training_method="unsupported"), "train")
