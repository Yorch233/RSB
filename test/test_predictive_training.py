from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from RSB.callbacks import FullStateModelCheckpoint
from RSB.pipelines.predictive import PredictiveModelTrainingPipeline
from RSB.training.predictive import build_predictive_callbacks, prepare_predictive_run
from RSB.utils.config import CURRENT_CONFIG_VERSION, Config, read_config_from_yaml


def make_config(run_dir: Path) -> Config:
    return Config(
        {
            "run_dir": str(run_dir),
            "learning_rate": 1e-3,
            "optimizer": "Adam",
            "reduction": "mean",
            "ema_rate": 0.9,
            "predictive_patience": 50,
            "logger": "none",
            "save_state_steps": 1,
            "checkpoints_total_limit": 2,
            "ncsnpp_operator_backend": "cuda_jit",
            "ncsnpp_cuda_jit_status": "available",
        }
    )


def test_full_state_checkpoint_uses_ddp_safe_train_end_hook() -> None:
    assert FullStateModelCheckpoint.on_train_end is ModelCheckpoint.on_train_end


def test_predictive_pipeline_uses_split_and_frequency_metric_names(monkeypatch) -> None:
    pipeline = PredictiveModelTrainingPipeline(nn.Linear(2, 2), method="NCSN++M")
    logged = []
    monkeypatch.setattr(pipeline, "log", lambda name, value, **kwargs: logged.append((name, value, kwargs)))
    batch = (torch.zeros(2, 2), torch.ones(2, 2))

    pipeline.training_step(batch, 0)
    pipeline.validation_step(batch, 0)

    assert [name for name, _, _ in logged] == [
        "train/loss",
        "train/loss_per_epoch",
        "valid/loss_per_epoch",
    ]
    assert logged[0][2]["on_step"] is True
    assert logged[0][2]["on_epoch"] is False
    assert all(entry[2]["on_epoch"] is True for entry in logged[1:])


def test_prepare_predictive_run_persists_name_method_and_wandb_id(tmp_path: Path) -> None:
    run = prepare_predictive_run(
        make_config(tmp_path),
        method="NCSN++M",
        now=datetime(2026, 7, 17, 9, 8),
        run_id="wandb-id",
    )

    assert run.run_path.name == "rsb_predictive_07170908"
    persisted = read_config_from_yaml(run.run_path / "config.yml")
    assert persisted.predictive_method == "NCSN++M"
    assert persisted.run_type == "predictive"
    assert persisted.predictive_backbone == "ncsnpp_base"
    assert persisted.run_id == "wandb-id"
    assert persisted.version == CURRENT_CONFIG_VERSION
    assert persisted.ncsnpp_operator_backend == "cuda_jit"
    assert persisted.ncsnpp_cuda_jit_status == "available"


def test_prepare_predictive_run_reuses_rank_zero_run_for_distributed_worker(monkeypatch, tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.update({"run_name": "rsb_predictive_distributed"})
    primary = prepare_predictive_run(config, method="NCSN++M", run_id="wandb-id")

    monkeypatch.setenv("LOCAL_RANK", "1")
    worker_config = make_config(tmp_path)
    worker_config.update({"run_name": "rsb_predictive_distributed"})
    worker = prepare_predictive_run(worker_config, method="NCSN++M")

    assert worker.run_path == primary.run_path
    assert worker.config.run_id == "wandb-id"
    assert worker.checkpoint_path is None


def test_lightning_callbacks_save_resumable_state_and_best_export(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.update({"run_name": "rsb_predictive_test", "run_id": "test-id"})
    config.save(tmp_path)
    clean = torch.zeros(8, 2)
    noisy = torch.ones(8, 2)
    loader = DataLoader(TensorDataset(clean, noisy), batch_size=2)
    pipeline = PredictiveModelTrainingPipeline(
        nn.Linear(2, 2),
        method="NCSN++M",
        learning_rate=config.learning_rate,
        optimizer_name=config.optimizer,
        reduction=config.reduction,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        callbacks=build_predictive_callbacks(config, tmp_path),
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(pipeline, train_dataloaders=loader, val_dataloaders=loader)

    last_checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    assert last_checkpoint.is_file()
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["optimizer_states"]
    assert checkpoint["loops"]
    assert any("EMACallback" in callback_name for callback_name in checkpoint["callbacks"])
    assert any("ValidationEarlyStopping" in callback_name for callback_name in checkpoint["callbacks"])
    assert set(load_file(tmp_path / "model.safetensors")) == {"bias", "weight"}
    best_config = read_config_from_yaml(tmp_path / "config.yml")
    assert best_config.best_valid_loss >= 0
    assert best_config.best_model_epoch in {0, 1}
    assert len(list((tmp_path / "checkpoints").glob("step=*.ckpt"))) <= config.checkpoints_total_limit


def test_prepare_predictive_run_resumes_last_checkpoint(tmp_path: Path) -> None:
    run_path = tmp_path / "rsb_predictive_07170908"
    checkpoints = run_path / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "last.ckpt").touch()
    config = make_config(tmp_path)
    config.update(
        {
            "predictive_method": "NCSN++M",
            "predictive_backbone": "ncsnpp_base",
            "run_name": run_path.name,
            "run_id": "wandb-id",
            "run_path": str(run_path),
        }
    )
    config.save(run_path)

    resumed = prepare_predictive_run(config, method="NCSN++M", checkpoint_path=run_path)

    assert resumed.run_path == run_path.resolve()
    assert resumed.checkpoint_path == run_path.resolve() / "checkpoints" / "last.ckpt"
    assert resumed.config.run_id == "wandb-id"
