"""Full-state and best-model checkpoint callbacks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from safetensors.torch import save_model
from torch import Tensor

from RSB.utils.config import Config

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class FullStateModelCheckpoint(ModelCheckpoint):
    """Periodically save bounded, fully resumable Lightning states."""

    def __init__(self, run_path: str | Path, *, save_state_steps: int, checkpoints_total_limit: int) -> None:
        """Initialize checkpoint storage below a run directory."""
        if save_state_steps < 1:
            raise ValueError("save_state_steps must be positive")
        if checkpoints_total_limit < 1:
            raise ValueError("checkpoints_total_limit must be positive")
        self.checkpoints_total_limit = checkpoints_total_limit
        super().__init__(
            dirpath=Path(run_path) / "checkpoints",
            filename="step={step}",
            monitor=None,
            save_top_k=-1,
            save_last=True,
            save_weights_only=False,
            save_on_exception=True,
            every_n_train_steps=save_state_steps,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save on schedule, then prune intermediate checkpoints beyond the configured limit."""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.is_global_zero:
            self._prune_intermediate_checkpoints()

    def _prune_intermediate_checkpoints(self) -> None:
        checkpoints = sorted(Path(self.dirpath).glob("step=*.ckpt"), key=lambda path: path.stat().st_mtime_ns)
        for checkpoint in checkpoints[: -self.checkpoints_total_limit]:
            checkpoint.unlink()


class BestModelExport(Callback):
    """Export the best monitored model and its matching run configuration."""

    def __init__(
        self,
        run_path: str | Path,
        config: Config,
        monitor: str = "valid/loss_per_epoch",
        *,
        mode: str = "min",
        config_field: str = "best_valid_loss",
    ) -> None:
        """Initialize the safetensors export callback."""
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.run_path = Path(run_path)
        self.config = config
        self.monitor = monitor
        self.mode = mode
        self.config_field = config_field
        self.best_score: float | None = config.get(config_field)

    def state_dict(self) -> dict[str, Any]:
        """Persist the current best score in full-state checkpoints."""
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the current best score on resume."""
        best_score = state_dict.get("best_score")
        self.best_score = float(best_score) if best_score is not None else None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Atomically export improved EMA weights and update config.yml."""
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            raise RuntimeError(f"Monitored metric {self.monitor!r} was not logged")
        score = float(metric.detach().cpu()) if isinstance(metric, Tensor) else float(metric)
        improved = self.best_score is None or (
            score < self.best_score if self.mode == "min" else score > self.best_score
        )
        if not improved:
            return

        self.best_score = score
        if not trainer.is_global_zero:
            return
        model = getattr(pl_module, "model", pl_module)
        model_path = self.run_path / "model.safetensors"
        temporary_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
        save_model(model, temporary_path)
        os.replace(temporary_path, model_path)
        self.config.update({self.config_field: score, "best_model_epoch": trainer.current_epoch})
        self.config.save(self.run_path)
