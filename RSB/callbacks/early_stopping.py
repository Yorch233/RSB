"""Validation-based early stopping callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import EarlyStopping

from RSB.utils.config import Config

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class ValidationEarlyStopping(EarlyStopping):
    """Stop training after a configured monitored metric stops improving."""

    metric_names = {
        "wait_count": "valid/early_stopping_wait_count_per_epoch",
        "best_value": "valid/early_stopping_best_value_per_epoch",
        "best_epoch": "valid/early_stopping_best_epoch_per_epoch",
    }

    def __init__(
        self,
        patience: int = 50,
        monitor: str = "valid/loss_per_epoch",
        *,
        mode: str = "min",
        config: Config | None = None,
        run_path: str | Path | None = None,
        best_field: str | None = None,
    ) -> None:
        """Initialize monitored early stopping and optional run-config persistence."""
        super().__init__(
            monitor=monitor,
            mode=mode,
            patience=patience,
            check_finite=True,
            check_on_train_epoch_end=False,
        )
        self.run_config = config
        self.run_path = Path(run_path) if run_path is not None else None
        self.best_field = best_field
        self.best_epoch = -1

    def state_dict(self) -> dict[str, Any]:
        """Persist the epoch associated with the best monitored value."""
        return {**super().state_dict(), "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore early-stopping state, including the best epoch."""
        parent_state = dict(state_dict)
        self.best_epoch = int(parent_state.pop("best_epoch", -1))
        super().load_state_dict(parent_state)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run the stopping check, log its state, and persist summary fields."""
        previous_best = float(self.best_score.detach().cpu())
        super().on_validation_end(trainer, pl_module)
        if trainer.sanity_checking:
            return
        best_value = float(self.best_score.detach().cpu())
        if best_value != previous_best:
            self.best_epoch = trainer.current_epoch
        if not trainer.is_global_zero:
            return

        logger = trainer.logger
        if logger is not None:
            logger.log_metrics(
                {
                    self.metric_names["wait_count"]: int(self.wait_count),
                    self.metric_names["best_value"]: best_value,
                    self.metric_names["best_epoch"]: self.best_epoch,
                },
                step=trainer.global_step,
            )
        if self.run_config is None or self.run_path is None:
            return
        updates = {"early_stop_cnt": int(self.wait_count)}
        if self.best_field is not None:
            updates[self.best_field] = float(self.best_score.detach().cpu())
        self.run_config.update(updates)
        self.run_config.save(self.run_path)
