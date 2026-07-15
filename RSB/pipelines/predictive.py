"""Lightning pipeline for predictive-model training."""

from __future__ import annotations

from typing import Literal

import lightning as L
import torch
from torch import Tensor, nn

OptimizerName = Literal["Adam", "AdamW"]


class PredictiveModelTrainingPipeline(L.LightningModule):
    """Train a predictive speech-enhancement model from noisy/clean spectra."""

    def __init__(
        self,
        model: nn.Module,
        *,
        method: str,
        learning_rate: float = 1e-4,
        optimizer_name: OptimizerName = "Adam",
        reduction: Literal["mean", "sum"] = "sum",
    ) -> None:
        """Initialize the predictive-model optimization pipeline.

        Args:
            model: Predictive network mapping a noisy spectrum to a clean spectrum.
            method: User-facing predictive method name.
            learning_rate: Optimizer learning rate.
            optimizer_name: Optimizer implementation.
            reduction: Per-sample loss reduction.
        """
        super().__init__()
        self.model = model
        self.method = method
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.reduction = reduction
        self.save_hyperparameters(ignore="model")

    def forward(self, noisy: Tensor) -> Tensor:
        """Predict a clean complex spectrum from a noisy spectrum."""
        return self.model(noisy)

    def _loss(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        clean, noisy = batch
        squared_error = torch.square(torch.abs(self(noisy) - clean))
        flattened = squared_error.reshape(squared_error.shape[0], -1)
        per_sample = flattened.mean(dim=-1) if self.reduction == "mean" else 0.5 * flattened.sum(dim=-1)
        return per_sample.mean()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute and log one training step."""
        del batch_idx
        loss = self._loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/loss_per_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute and log one validation step."""
        del batch_idx
        loss = self._loss(batch)
        self.log("valid/loss_per_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the configured optimizer."""
        optimizers: dict[str, type[torch.optim.Optimizer]] = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        try:
            optimizer_type = optimizers[self.optimizer_name]
        except KeyError as error:
            supported = ", ".join(optimizers)
            raise ValueError(f"Unsupported optimizer {self.optimizer_name!r}; choose from {supported}") from error
        return optimizer_type(self.parameters(), lr=self.learning_rate)
