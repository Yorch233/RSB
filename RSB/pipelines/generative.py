"""Lightning training pipeline for the generative RSB model."""

from __future__ import annotations

from typing import Literal

import lightning as L
import torch
from torch import Tensor

from RSB.data import STFTUtil
from RSB.modeling_rsb import RSB, validate_training_method
from RSB.pipelines.predictive import OptimizerName


class RSBTrainingPipeline(L.LightningModule):
    """Own the RSB optimization, perturbation, loss, and validation flow."""

    def __init__(
        self,
        model: RSB,
        *,
        learning_rate: float = 1e-4,
        optimizer_name: OptimizerName = "Adam",
        reduction: Literal["mean", "sum"] = "sum",
        regularization_weight: Literal["quadratic", "cosine", "linear"] = "quadratic",
        time_loss_weight: float = 1e-3,
        t_min: float = 1e-4,
        t_max: float = 1.0,
    ) -> None:
        """Initialize an RSB training pipeline."""
        super().__init__()
        validate_training_method(model.training_method)
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.reduction = reduction
        self.regularization_weight = regularization_weight
        self.time_loss_weight = time_loss_weight
        self.t_min = t_min
        self.t_max = t_max
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: Tensor, t: Tensor, condition: list[Tensor]) -> Tensor:
        """Delegate target prediction to the RSB model."""
        return self.model(x, t, condition)

    def omega(self, timestep: Tensor, dimensions: int) -> Tensor:
        """Compute the time-varying perturbation weight."""
        if self.regularization_weight == "quadratic":
            weight = timestep.square()
        elif self.regularization_weight == "cosine":
            weight = (1.0 - torch.cos(torch.pi * timestep)) / 2.0
        else:
            weight = timestep
        return weight.reshape((weight.shape[0],) + (1,) * (dimensions - 1))

    def _reduce_prediction_loss(self, loss: Tensor, timestep: Tensor) -> Tensor:
        flattened = loss.reshape(loss.shape[0], -1)
        per_sample = flattened.mean(dim=-1) if self.reduction == "mean" else 0.5 * flattened.sum(dim=-1)
        return (per_sample * self.model.sde.compute_weight(timestep)).mean()

    @staticmethod
    def _time_loss(predicted_audio: Tensor, target_audio: Tensor) -> Tensor:
        """Match the reference per-sample summed waveform L1 objective."""
        return torch.nn.functional.l1_loss(predicted_audio, target_audio, reduction="sum") / target_audio.shape[0]

    def _shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        self.model.sync_device()
        clean, noisy = batch[:2]
        posterior_mean = None
        if self.model.training_method == "regularization":
            if len(batch) < 3:
                raise RuntimeError("Regularization training requires an offline posterior mean in each dataset batch")
            posterior_mean = batch[2]

        timestep = torch.rand(clean.shape[0], device=clean.device) * (self.t_max - self.t_min) + self.t_min
        target = clean
        terminal = noisy
        condition = [noisy]
        if self.model.training_method == "regularization":
            if posterior_mean is None:
                raise RuntimeError("Regularization training requires a posterior mean")
            weight = self.omega(timestep, clean.ndim)
            target = weight * posterior_mean + (1.0 - weight) * clean

        perturbed = self.model.sde.q_sample(t=timestep, x0=target, x1=terminal)
        network_output = self(perturbed, timestep, condition)
        label = self.model.sde.compute_label(xt=perturbed, t=timestep, x0=target, x1=terminal)
        prediction_loss = self._reduce_prediction_loss(torch.square(torch.abs(network_output - label)), timestep)
        predicted_clean = self.model.sde.compute_pred_x0(
            xt=perturbed,
            t=timestep,
            x1=terminal,
            net_out=network_output,
        )
        clean_audio = STFTUtil.istft(target.squeeze(1)).squeeze(1)
        predicted_audio = STFTUtil.istft(predicted_clean.squeeze(1)).squeeze(1)
        time_loss = self._time_loss(predicted_audio, clean_audio)
        total_loss = prediction_loss + self.time_loss_weight * time_loss
        return {"loss": total_loss, "prediction_loss": prediction_loss, "time_loss": time_loss}

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """Run and log one RSB training step."""
        del batch_idx
        losses = self._shared_step(batch)
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/loss_per_epoch", losses["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train/prediction_loss_per_epoch",
            losses["prediction_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("train/time_loss_per_epoch", losses["time_loss"], on_step=False, on_epoch=True, sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """Run and log one RSB validation step."""
        del batch_idx
        losses = self._shared_step(batch)
        self.log(
            "valid/loss_per_epoch",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "valid/prediction_loss_per_epoch",
            losses["prediction_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("valid/time_loss_per_epoch", losses["time_loss"], on_step=False, on_epoch=True, sync_dist=True)
        return losses["loss"]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the configured torch optimizer."""
        optimizer_types: dict[str, type[torch.optim.Optimizer]] = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        try:
            optimizer_type = optimizer_types[self.optimizer_name]
        except KeyError as error:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name!r}") from error
        return optimizer_type(self.model.parameters(), lr=self.learning_rate)
