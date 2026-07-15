"""Exponential moving average callback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback
from torch import Tensor

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class EMACallback(Callback):
    """Track model parameters with EMA and evaluate using the averaged weights."""

    def __init__(self, decay: float = 0.999) -> None:
        """Initialize EMA.

        Args:
            decay: Weight assigned to the previous moving average.
        """
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow_parameters: list[Tensor] = []
        self.backup_parameters: list[Tensor] = []
        self.num_updates = 0

    @staticmethod
    def _parameters(pl_module: LightningModule) -> list[Tensor]:
        return [parameter for parameter in pl_module.parameters() if parameter.requires_grad]

    def _initialize(self, pl_module: LightningModule) -> None:
        if not self.shadow_parameters:
            self.shadow_parameters = [parameter.detach().clone() for parameter in self._parameters(pl_module)]

    def state_dict(self) -> dict[str, Any]:
        """Save EMA weights and update count in the Lightning checkpoint."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_parameters": self.shadow_parameters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA state when training resumes."""
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.shadow_parameters = list(state_dict["shadow_parameters"])

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize EMA after Lightning places the model on its training device."""
        del trainer
        self._initialize(pl_module)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA after each optimizer step."""
        del trainer, outputs, batch, batch_idx
        parameters = self._parameters(pl_module)
        self._initialize(pl_module)
        for shadow, parameter in zip(self.shadow_parameters, parameters, strict=True):
            shadow.data = shadow.to(device=parameter.device, dtype=parameter.dtype)
            shadow.lerp_(parameter.detach(), 1.0 - self.decay)
        self.num_updates += 1

    @torch.no_grad()
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Swap EMA parameters in for validation."""
        del trainer
        parameters = self._parameters(pl_module)
        self._initialize(pl_module)
        self.backup_parameters = [parameter.detach().clone() for parameter in parameters]
        for parameter, shadow in zip(parameters, self.shadow_parameters, strict=True):
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore trainable parameters after validation."""
        del trainer
        if not self.backup_parameters:
            return
        for parameter, backup in zip(self._parameters(pl_module), self.backup_parameters, strict=True):
            parameter.copy_(backup)
        self.backup_parameters = []

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Restore trainable weights if validation exits with an exception."""
        del exception
        self.on_validation_end(trainer, pl_module)
