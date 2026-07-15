"""Core Regularized Schrodinger Bridge model and sampling interface."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import torch
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_model
from torch import Tensor, nn

from RSB.backbone import BackboneRegister
from RSB.data import STFTUtil
from RSB.sdes import SB_VESDE, SB_VPSDE
from RSB.solver import ODESolver, SDESolver
from RSB.utils.config import read_config_from_yaml

SolverName = Literal["SDE", "ODE"]
SUPPORTED_TRAINING_METHODS = frozenset({"none", "regularization"})


def validate_training_method(training_method: str) -> None:
    """Require one of the training methods supported by the released model."""
    if training_method not in SUPPORTED_TRAINING_METHODS:
        choices = ", ".join(sorted(SUPPORTED_TRAINING_METHODS))
        raise ValueError(f"training_method must be one of: {choices}")


class RSB(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="regularized-schrodinger-bridge",
    repo_url="https://github.com/Yorch233/RSB",
    tags=["audio", "speech-enhancement", "schrodinger-bridge"],
):
    """Regularized Schrodinger Bridge model for speech enhancement.

    The Hugging Face mixin provides ``save_pretrained`` and ``from_pretrained``;
    weights are serialized as safetensors together with the constructor config.
    """

    def __init__(
        self,
        backbone: str = "ncsnpp_base",
        training_method: str = "none",
        training_target: str = "data",
        loss_weight_type: str = "constant",
        bridge_type: Literal["VE", "VP"] = "VE",
        sampling_solver: SolverName = "SDE",
        device: str | torch.device = "cpu",
        backbone_kwargs: dict[str, Any] | None = None,
        sde_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the backbone, bridge SDE, and sampling algorithms."""
        super().__init__()
        validate_training_method(training_method)
        self.backbone_name = backbone
        self.training_method = training_method
        self.training_target = training_target
        self.loss_weight_type = loss_weight_type
        self.bridge_type = bridge_type
        self.sampling_solver = sampling_solver

        self.generator = BackboneRegister.fetch(backbone)(
            input_channels=4,
            **(backbone_kwargs or {}),
        )
        sde_types = {"VE": SB_VESDE, "VP": SB_VPSDE}
        try:
            sde_type = sde_types[bridge_type]
        except KeyError as error:
            raise ValueError(f"Unsupported bridge type: {bridge_type!r}") from error
        self.sde = sde_type(
            training_target=training_target,
            loss_weight_type=loss_weight_type,
            device=device,
            **(sde_kwargs or {}),
        )

        self._sampling_target: Tensor | None = None
        self._sampling_condition: list[Tensor] = []
        self._sampling_evaluations = 0
        self.sde_sampler = self.sde.get_sde_solver(model_fn=self._predict_x0)
        self.ode_sampler = self.sde.get_ode_solver(model_fn=self._predict_x0)
        self.sampler = self._select_sampler(sampling_solver)
        self.to(device)

    @property
    def model_device(self) -> torch.device:
        """Return the device holding the model parameters."""
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        map_location: str | torch.device = "cpu",
        **model_kwargs: Any,
    ) -> Self:
        """Load a Hub checkpoint or a local RSB run containing config.yml."""
        model_path = Path(pretrained_model_name_or_path)
        run_config_path = model_path / "config.yml"
        if model_path.is_dir() and run_config_path.is_file() and not (model_path / "config.json").is_file():
            run_config = read_config_from_yaml(run_config_path)
            constructor_keys = {
                "training_method",
                "training_target",
                "loss_weight_type",
                "bridge_type",
                "sampling_solver",
                "backbone_kwargs",
                "sde_kwargs",
            }
            constructor_config = {key: value for key, value in run_config.dict().items() if key in constructor_keys}
            constructor_config["backbone"] = run_config.get(
                "generative_backbone", run_config.get("backbone", "ncsnpp_base")
            )
            constructor_config.update(model_kwargs)
            constructor_config["device"] = map_location
            model = cls(**constructor_config)
            load_model(model, model_path / "model.safetensors", device=str(map_location))
            return model.eval()
        return super().from_pretrained(
            pretrained_model_name_or_path,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            map_location=map_location,
            **model_kwargs,
        )

    def sync_device(self) -> None:
        """Move non-module SDE state and samplers alongside model parameters."""
        device = self.model_device
        self.sde.device = device
        self.sde_sampler.device = device
        self.ode_sampler.device = device
        for name, value in vars(self.sde).items():
            if isinstance(value, Tensor):
                setattr(self.sde, name, value.to(device))

    def _select_sampler(self, solver: SolverName) -> SDESolver | ODESolver:
        if solver == "SDE":
            return self.sde_sampler
        if solver == "ODE":
            return self.ode_sampler
        raise ValueError(f"Unsupported sampling solver: {solver!r}")

    def forward(self, x: Tensor, t: Tensor, condition: Sequence[Tensor] | None = None) -> Tensor:
        """Predict the configured SDE target for a perturbed spectrum."""
        inputs = torch.cat([x, *(condition or [])], dim=1)
        return self.generator(inputs, t)

    @torch.no_grad()
    def _predict_x0(self, xt: Tensor, timestep: Tensor | float) -> Tensor:
        if self._sampling_target is None:
            raise RuntimeError("Sampling context has not been initialized")
        time = torch.as_tensor(timestep, device=self.model_device, dtype=torch.float32).expand(xt.shape[0])
        network_output = self(xt, time, condition=self._sampling_condition)
        self._sampling_evaluations += 1
        return self.sde.compute_pred_x0(
            xt=xt,
            t=time,
            x1=self._sampling_target,
            net_out=network_output,
        )

    @torch.no_grad()
    def sample(
        self,
        observation: Tensor,
        *,
        num_steps: int = 5,
        solver: SolverName | None = None,
        skip_type: str = "time_uniform",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an enhanced spectrum from a noisy observation spectrum."""
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        self.sync_device()
        observation = observation.to(self.model_device)
        self._sampling_target = observation
        self._sampling_condition = [observation]
        self._sampling_evaluations = 0
        active_sampler = self._select_sampler(solver or self.sampling_solver)
        self.sampler = active_sampler
        try:
            sample, trajectory, predictions = active_sampler.sampling(
                x=observation,
                num_step=num_steps,
                skip_type=skip_type,
            )
        finally:
            self._sampling_target = None
            self._sampling_condition = []
        if self._sampling_evaluations != num_steps:
            raise RuntimeError(
                f"Sampler performed {self._sampling_evaluations} model evaluations; expected {num_steps}"
            )
        return sample, trajectory, predictions

    @torch.no_grad()
    def enhance(
        self,
        audio: Tensor,
        *,
        num_steps: int = 5,
        solver: SolverName | None = None,
        skip_type: str = "time_uniform",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Enhance a waveform and return it with spectral sampling trajectories."""
        observation, invert = STFTUtil.to_stft(audio, device=self.model_device)
        enhanced, trajectory, predictions = self.sample(
            observation,
            num_steps=num_steps,
            solver=solver,
            skip_type=skip_type,
        )
        return invert(enhanced), trajectory, predictions
