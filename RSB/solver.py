"""Numerical solvers for reverse Schrodinger Bridge sampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

PredictionFunction = Callable[[Tensor, Tensor | float], Tensor]


class Solver:
    """Base numerical solver with shared timestep construction."""

    def __init__(
        self,
        sde: Any,
        model_fn: PredictionFunction | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize a solver for an SDE and prediction function."""
        self.sde = sde
        self.model_fn = model_fn
        self.device = torch.device(device if device is not None else sde.device)

    def sampling(self, x: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a reverse trajectory."""
        raise NotImplementedError

    def get_time_steps(
        self,
        skip_type: str = "time_uniform",
        t_start: float = 1.0,
        t_end: float = 0.0,
        num_step: int = 20,
    ) -> Tensor:
        """Construct a descending linear or quadratic timestep schedule."""
        if skip_type == "time_uniform":
            return torch.linspace(t_start, t_end, num_step + 1, device=self.device)
        if skip_type == "time_quadratic":
            roots = torch.linspace(t_start**0.5, t_end**0.5, num_step + 1, device=self.device)
            return roots.square()
        raise ValueError(f"Unsupported skip_type {skip_type!r}; choose 'time_uniform' or 'time_quadratic'")

    @staticmethod
    def _stack_reverse(values: list[Tensor]) -> Tensor:
        return torch.flip(torch.stack([value.to("cpu", non_blocking=True) for value in values], dim=1), dims=(1,))

    def _require_model_fn(self) -> PredictionFunction:
        if self.model_fn is None:
            raise RuntimeError("A prediction function is required for sampling")
        return self.model_fn


class SDESolver(Solver):
    """Euler-Maruyama solver for the reverse bridge SDE."""

    @torch.no_grad()
    def sampling(
        self,
        x: Tensor,
        num_step: int = 5,
        skip_type: str = "time_uniform",
        t_max: float = 1.0,
        t_min: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Integrate the stochastic reverse process."""
        state = x.to(self.device, non_blocking=True)
        states = [state]
        predictions: list[Tensor] = []
        timesteps = self.get_time_steps(skip_type, t_max, t_min, num_step)
        model_fn = self._require_model_fn()
        for index in range(num_step):
            timestep, previous_timestep = timesteps[index], timesteps[index + 1]
            prediction = model_fn(state, timestep)
            state = self.sde.first_order_sde_sampling(state, prediction, timestep, previous_timestep)
            predictions.append(prediction)
            states.append(state)
        return state, self._stack_reverse(states), self._stack_reverse(predictions)


class ODESolver(Solver):
    """First-order solver for the probability-flow bridge ODE."""

    @torch.no_grad()
    def sampling(
        self,
        x: Tensor,
        num_step: int = 5,
        skip_type: str = "time_uniform",
        t_max: float = 1.0,
        t_min: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Integrate the deterministic reverse process."""
        state = x.to(self.device, non_blocking=True)
        terminal = state
        states = [state]
        predictions: list[Tensor] = []
        timesteps = self.get_time_steps(skip_type, t_max, t_min, num_step)
        model_fn = self._require_model_fn()
        for index in range(num_step):
            timestep, previous_timestep = timesteps[index], timesteps[index + 1]
            prediction = model_fn(state, timestep)
            state = self.sde.first_order_ode_sampling(
                x1=terminal,
                xt=state,
                x0=prediction,
                t=timestep,
                t_prev=previous_timestep,
            )
            predictions.append(prediction)
            states.append(state)
        return state, self._stack_reverse(states), self._stack_reverse(predictions)
