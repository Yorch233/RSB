# RSB/solver.py
from typing import Callable

import torch


class Solver:
    """Abstract base class for diffusion model solvers.
    
    This class defines the common interface and utilities for numerical solvers
    used in Schrödinger Bridge and diffusion-based generative models.
    """

    def __init__(
        self,
        sde: object,
        model_fn: Callable = None,
        device: torch.device = None,
    ):
        """
        Initializes the solver.

        Args:
            sde (object): An instance of an SDE class (e.g., SB_VPSDE, SB_VESDE)
                          that defines the dynamics.
            model_fn (Callable, optional): A function that takes (x_t, t) and returns
                                           a prediction (e.g., x0, noise, score).
            device (torch.device, optional): The device to run computations on.
                                             Defaults to the SDE's device.
        """
        self.sde = sde
        self.model_fn = model_fn
        self.device = device if device is not None else sde.device

    def sampling(self, x: torch.Tensor):
        """
        Main sampling method to be implemented by subclasses.
        This method should generate a sample by evolving the input `x` backwards
        in time according to the SDE/ODE dynamics.

        Args:
            x (torch.Tensor): The initial state (often the terminal observation).

        Returns:
            tuple: Typically returns the final sample and intermediate states.
        """
        raise NotImplementedError()

    def get_time_steps(self,
                       skip_type: str = "time_uniform",
                       t_start: float = 1.0,
                       t_end: float = 0.0,
                       num_step: int = 20):
        """
        Generate time steps based on the skip type for discretizing the solver.

        Args:
            skip_type (str): Strategy for time step spacing.
                             Options: 'time_uniform', 'time_quadratic'.
            t_start (float): Start time (usually 1.0 for normalized time).
            t_end (float): End time (usually 0.0 for normalized time).
            num_step (int): Number of discretization steps.

        Returns:
            torch.Tensor: A 1D tensor of time steps on the specified device.
        """
        if skip_type == 'time_uniform':
            # Linear spacing between t_start and t_end
            return torch.linspace(t_start, t_end, num_step + 1).to(self.device)
        elif skip_type == 'time_quadratic':
            # Quadratic spacing, concentrating more steps near t_end
            t_order = 2
            t = torch.linspace(t_start**(1. / t_order), t_end**(1. / t_order),
                               num_step + 1).pow(t_order).to(self.device)
            return t
        else:
            raise ValueError(
                f"Unsupported skip_type '{skip_type}', must be one of 'time_uniform', or 'time_quadratic'."
            )


class SDESolver(Solver):
    """Stochastic Differential Equation (SDE) solver implementation.

    Performs sampling by numerically integrating the reverse SDE. 

    Args:
        sde (object): Stochastic differential equation object defining dynamics.
        model_fn (Callable): Function that predicts x0 (or another target) from (xt, t).
        device (torch.device): Computation device.
    """

    def __init__(
        self,
        sde: object,
        model_fn: Callable = None,
        device: torch.device = None,
    ):
        """
        Initializes the SDE solver.

        Args:
            sde (object): An instance of an SDE class (e.g., SB_VPSDE, SB_VESDE).
            model_fn (Callable, optional): A function that takes (x_t, t) and returns
                                           a prediction (e.g., x0).
            device (torch.device, optional): The device to run computations on.
        """
        super().__init__(sde, model_fn, device)

    def sampling(self,
                 x,
                 num_step: int = 5,
                 skip_type: str = "time_uniform",
                 t_max: float = 1.0,
                 t_min: float = 0.0):
        """
        Perform the SDE sampling process with specified discretization.

        Iteratively applies the reverse SDE dynamics from `t_max` to `t_min` using
        the provided `model_fn` to estimate the drift term.

        Args:
            x (torch.Tensor): Initial state (e.g., terminal observation `x1`).
            num_step (int): Number of discretization steps for the solver.
            skip_type (str): Time step spacing strategy ('time_uniform', 'time_quadratic').
            t_max (float): Maximum time (start of reverse process, default 1.0).
            t_min (float): Minimum time (end of reverse process, default 0.0).

        Returns:
            tuple: (final_sample, backward_trajectory_stack, predicted_x0_stack)
                - final_sample (torch.Tensor): The final generated sample (x0 estimate).
                - backward_trajectory_stack (torch.Tensor): Stack of intermediate xt states.
                - predicted_x0_stack (torch.Tensor): Stack of x0 predictions at each step.
        """
        x = x.to(self.device, non_blocking=True)

        xs = [x]  # Store the trajectory of states
        pred_x0s = []  # Store the predictions of x0

        # Generate time steps for the reverse process
        timesteps = self.get_time_steps(skip_type, t_max, t_min, num_step)

        # Iterate through time steps in reverse order
        for i in range(0, num_step):
            t, t_prev = timesteps[i], timesteps[i + 1]

            # Use the model to predict x0 from the current state xt and time t
            pred_x0 = self.model_fn(x, t)

            # Apply one step of the reverse SDE dynamics
            x = self.sde.fisrt_order_sde_sampling(
                x, pred_x0, t, t_prev)  # Note: Typo in method name

            # Store intermediate results
            pred_x0s.append(pred_x0)
            xs.append(x)

        # Move results back to CPU for storage/return
        xs = [x.to("cpu", non_blocking=True) for x in xs]
        pred_x0s = [
            pred_x0.to("cpu", non_blocking=True) for pred_x0 in pred_x0s
        ]

        # Helper function to stack and flip the trajectory for consistent ordering
        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1),
                                              dims=(1, ))

        return x, stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)


class ODESolver(Solver):
    """Ordinary Differential Equation (ODE) solver implementation.

    Performs sampling by numerically integrating the probability flow ODE associated
    with the SDE. 

    Args:
        sde (object): Stochastic differential equation object defining dynamics.
        model_fn (Callable): Function that predicts x0 (or another target) from (xt, t).
        device (torch.device): Computation device.
    """

    def __init__(
        self,
        sde: object,
        model_fn: Callable = None,
        device: torch.device = None,
    ):
        """
        Initializes the ODE solver.

        Args:
            sde (object): An instance of an SDE class (e.g., SB_VPSDE, SB_VESDE).
            model_fn (Callable, optional): A function that takes (x_t, t) and returns
                                           a prediction (e.g., x0).
            device (torch.device, optional): The device to run computations on.
        """
        super().__init__(sde, model_fn, device)

    def sampling(self,
                 x,
                 num_step: int = 5,
                 skip_type: str = "time_uniform",
                 t_max: float = 1.0,
                 t_min: float = 0.0):
        """
        Perform the ODE sampling process with specified discretization.

        Iteratively applies the probability flow ODE dynamics from `t_max` to `t_min`
        using the provided `model_fn`.

        Args:
            x (torch.Tensor): Initial state (e.g., terminal observation `x1`).
            num_step (int): Number of discretization steps for the solver.
            skip_type (str): Time step spacing strategy ('time_uniform', 'time_quadratic').
            t_max (float): Maximum time (start of reverse process, default 1.0).
            t_min (float): Minimum time (end of reverse process, default 0.0).

        Returns:
            tuple: (final_sample, backward_trajectory_stack, predicted_x0_stack)
                - final_sample (torch.Tensor): The final generated sample (x0 estimate).
                - backward_trajectory_stack (torch.Tensor): Stack of intermediate xt states.
                - predicted_x0_stack (torch.Tensor): Stack of x0 predictions at each step.
        """
        x = x.to(self.device, non_blocking=True)

        xs = [x]  # Store the trajectory of states
        pred_x0s = []  # Store the predictions of x0

        # Generate time steps for the reverse process
        # Note: Corrected `NFE` to `num_step` to match function signature
        timesteps = self.get_time_steps(skip_type, t_max, t_min, num_step)

        # Iterate through time steps in reverse order
        # Note: Corrected loop to use `num_step` instead of undefined `NFE`
        for i in range(0, num_step):
            t, t_prev = timesteps[i], timesteps[i + 1]

            # Use the model to predict x0 from the current state xt and time t
            pred_x0 = self.model_fn(x, t)

            # Apply one step of the probability flow ODE dynamics
            # Note: Corrected method name from `first_order_ODE_Solver` to match SDE class method
            x = self.sde.fisrt_order_ode_sampling(
                x1=x, xt=x, x0=pred_x0, t=t,
                t_prev=t_prev)  # Note: Typo in method name

            # Store intermediate results
            pred_x0s.append(pred_x0)
            xs.append(x)

        # Move results back to CPU for storage/return
        xs = [x.to("cpu", non_blocking=True) for x in xs]
        pred_x0s = [
            pred_x0.to("cpu", non_blocking=True) for pred_x0 in pred_x0s
        ]

        # Helper function to stack and flip the trajectory for consistent ordering
        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1),
                                              dims=(1, ))

        return x, stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
