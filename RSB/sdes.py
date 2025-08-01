import torch
from RSB.solver import ODESolver, SDESolver


def unsqueeze_xdim(*args, xdim):
    """
    Adds singleton dimensions to input tensors to match the length specified by `xdim`.
    Supports a variable number of inputs and returns corresponding outputs.

    Args:
        *args: Input tensors or lists/tuples of tensors.
        xdim (tuple): Target dimensions (used to determine number of dims to add).

    Returns:
        Single tensor or tuple of tensors: Tensors with expanded dimensions.
    """
    results = []
    for z in args:
        if isinstance(z, torch.Tensor):
            # Create indexing tuple like (..., None, None, ...) for xdim length
            bc_dim = (..., ) + (None, ) * len(xdim)
            results.append(z[bc_dim])
        elif isinstance(z, (list, tuple)):
            # Recursively apply to elements of lists/tuples
            results.append([unsqueeze_xdim(item, xdim=xdim) for item in z])
        else:
            results.append(z)
    # Return single item or tuple based on number of inputs
    return results[0] if len(results) == 1 else tuple(results)


class SB_SDE:
    """
    Base class for Schrödinger Bridge Stochastic Differential Equations (SDEs).
    Defines the core mathematical framework for VP (Variance Preserving) and VE (Variance Exploding) SDEs
    used in the RSB (Refined Schrödinger Bridge) model.
    """

    def __init__(self,
                 training_target="data",
                 loss_weight_type="constant",
                 device="cpu"):
        """
        Initializes the SDE base class.

        Args:
            training_target (str): Specifies the target for the neural network during training.
                                   Options: "data" (predicts x0), "noise" (predicts noise),
                                   "score" (predicts scaled score), "vector" (predicts x1-x0).
            loss_weight_type (str): Type of weighting applied to the loss function.
                                    Options: "constant", "snr", "min_snr_<value>".
            device (str): The device ('cpu' or 'cuda') on which tensors are computed.
        """
        self.device = device
        self.training_target = training_target
        self.loss_weight_type = loss_weight_type

    def T(self, t):
        """
        Returns the terminal time (often normalized to 1).

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Tensor of ones with the same shape as `t`.
        """
        return torch.ones_like(t, device=self.device)

    def marginal_alpha(self, t):
        """
        Calculates the marginal alpha coefficient at time `t`.
        This represents the signal coefficient (how the original signal scales over time).
        Must be implemented by subclasses (VP/VE).

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Alpha values at time `t`.
        """
        raise NotImplementedError()

    def marginal_alpha_bar(self, t):
        """
        Calculates the marginal alpha_bar coefficient at time `t`.
        Often used in the context of bridging between initial and final states.

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Alpha_bar values at time `t`.
        """
        return self.marginal_alpha(t) / self.marginal_alpha(self.T(t))

    def marginal_sigma(self, t):
        """
        Calculates the marginal sigma (standard deviation) at time `t`.

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Sigma values at time `t`.
        """
        return torch.sqrt(self.marginal_sigma_square(t))

    def marginal_sigma_bar(self, t):
        """
        Calculates the marginal sigma_bar (standard deviation) at time `t`.

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Sigma_bar values at time `t`.
        """
        return torch.sqrt(self.marginal_sigma_bar_square(t))

    def marginal_sigma_square(self, t):
        """
        Calculates the marginal sigma squared (variance) at time `t`.
        Must be implemented by subclasses (VP/VE).

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Sigma squared values at time `t`.
        """
        raise NotImplementedError()

    def marginal_sigma_bar_square(self, t):
        """
        Calculates the marginal sigma_bar squared (variance) at time `t`.

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Sigma_bar squared values at time `t`.
        """
        return self.marginal_sigma_square(
            self.T(t)) - self.marginal_sigma_square(t)

    def q_sample(self, t, x0, x1):
        """
        Samples from the conditional distribution q(x_t | x_0, x_1).
        This generates intermediate noisy states given the initial (x0) and final (x1) states.

        Args:
            t (torch.Tensor): Time steps.
            x0 (torch.Tensor): Initial condition tensor (e.g., clean data).
            x1 (torch.Tensor): Terminal condition tensor (e.g., noisy/observed data).

        Returns:
            torch.Tensor: Sampled x_t values (intermediate noisy states).
        """
        _, *xdim = x0.shape
        # Calculate weights and variance for the sampling distribution
        w_x0 = self.marginal_alpha(t) * self.marginal_sigma_bar_square(
            t) / self.marginal_sigma_square(self.T(t))
        w_x1 = self.marginal_alpha_bar(t) * self.marginal_sigma_square(
            t) / self.marginal_sigma_square(self.T(t))
        var = self.marginal_alpha(t)**2 * self.marginal_sigma_bar_square(
            t) * self.marginal_sigma_square(t) / self.marginal_sigma_square(
                self.T(t))
        # Expand dimensions for broadcasting
        w_x0, w_x1, var = unsqueeze_xdim(w_x0, w_x1, var, xdim=xdim)
        # Compute mean and sample
        mean = w_x1 * x1 + w_x0 * x0
        x_t = mean + var.sqrt() * torch.randn_like(mean)
        return x_t

    def fisrt_order_sde_sampling(
            self, xt, x0, t,
            t_prev):  # Note: Typo in 'fisrt' (should be 'first')
        """
        Performs a first-order Euler-Maruyama step for SDE sampling.
        Computes the previous state x_{t_prev} given the current state xt, initial state x0, and time steps.

        Args:
            xt (torch.Tensor): Current state tensor at time `t`.
            x0 (torch.Tensor): Initial condition tensor (predicted or known).
            t (torch.Tensor): Current time step.
            t_prev (torch.Tensor): Previous time step.

        Returns:
            torch.Tensor: Estimated state x_{t_prev}.
        """
        _, *xdim = x0.shape
        # Calculate weights and variance for the transition
        w_xt = self.marginal_alpha(t_prev) * self.marginal_sigma_square(
            t_prev) / (self.marginal_alpha(t) * self.marginal_sigma_square(t))
        w_x0 = self.marginal_alpha(t_prev) * (
            1 -
            self.marginal_sigma_square(t_prev) / self.marginal_sigma_square(t))
        var = self.marginal_alpha(t_prev)**2 * self.marginal_sigma_square(
            t_prev) * (1 - self.marginal_sigma_square(t_prev) /
                       self.marginal_sigma_square(t))
        # Expand dimensions for broadcasting
        w_x0, w_xt, var = unsqueeze_xdim(w_x0, w_xt, var, xdim=xdim)
        # Compute deterministic part of the update
        x_prev = w_xt * xt + w_x0 * x0
        # Add stochastic (diffusion) part if not at the final step (t_prev > 0)
        if t_prev > 0:
            x_prev = x_prev + var.sqrt() * torch.randn_like(x_prev)
        return x_prev

    def fisrt_order_ode_sampling(
            self, x1, xt, x0, t,
            t_prev):  # Note: Typo in 'fisrt' (should be 'first')
        """
        Performs a first-order step for ODE sampling.
        Computes the previous state x_{t_prev} given the current state xt, initial state x0, final state x1, and time steps.

        Args:
            x1 (torch.Tensor): Terminal condition tensor (e.g., noisy/observed data).
            xt (torch.Tensor): Current state tensor at time `t`.
            x0 (torch.Tensor): Initial condition tensor (predicted or known).
            t (torch.Tensor): Current time step.
            t_prev (torch.Tensor): Previous time step.

        Returns:
            torch.Tensor: Estimated state x_{t_prev}.
        """
        _, *xdim = x0.shape
        # Calculate weights for the deterministic ODE flow
        w_xt = self.marginal_alpha(t_prev) * self.marginal_sigma(
            t_prev) * self.marginal_sigma_bar(t_prev) / (
                self.marginal_alpha(t) * self.marginal_sigma_square(t) *
                self.marginal_sigma_bar(t))
        w_x0 = self.marginal_alpha(t_prev) * self.marginal_sigma_square(
            self.T(t)) * (
                self.marginal_sigma_bar_square(t) -
                (self.marginal_sigma_bar(t) * self.marginal_sigma(t_prev) *
                 self.marginal_sigma_bar(t_prev)) / self.marginal_sigma(t))
        w_x1 = self.marginal_alpha(t_prev) / (self.marginal_alpha(
            self.T(t)) * self.marginal_sigma_square(self.T(t))) * (
                self.marginal_sigma_square(t) -
                (self.marginal_sigma(t) * self.marginal_sigma(t_prev) *
                 self.marginal_sigma_bar(t_prev)) / self.marginal_sigma_bar(t))
        # Expand dimensions for broadcasting
        w_x0, w_xt, w_x1 = unsqueeze_xdim(w_x0, w_xt, w_x1, xdim=xdim)
        # Compute the deterministic update
        x_prev = w_xt * xt + w_x0 * x0 + w_x1 * x1
        return x_prev

    def compute_pred_x0(self, xt, t, x1=None, net_out=None):
        """
        Computes the predicted initial state x0 from the noisy state xt and the network output.

        Args:
            xt (torch.Tensor): Noisy state tensor at time `t`.
            t (torch.Tensor): Time steps.
            x1 (torch.Tensor, optional): Terminal condition tensor (required for 'vector' target).
            net_out (torch.Tensor): Output from the neural network (interpretation depends on `training_target`).

        Returns:
            torch.Tensor: Predicted initial state x0.
        """
        assert net_out is not None, "net_out should be provided for prediction"
        if self.training_target == "data":
            # Network directly predicts x0
            pred_x0 = net_out
        elif self.training_target == "noise":
            # Network predicts noise; reconstruct x0
            alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)
            _, *xdim = xt.shape
            alpha_t = unsqueeze_xdim(alpha_t, xdim)
            sigma_t = unsqueeze_xdim(sigma_t, xdim)
            pred_x0 = (xt - sigma_t * net_out) / alpha_t
        elif self.training_target == "score":
            # Network predicts scaled score; reconstruct x0
            alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)
            _, *xdim = xt.shape
            alpha_t = unsqueeze_xdim(alpha_t, xdim)
            sigma_t = unsqueeze_xdim(sigma_t, xdim)
            pred_x0 = (xt - alpha_t * sigma_t * net_out) / alpha_t
        elif self.training_target == "vector":
            # Network predicts x1 - x0; reconstruct x0
            assert x1 is not None, "x1 should be provided for vector training target"
            pred_x0 = x1 - net_out
        else:
            raise NotImplementedError(
                f"Training target {self.training_target} is not implemented.")
        return pred_x0

    def compute_weight(self, t):
        """
        Computes the loss weighting factor based on the time step and configuration.

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Loss weights for each time step.
        """
        if self.training_target == "data":
            if self.loss_weight_type == 'constant':
                loss_weight = torch.ones_like(t, device=self.device)
            elif self.loss_weight_type == 'snr':
                # Weight by Signal-to-Noise Ratio
                loss_weight = torch.exp(self.marginal_logSNR(t))
            elif self.loss_weight_type.startswith("min_snr_"):
                # Weight by min(SNR, k) to stabilize training
                k = float(self.loss_weight_type.split('min_snr_')[-1])
                snr = torch.exp(self.marginal_logSNR(t))
                loss_weight = torch.stack([snr, k * torch.ones_like(t)],
                                          dim=1).min(dim=1)[0]
        else:
            # Default constant weight for other targets
            loss_weight = torch.ones_like(t, device=self.device)
        return loss_weight

    def compute_label(self, xt, t, x0, x1):
        """
        Computes the target label for the neural network during training.

        Args:
            xt (torch.Tensor): Noisy state tensor at time `t`.
            t (torch.Tensor): Time steps.
            x0 (torch.Tensor): Ground truth initial state.
            x1 (torch.Tensor): Terminal condition tensor (required for 'vector' target).

        Returns:
            torch.Tensor: Target label for the neural network.
        """
        if self.training_target == "data":
            # Target is the initial state itself
            label = x0
        elif self.training_target == "noise":
            # Target is the noise added to x0 to get xt
            xt = xt.detach()  # Detach xt to avoid gradients flowing through it
            alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)
            _, *xdim = x0.shape
            alpha_t, sigma_t = unsqueeze_xdim(alpha_t, sigma_t, xdim)
            label = (xt - x0 * alpha_t) / sigma_t
        elif self.training_target == "score":
            # Target is the scaled score of the data distribution
            xt = xt.detach()
            alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)
            _, *xdim = x0.shape
            alpha_t, sigma_t = unsqueeze_xdim(alpha_t, sigma_t, xdim)
            label = (xt - x0 * alpha_t) / alpha_t / sigma_t
        elif self.training_target == "vector":
            # Target is the difference between final and initial states
            label = x1 - x0
        else:
            raise NotImplementedError(
                f"Training target {self.training_target} is not implemented.")
        return label

    def get_sde_solver(self, model_fn):
        """
        Instantiates an SDE solver using this SDE and a given model function.

        Args:
            model_fn (callable): Function that takes (xt, t) and returns a prediction (e.g., x0).

        Returns:
            SDESolver: An instance of the SDE solver.
        """
        return SDESolver(self, model_fn=model_fn)

    def get_ode_solver(self, model_fn):
        """
        Instantiates an ODE solver using this SDE and a given model function.

        Args:
            model_fn (callable): Function that takes (xt, t) and returns a prediction (e.g., x0).

        Returns:
            ODESolver: An instance of the ODE solver.
        """
        return ODESolver(self, model_fn=model_fn)


# --- Subclasses for specific SDE types ---


class SB_VPSDE(SB_SDE):
    """
    Variance Preserving SDE (VP-SDE) subclass.
    In this SDE, the signal coefficient alpha(t) decays over time, while the noise level sigma(t) increases,
    but the total variance remains controlled.
    """

    def __init__(self,
                 beta0=0.01,
                 beta1=20,
                 c=0.3,
                 training_target="data",
                 loss_weight_type="constant",
                 device="cpu"):
        """
        Initializes the VP-SDE with specific parameters.

        Args:
            beta0 (float): Initial diffusion coefficient.
            beta1 (float): Final diffusion coefficient.
            c (float): Scaling factor for sigma.
            training_target (str): Target for training (passed to base class).
            loss_weight_type (str): Loss weighting type (passed to base class).
            device (str): Computation device (passed to base class).
        """
        super().__init__(
            training_target=training_target,
            loss_weight_type=loss_weight_type,
            device=device,
        )
        self.beta0 = torch.tensor(beta0).to(self.device)
        self.beta1 = torch.tensor(beta1).to(self.device)
        self.c = torch.tensor(c).to(self.device)

    def marginal_alpha(self, t):
        """
        Calculates alpha(t) for VP-SDE: exp(-0.5 * integral(beta(s) ds)).

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Alpha values.
        """
        return torch.exp(-0.5 * (self.beta0 * t + 0.5 *
                                 (self.beta1 - self.beta0) * t**2))

    def marginal_sigma_square(self, t):
        """
        Calculates sigma^2(t) for VP-SDE: c * (exp(integral(beta(s) ds)) - 1).

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Sigma squared values.
        """
        return self.c * torch.expm1(self.beta0 * t + 0.5 *
                                    (self.beta1 - self.beta0) * t**2)


class SB_VESDE(SB_SDE):
    """
    Variance Exploding SDE (VE-SDE) subclass.
    In this SDE, the signal coefficient alpha(t) is constant (usually 1),
    and the noise level sigma(t) grows exponentially over time.
    """

    def __init__(self,
                 c=0.4,
                 k=2.6,
                 training_target="data",
                 loss_weight_type="constant",
                 device="cpu"):
        """
        Initializes the VE-SDE with specific parameters.

        Args:
            c (float): Scaling factor for sigma.
            k (float): Base for exponential growth of sigma.
            training_target (str): Target for training (passed to base class).
            loss_weight_type (str): Loss weighting type (passed to base class).
            device (str): Computation device (passed to base class).
        """
        super().__init__(training_target=training_target,
                         loss_weight_type=loss_weight_type,
                         device=device)
        self.k = torch.tensor(k).to(self.device)
        self.c = torch.tensor(c).to(self.device)

    def marginal_alpha(self, t):
        """
        Calculates alpha(t) for VE-SDE: constant (1).

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Alpha values (all ones).
        """
        return torch.ones_like(t, device=self.device)

    def marginal_sigma_square(self, t):
        """
        Calculates sigma^2(t) for VE-SDE: c * (k^(2t) - 1) / (2 * ln(k)).

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Sigma squared values.
        """
        return self.c * (self.k**(2 * t) - 1) / (2 * torch.log(self.k))
