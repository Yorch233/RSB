# RSB/modeling_rsb.py
import os

import torch
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_model
from torch import nn

from RSB.backbone import BackboneRegister
from RSB.common.config import Config, read_config_from_yaml
from RSB.dataset.ComplexSpecDatatet import STFTUtil
from RSB.sdes import SB_VESDE, SB_VPSDE


class RSB(nn.Module):
    """
    RSB for inverse problems.
    """

    def __init__(self,
                 backbone: str = 'ncsnpp_base',
                 training_method: str = 'none',
                 training_target: str = 'data',
                 loss_weight_type: str = 'constant',
                 bridge_type: str = 'VE',
                 device: str = 'cuda',
                 **ignored_kwargs):
        """
        Initialize the RSB model.

        Args:
            backbone (str): Name of the netowrk backbone architecture.
            training_method (str): Training approach ('none', 'regularization', etc.).
            training_target (str): Target for training ('data', etc.).
            loss_weight_type (str): Type of loss weighting ('constant', etc.).
            bridge_type (str): Type of Schrödinger bridge (e.g. 'VE' for Variance Exploding).
            device (str): Device to run the model on ('cuda' or 'cpu').
            **ignored_kwargs: Additional keyword arguments that are ignored.
        """
        super().__init__()
        self.training_method = training_method
        self.device = device

        # Initialize generative model backbone
        # Input channels depend on whether conditioning is used
        if 'condition' not in training_method:
            input_channels = 4  # Standard input (no additional conditioning)
        else:
            input_channels = 6  # Additional channels for conditioning information
        self.generator = BackboneRegister.fetch(backbone)(
            input_channels=input_channels)

        # Initialize Stochastic Differential Equation (SDE) based on bridge type
        sde_cls = None
        if bridge_type == 'VP':
            sde_cls = SB_VPSDE  # Variance Preserving SDE
        elif bridge_type == 'VE':
            sde_cls = SB_VESDE  # Variance Exploding SDE

        self.sde = sde_cls(training_target=training_target,
                           loss_weight_type=loss_weight_type,
                           device=self.device)
        self.to(self.device)

    def forward(self, x, t, cond=[]):
        """
        Forward pass through the generator network.

        Args:
            x (torch.Tensor): Input tensor (typically noisy data).
            t (torch.Tensor): Time step tensor.
            cond (list): List of conditioning tensors.

        Returns:
            torch.Tensor: Output from the generator network.
        """
        input = torch.cat([x] + cond,
                          dim=1)  # Concatenate input with conditioning
        return self.generator(input, t)

    def sampling(
        self,
        audio,
        predictive_fn=None,
        num_step=5,
        solver="SDE",
        skip_type="time_uniform",
    ):
        """
        Perform sampling/inference to generate enhanced audio.

        Args:
            audio (torch.Tensor): Input audio waveform to be enhanced.
            predictive_fn (callable, optional): Function to predict initial state from observation.
            num_step (int): Number of sampling steps.
            solver (str): Type of solver ('SDE' or 'ODE').
            skip_type (str): Time step scheduling ('time_uniform', etc.).

        Returns:
            tuple: (enhanced_audio, intermediate_states, predicted_x0s)
                - enhanced_audio: Final generated audio waveform
                - intermediate_states: List of intermediate states during sampling
                - predicted_x0s: List of predicted initial states
        """
        # Convert audio to STFT representation
        y, invert_fn = STFTUtil.to_stft(audio, device=self.device)

        condition = [y]  # Start with observation as condition

        # Apply predictive function if specified
        if self.training_method not in ['none', 'regularization']:
            x_star = predictive_fn(y)  # Predict initial state from observation
            if 'optimal' in self.training_method:
                y = x_star  # Use prediction as target
            if 'condition' in self.training_method:
                condition.append(
                    x_star)  # Add prediction as additional conditioning

        global count
        count = 0  # Counter for tracking number of model evaluations

        @torch.no_grad()
        def pred_x0_fn(xt, timestep):
            """Prediction function for x0 estimation.
            
            Args:
                xt (torch.Tensor): Current state tensor at time t
                timestep (float): Current time step value
                
            Returns:
                torch.Tensor: Predicted initial state (x0)
            """
            global count

            # Create time step tensor for batch
            timestep = torch.full((xt.shape[0], ),
                                  timestep,
                                  device=self.device,
                                  dtype=torch.float32)

            # Forward pass through generator
            out = self.forward(xt, timestep, cond=condition)
            count = count + 1  # Increment evaluation counter

            # Compute predicted initial state using SDE
            return self.sde.compute_pred_x0(
                xt=xt,
                t=timestep,
                x1=y,  # Target/final state
                net_out=out)

        # Select appropriate solver based on configuration
        if solver == "SDE":
            sampler = self.sde.get_sde_solver(model_fn=pred_x0_fn)
        elif solver == "ODE":
            sampler = self.sde.get_ode_solver(model_fn=pred_x0_fn)
        else:
            raise NotImplementedError(f"Unsupported sampling method: {solver}")

        # Perform the actual sampling process
        x, xs, pred_x0s = sampler.sampling(x=y,
                                           num_step=num_step,
                                           skip_type=skip_type)

        # Convert back from STFT to audio waveform
        audio = invert_fn(x)

        # Verify that the number of evaluations matches expected steps
        assert count == num_step, "num_step count mismatch between expected and actual evaluations."

        return audio, xs, pred_x0s
