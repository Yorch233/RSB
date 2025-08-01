# RSB/trainer.py
import logging
import os
import shutil
import time
from datetime import datetime
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch import optim
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Log to console
)
logger = get_logger(__name__)  # Get logger for this module

# Import project-specific modules
from RSB.backbone import BackboneRegister
from RSB.common.config import Config
from RSB.dataset.ComplexSpecDatatet import ComplexSpec, STFTUtil
from RSB.evaluate.registry import MetricRegister
from RSB.modeling_rsb import RSB  # Assuming the model class is named RSB


class RSB_Trainer():
    """
    Trainer class for the RSB model.
    Handles training loop, validation, checkpointing, and evaluation.
    """

    def __init__(self, config: Config):
        """
        Initializes the RSB trainer.

        Args:
            config (Config): Configuration object containing training parameters.
        """
        self.config = config
        set_seed(self.config.seed)  # Set random seed for reproducibility

        # --- Setup run identifiers and paths ---
        self.config.run_name = self.config.get(
            "run_name",
            f"RSB_{self.config.bridge_type}_{datetime.now().strftime('%m%d%H%M')}"
        )
        self.config.run_id = self.config.get(
            "run_id", f"abc{datetime.now().strftime('%m%d%H%M%S')}")
        self.config.output_path = self.config.get(
            "output_path",
            os.path.join(self.config.run_dir, self.config.run_name))
        self.output_path = self.config.output_path
        self.checkpoint_path = os.path.join(self.output_path, "checkpoints")
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # --- Initialize Accelerator for distributed training/mixed precision ---
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(log_with=self.config.get('log_with'),
                                       kwargs_handlers=[ddp_kwargs])
        self.device = self.accelerator.device

        # --- Initialize experiment tracking (e.g., with Weights & Biases) ---
        self.accelerator.init_trackers(
            "RSB",  # Project name
            config=self.config.dict(),  # Log config parameters
            init_kwargs={
                "wandb": {
                    "name": self.config.run_name,
                    "id": self.config.run_id,
                    "resume": self.config.get("resume", False)
                }
            })

        # --- Initialize the RSB model ---
        self.RSB = RSB(backbone=self.config.generative_backbone,
                       training_method=self.config.training_method,
                       training_target=self.config.training_target,
                       loss_weight_type=self.config.loss_weight_type,
                       bridge_type=self.config.bridge_type,
                       device=self.device)

        # --- Initialize optimizer ---
        if self.config.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.RSB.parameters(),
                                        lr=self.config.learning_rate)
        elif self.config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.RSB.parameters(),
                                         lr=self.config.learning_rate)
        else:
            raise NotImplementedError(
                f'Optimizer {self.config.optimizer} not supported yet!')

        # --- Setup data loaders ---
        self.train_dataloader = self.get_dataloader('train')
        self.valid_dataloader = self.get_dataloader('valid')

        # --- Prepare model, optimizer, and dataloaders with Accelerator ---
        # This handles device placement, DDP wrapping, mixed precision, etc.
        self.RSB, self.optimizer, self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(
            self.RSB, self.optimizer, self.train_dataloader,
            self.valid_dataloader)

        # --- Setup Exponential Moving Average (EMA) for model weights ---
        if self.config.get('ema_rate') is not None:
            self.ema = EMA(self.RSB.parameters(), decay=self.config.ema_rate)
            self.ema.to(self.device)
            # Register EMA state for automatic saving/loading with Accelerator
            self.accelerator.register_for_checkpointing(self.ema)
        else:
            self.ema = None

        # --- Initialize predictive model (if required by training method) ---
        if self.config.training_method != 'none':
            # Fetch the predictive backbone (e.g., for estimating x0 from x1)
            self.preditive_model = BackboneRegister.fetch(
                self.config.predictive_backbone)(discriminative=True)
            self.preditive_model.to(self.device)
            # Load pretrained weights for the predictive model
            predictive_checkpoint_path = os.path.join(
                'pretrained_predictive_model', self.config.dataset,
                f'{self.config.predictive_backbone}.pt')
            if not os.path.exists(predictive_checkpoint_path):
                raise RuntimeError(
                    f"The discriminator checkpoint at path '{predictive_checkpoint_path}' does not exist. "
                    f"Please ensure the path is correct or the discriminator has been pre-trained."
                )
            checkpoint = torch.load(predictive_checkpoint_path,
                                    map_location='cpu')
            self.preditive_model.load_state_dict(checkpoint)
            self.preditive_model.to(self.device)
            self.preditive_model.eval()  # Set to evaluation mode

        # --- Setup loss reduction operation ---
        if self.config.get("reduction", 'mean') == 'mean':
            self._reduce_op = torch.mean
        else:
            self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(
                *args, **kwargs)

        # --- Initialize L1 loss function for time-domain loss ---
        self._l1_loss = torch.nn.L1Loss(reduction='sum')

    def save_state(self):
        """
        Saves the current training state (model, optimizer, RNG states, EMA).
        Manages checkpoint limits by removing old checkpoints.
        """
        if self.accelerator.is_main_process:  # Only save on main process in distributed setting
            # Manage checkpoint limit
            if self.config.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.checkpoint_path)
                checkpoints = [
                    d for d in checkpoints if d.startswith("checkpoint")
                ]
                # Sort by step number
                checkpoints = sorted(checkpoints,
                                     key=lambda x: int(x.split("_steps=")[1]))
                if len(checkpoints) >= self.config.checkpoints_total_limit:
                    # Calculate how many to remove
                    num_to_remove = len(
                        checkpoints) - self.config.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(
                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                    )
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint_path = os.path.join(
                            self.checkpoint_path, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint_path
                                      )  # Remove old checkpoint directory

            # Save the state using Accelerator
            save_path = os.path.join(
                self.checkpoint_path,
                f"checkpoint_steps={self.config.num_steps}")
            self.accelerator.save_state(
                save_path)  # Saves optimizer, RNG, EMA (if registered)
            self.accelerator.save_model(self.RSB,
                                        save_path)  # Save model weights
            self.config.save(save_path)  # Save config file
            logger.info(f"Saved checkpoint to {save_path}")

    def load_state(self):
        """
        Loads the latest training state from checkpoints.
        """
        checkpoints = os.listdir(self.checkpoint_path)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        # Sort by step number to get the latest
        checkpoints = sorted(checkpoints,
                             key=lambda x: int(x.split("_steps=")[1]))
        if checkpoints:
            save_path = os.path.join(self.checkpoint_path, checkpoints[-1])
            self.accelerator.load_state(
                save_path)  # Load state using Accelerator
            logger.info(f"Loaded checkpoint from {save_path}")
        else:
            logger.warning("No checkpoints found to load.")

    def save_model(self):
        """
        Saves the final trained model weights and configuration.
        """
        self.accelerator.wait_for_everyone()  # Ensure all processes are synced
        if self.accelerator.is_main_process:
            self.accelerator.save_model(
                self.RSB, self.output_path)  # Save final model weights
            self.config.save(self.output_path)  # Save final config
            logger.info(
                f"Saved final model weights and config to {self.output_path}")

    def get_dataloader(self, subset: str = 'train'):
        """
        Creates a DataLoader for a given dataset subset.

        Args:
            subset (str): 'train' or 'valid'/'test'.

        Returns:
            torch.utils.data.DataLoader: Configured DataLoader.
        """
        dataset = ComplexSpec(
            config=self.config,
            dataset=self.config.dataset,
            subset=subset,
            # Shuffle only for training data
            shuffle_spec=(subset == 'train'),
            return_spec=True,  # Return spectrograms
            # Load pre-computed posterior mean if available
            load_posterior_mean=self.config.get("load_posterior_mean", True),
            dummy=self.config.dummy)  # Use dummy data if specified

        dataloader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,  # Speed up GPU transfer
            batch_size=self.config.batch_size,
            # Shuffle only for training data
            shuffle=(subset == 'train'))
        return dataloader

    def print(self, *args):
        """Wrapper for accelerator.print to ensure output on main process."""
        self.accelerator.print(*args)

    @torch.no_grad()  # Disable gradient computation for inference
    def x_star(self, x1):
        """
        Estimates the initial state x0 (x_star) from the terminal state x1 using the predictive model.

        Args:
            x1 (torch.Tensor): Terminal observation (e.g., noisy spectrogram).

        Returns:
            torch.Tensor: Estimated initial state (x_star).
        """
        return self.preditive_model(x1)

    def omega(self, t):
        """
        Defines the weighting function omega(t) for regularization.

        Args:
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Weight values, reshaped for broadcasting.
        """
        if self.config.get("regularization_weight", "quadratic") == 'quadratic':
            omega_t = t**2  # Quadratic weighting
        else:
            omega_t = t  # Linear weighting
        # Reshape for broadcasting with spectrogram dimensions (B, C, F, T)
        return omega_t[(..., ) + (None, ) * 3]

    def perturb(self, x0, x_star, t):
        """
        Applies perturbation for regularization training: x_hat_t = omega(t) * x_star + (1 - omega(t)) * x0.

        Args:
            x0 (torch.Tensor): True initial state.
            x_star (torch.Tensor): Estimated initial state.
            t (torch.Tensor): Time steps.

        Returns:
            torch.Tensor: Perturbed state x_hat_t.
        """
        x_hat_t = self.omega(t) * x_star + (1 - self.omega(t)) * x0
        return x_hat_t

    def step(self, batch):
        """
        Performs a single training/validation step.

        Args:
            batch: A batch of data from the dataloader (typically [x0, x1, x_star_optional]).

        Returns:
            tuple: (total_loss, dict_of_individual_losses)
        """
        x0, x1 = batch[0], batch[1]  # x0: clean spec, x1: noisy/terminal spec

        # Get x_star (estimated x0) if needed by the training method
        if self.config.training_method not in ['none']:
            # Use precomputed x_star from batch or compute it
            x_star = self.x_star(x1) if not self.config.get(
                "load_posterior_mean", True) else batch[2]

        # Sample random time steps for training
        timestep = torch.rand(x0.shape[0], device=x0.device) * (
            self.config.t_max - self.config.t_min) + self.config.t_min

        # Prepare conditioning inputs for the RSB model
        condition = [x1]  # Always condition on x1
        if self.config.training_method == 'regularization':
            # For regularization, perturb x0 using x_star and time
            x_hat_t = self.perturb(x0, x_star, timestep)
            x0 = x_hat_t  # Use perturbed x0 as the target initial state
        else:
            # For other methods, modify x1 or add x_star to conditions
            if 'optimal' in self.config.training_method:
                x1 = x_star  # Use estimated x0 as the target terminal state
            if 'condition' in self.config.training_method:
                condition.append(
                    x_star)  # Add x_star as additional conditioning

        # Sample xt from the conditional distribution q(xt|x0, x1)
        xt = self.RSB.sde.q_sample(t=timestep, x0=x0, x1=x1)

        # Forward pass through the RSB generator
        netout = self.RSB(x=xt, t=timestep, cond=condition)

        # Compute the target label for training (depends on training_target config)
        label = self.RSB.sde.compute_label(xt=xt, t=timestep, x0=x0, x1=x1)

        # --- Compute Losses ---

        # 1. Prediction loss (e.g., MSE between network output and target label)
        prediction_batch_loss = torch.square(
            torch.abs(netout -
                      label))  # Square of L2 norm of complex difference
        # Reduce loss per sample and apply SDE-specific weighting
        prediction_loss = torch.mean(
            self._reduce_op(prediction_batch_loss.reshape(
                prediction_batch_loss.shape[0], -1),
                            dim=-1) * self.RSB.sde.compute_weight(timestep))

        # 2. Time-domain L1 loss between reconstructed and true audio
        # Reconstruct x0 estimate from network output
        x0_hat = self.RSB.sde.compute_pred_x0(xt=xt,
                                              t=timestep,
                                              x1=x1,
                                              net_out=netout)
        # Convert spectrograms back to waveforms
        x_wav = STFTUtil.istft(x0.squeeze(1)).squeeze(
            1)  # Remove channel dim for istft
        x_hat_wav = STFTUtil.istft(x0_hat.squeeze(1)).squeeze(1)
        # Compute L1 loss in time domain
        time_loss = self._l1_loss(
            x_wav, x_hat_wav) / x_wav.shape[0]  # Average over batch

        # 3. Total loss
        total_loss = prediction_loss + time_loss * self.config.get(
            "time_loss_weight", 0)

        # Return total loss and individual components
        return total_loss, {
            "total_loss": total_loss,
            "prediction_loss": prediction_loss,
            "time_loss": time_loss
        }

    def evaluate_step(self, epoch, valid_dataloader):
        """
        Performs validation evaluation at the end of an epoch.

        Args:
            epoch (int): Current epoch number.
            valid_dataloader: DataLoader for validation data.
        """
        # Use EMA weights if available
        with self.ema.average_parameters() if self.ema else torch.no_grad():
            self.RSB.eval()  # Set model to evaluation mode
            num_valid_steps = 0
            valid_loss = {
                "total_loss": 0.0,
                "prediction_loss": 0.0,
                "time_loss": 0.0
            }

            # Iterate through validation data
            for batch in tqdm(
                    valid_dataloader,
                    desc="Validating",
                    disable=not self.accelerator.is_local_main_process):
                with torch.no_grad():  # Disable gradients for validation
                    _, losses = self.step(batch)  # Compute validation losses
                num_valid_steps += 1
                # Accumulate losses
                for key in losses.keys():
                    valid_loss[key] += losses[key].item()

            # Log average validation losses
            if num_valid_steps > 0:
                self.accelerator.log(
                    {
                        f'valid/{loss}': value / num_valid_steps
                        for loss, value in valid_loss.items()
                    },
                    step=self.config.num_steps)

            # --- Evaluate audio quality metrics ---
            result = self.evaluate_metrics(subset='valid',
                                           n_samples=50,
                                           num_step=20)
            pesq_score = float(result.get('PESQ', 0.0))
            sisdr_score = float(result.get('SI_SDR',
                                           -100.0))  # Default low value

            # --- Model checkpointing based on metrics ---
            # Save best model based on PESQ
            if self.config.best_pesq is None or pesq_score > self.config.best_pesq:
                self.config.best_pesq = pesq_score
                self.save_model()  # Save model weights and config
                logger.info(f"New best PESQ: {pesq_score:.4f}, model saved.")

            # Early stopping based on SI-SDR
            if self.config.best_sisdr is None or sisdr_score > self.config.best_sisdr:
                self.config.best_sisdr = sisdr_score
                self.config.early_stop_cnt = 0  # Reset counter
                logger.info(f"New best SI-SDR: {sisdr_score:.4f}")
            else:
                self.config.early_stop_cnt += 1  # Increment counter

        # --- Evaluate on test set (for monitoring, not used for saving) ---
        self.evaluate_metrics(subset='test', n_samples=5, num_step=20)

        # Log early stopping counter and epoch
        self.accelerator.log(
            {
                "valid/early_stop_cnt": self.config.early_stop_cnt,
                "epoch": epoch
            },
            step=self.config.num_steps)
        self.print(
            f"Epoch {epoch} - early_stop_cnt: {self.config.early_stop_cnt}, "
            f"Best PESQ: {self.config.best_pesq:.4f}, Best SI-SDR: {self.config.best_sisdr:.4f}"
        )

    def train_one_epoch(self, epoch, train_dataloader, valid_dataloader):
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            train_dataloader: DataLoader for training data.
            valid_dataloader: DataLoader for validation data.
        """
        # Iterate through training batches
        for batch in tqdm(train_dataloader,
                          desc=f"Epoch {epoch} Training",
                          disable=not self.accelerator.is_local_main_process):
            self.RSB.train()  # Set model to training mode
            self.optimizer.zero_grad()  # Clear gradients

            # Forward pass and loss computation
            train_loss, losses = self.step(batch)

            # Backward pass
            self.accelerator.backward(train_loss)  # Handles scaling for AMP
            self.optimizer.step()  # Update model parameters

            # Update EMA weights if used
            if self.ema is not None:
                self.ema.update()

            # --- Logging and Checkpointing ---
            # sync_gradients is True when gradients are synchronized across processes (after accumulation)
            if self.accelerator.sync_gradients:
                self.config.num_steps += 1  # Increment global step counter

                # Log training losses periodically
                if self.config.num_steps % self.config.log_steps == 0:
                    self.accelerator.log(
                        {
                            f'train/{loss}': value.item()
                            for loss, value in losses.items()
                        },
                        step=self.config.num_steps)

                # Save training state periodically
                if self.config.num_steps % self.config.save_state_steps == 0:
                    self.save_state()

        # Perform validation at the end of the epoch
        self.evaluate_step(epoch, valid_dataloader)
        logger.info(f"Epoch {epoch} finished")

        # Update epoch counter in config
        self.config.current_epoch = self.config.current_epoch + 1

    def train(self, is_resume=False):
        """
        Main training loop.

        Args:
            is_resume (bool): Whether to resume training from a checkpoint.
        """
        # Load checkpoint if resuming
        if is_resume:
            self.load_state()

        # Initialize or load training state variables from config
        self.config.best_pesq = self.config.get("best_pesq", None)
        self.config.best_sisdr = self.config.get("best_sisdr", None)
        self.config.current_epoch = self.config.get("current_epoch", 0)
        self.config.num_steps = self.config.get("num_steps", 0)
        self.config.early_stop_cnt = self.config.get("early_stop_cnt", 0)

        # Print configuration
        self.config.print()

        # --- Handle resuming from mid-epoch ---
        if self.config.num_steps > 0:
            # Skip batches already processed in the current epoch
            skipped_dataloader = self.accelerator.skip_first_batches(
                self.train_dataloader,
                self.config.num_steps % len(self.train_dataloader))
            logger.info(
                f"Resuming training from step {self.config.num_steps}. "
                f"Skipping {self.config.num_steps % len(self.train_dataloader)} batches."
            )
            # Run one epoch with the remaining batches
            self.train_one_epoch(self.config.current_epoch, skipped_dataloader,
                                 self.valid_dataloader)

        # --- Main training loop ---
        for epoch in range(self.config.current_epoch, self.config.num_epoch):
            self.train_one_epoch(epoch, self.train_dataloader,
                                 self.valid_dataloader)

            # Check for early stopping
            if self.config.early_stop_cnt >= self.config.patience:
                logger.info(
                    f"Early stopping triggered after {self.config.patience} epochs without improvement."
                )
                # Signal to Accelerator to stop (useful in multi-process settings)
                self.accelerator.set_trigger()
                if self.accelerator.check_trigger():
                    break  # Exit training loop

        # Finalize training (e.g., close trackers)
        self.accelerator.end_training()
        logger.info("Training completed.")

def evaluate_metrics(self, subset='valid', n_samples=0, solver='SDE', num_step=5):
        """
        Evaluates audio quality metrics (PESQ, SI-SDR) on a dataset subset.

        Args:
            subset (str): Dataset subset ('valid' or 'test').
            n_samples (int): Number of samples to evaluate (0 for all).
            solver (str): Sampling solver type ('SDE' or 'ODE').
            num_step (int): Number of sampling steps.

        Returns:
            dict: Average metric scores.
        """
        # Create dataset for raw waveform access (not spectrograms)
        dataset = ComplexSpec(self.config, dataset=self.config.dataset, subset=subset, return_raw=True)
        if n_samples < 1:
            n_samples = len(dataset) # Evaluate all samples
        else:
            n_samples = min(n_samples, len(dataset)) # Limit to dataset size

        # Fetch registered metrics (PESQ and SI-SDR)
        metrics = MetricRegister.fetch(['pesq', 'si_sdr'])
        result = {}

        # Evaluate metrics for each sample
        for i in tqdm(range(0, n_samples), desc=f'Evaluating Metrics {n_samples}/{len(dataset)}',
                       leave=False, ncols=200, disable=not self.accelerator.is_local_main_process):
            # Get clean (x) and noisy/terminal (y) waveforms
            x, y = dataset[i] # x: clean wav, y: noisy wav (as tensor)

            # Perform sampling/inference using the RSB model
            x_hat, _, _ = self.RSB.sampling(
                y, predictive_fn=self.x_star, num_step=num_step, solver=solver
            ) # x_hat: enhanced waveform

            # Convert tensors to numpy arrays for metric calculation
            clean_sig = x.cpu().squeeze().numpy() # Remove batch dim
            enhanced_sig = x_hat.type(torch.float32).cpu().squeeze().numpy()

            # Compute metrics for the sample
            for metric_name, metric_class in metrics.items():
                try:
                    metric_res = metric_class.compute(
                        ref_wav=clean_sig, deg_wav=enhanced_sig, sample_rate=self.config.sample_rate)
                    # Accumulate metric values
                    for item_key, item_value in metric_res.items():
                        if item_key not in result:
                            result[item_key] = 0.0
                        result[item_key] += item_value
                except Exception as e:
                    logger.warning(f"Error computing metric for sample {i} with {metric_name}: {e}")
                    # Add zero or skip? Here we assume metric_res keys are consistent.

        # Calculate average metric scores
        if n_samples > 0:
            result = {k: v / n_samples for k, v in result.items()}
        else:
            result = {k: 0.0 for k in result.keys()} # Avoid division by zero

        # Log metrics using Accelerator
        self.accelerator.log(
            {f'{subset}/{name}': metric for name, metric in result.items()},
            step=self.config.num_steps)

        # Print results on main process
        pesq_avg = result.get('PESQ', 0.0)
        sisdr_avg = result.get('SI_SDR', 0.0)
        self.print(f'{subset} Evaluation ({n_samples} samples) - '
                   f'Avg PESQ: {pesq_avg:.4f}, Avg SI-SDR: {sisdr_avg:.4f}')

        return result
