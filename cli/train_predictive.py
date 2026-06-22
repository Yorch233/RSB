import argparse
import logging
import os
import shutil
from datetime import datetime
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from RSB.backbone import BackboneRegister
from RSB.common.config import Config, read_config_from_yaml
from RSB.dataset.ComplexSpecDatatet import ComplexSpec, STFTUtil
from RSB.evaluate import MetricRegister
from tqdm import tqdm

def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing configuration parameters for training.

    Arguments:
        dataset (str): Name of the dataset to use for training and validation (default: "vocebank+demand").
        learning_rate (float): Initial learning rate for the Adam optimizer (default: 1e-4).
        batch_size (int): Number of samples per batch during training (default: 16).
        predictive_backbone (str): Architecture of the predictive backbone model (default: "ncsnpp_base").
        checkpoint_path (str): Path to the directory containing checkpoints for resuming training (default: "").
        patience (int): Number of epochs with no improvement after which training will be stopped early (default: 20).
        log_steps (int): Interval at which to log training metrics (default: 100).
        resume (bool): If set, resume training from the latest checkpoint (default: False).
    """
    # Parse command-line arguments with defaults from config
    parser = argparse.ArgumentParser(
        description="Train a predictive model for speech enhancement using the DisperSE framework."
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=default_config.get('dataset', 'vocebank+demand'),
        help='Name of the dataset to use for training and validation (e.g., vocebank+demand).'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=default_config.get('learning_rate', 1e-4),
        help='Initial learning rate for the Adam optimizer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=default_config.get('batch_size', 16),
        help='Number of samples per batch during training.'
    )
    parser.add_argument(
        '--predictive_backbone',
        type=str,
        default=default_config.get('predictive_backbone', 'ncsnpp_base'),
        help='Architecture of the predictive backbone model (e.g., ncsnpp_base, ncsnpp_large).'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default="",
        help='Path to the directory containing checkpoints for resuming training. If provided, --resume is implied.'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=default_config.get('patience', 20),
        help='Number of epochs with no improvement in validation loss after which training will be stopped early.'
    )
    parser.add_argument(
        '--log_steps',
        type=int,
        default=default_config.get('log_steps', 100),
        help='Interval (in training steps) at which to log training metrics (e.g., loss).'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='If set, resume training from the latest checkpoint found in the specified --checkpoint_path or default output directory.'
    )
    
    args = parser.parse_args()
    return args

class PredictiveTrainer:
    """Trainer class for predictive model training workflow"""

    def __init__(self, config: Config):
        """
        Initialize predictive trainer with configuration
        Args:
            config: Configuration object containing training parameters
        """
        self.config = config
        # Generate unique run name using current timestamp (MMDDHHMM format)
        self.config.run_name = f'Predictive_{datetime.now().strftime("%m%d%H%M")}'
        # Define output path with fallback to run_dir if not specified
        self.config.output_path = self.config.get(
            "output_path",
            os.path.join(self.config.run_dir, self.config.run_name))
        # Generate unique run ID with higher precision timestamp (MMDDHHMMSS format)
        self.config.run_id = self.config.get(
            "run_id", f"predictive_{datetime.now().strftime('%m%d%H%M%S')}")
        # Checkpoint directory path under main output path
        self.checkpoint_path = os.path.join(self.config.output_path,
                                            "checkpoints")
        # Create output directories if not exist
        os.makedirs(self.config.output_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        # Configure DDP with no unused parameter checking
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False)
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(log_with=self.config.log_with,
                                       kwargs_handlers=[ddp_kwargs])
        # Initialize logger from accelerate
        self.logger = get_logger(__name__)
        # Initialize logging trackers (e.g., wandb) with run metadata
        self.accelerator.init_trackers("DisperSE_Predictive",
                                       config=config,
                                       init_kwargs={
                                           "wandb": {
                                               "name": config.run_name,
                                               "id": config.run_id
                                           }
                                       })
        # Configure loss reduction operation based on config
        if self.config.get("reduction", 'mean') == 'mean':
            self._reduce_op = torch.mean
        else:
            self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(
                *args, **kwargs)
        # Initialize model from backbone registry with discriminative mode
        self.model = BackboneRegister.fetch(
            config.predictive_backbone)(discriminative=True)
        # Initialize Adam optimizer with configured learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.learning_rate)
        # Prepare training and validation data loaders
        self.train_loader = self.prepare_dataloader('train')
        self.valid_loader = self.prepare_dataloader('valid')
        # Prepare model and optimizer for distributed training
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer)

    def prepare_dataloader(self, subset):
        """
        Prepare DataLoader for specified subset (train/valid)
        Args:
            subset: Dataset subset ('train' or 'valid')
        Returns:
            Prepared DataLoader with accelerator integration
        """
        # Create dataset instance with subset-specific configuration
        dataset = ComplexSpec(self.config,
                              dataset=self.config.dataset,
                              subset=subset,
                              shuffle_spec=(subset == 'train'),
                              return_spec=True)
        # Prepare DataLoader with accelerator (handles distributed sampling)
        return self.accelerator.prepare(
            torch.utils.data.DataLoader(dataset,
                                        batch_size=self.config.batch_size,
                                        shuffle=(subset == 'train'),
                                        pin_memory=True))

    def train_epoch(self, epoch):
        """
        Execute one training epoch
        Args:
            epoch: Current epoch number
        Returns:
            Average training loss over the epoch
        """
        self.model.train()
        total_loss = 0
        self.config.current_epoch = epoch
        # Use accelerator's gradient accumulation context
        with self.accelerator.accumulate(self.model):
            for batch in tqdm(
                    self.train_loader,
                    desc=f'Epoch {epoch}',
                    disable=not self.accelerator.is_local_main_process):
                x, y = batch  # Unpack input and target from batch
                x_hat = self.model(y)  # Forward pass: predict x from y
                # Calculate prediction loss (MSE on magnitude)
                prediction_batch_loss = torch.square(torch.abs(x_hat - x))
                # Reduce loss across batch dimensions
                loss = torch.mean(
                    self._reduce_op(prediction_batch_loss.reshape(
                        prediction_batch_loss.shape[0], -1),
                                    dim=-1))
                # Backpropagation with accelerator
                self.accelerator.backward(loss)
                # Optimizer step and gradient reset
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
                # Update global step counter and log at intervals
                if self.accelerator.sync_gradients:
                    self.config.num_steps += 1
                    if self.config.num_steps % self.config.log_steps == 0:
                        self.accelerator.log({f'train/loss': loss.item()},
                                             step=self.config.num_steps)
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, epoch):
        """
        Evaluate model on validation set
        Args:
            epoch: Current epoch number
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        for batch in self.valid_loader:
            x, y = batch
            x_hat = self.model(y)  # Inference without gradient
            # Calculate validation loss (same as training)
            prediction_batch_loss = torch.square(torch.abs(x_hat - x))
            loss = torch.mean(
                self._reduce_op(prediction_batch_loss.reshape(
                    prediction_batch_loss.shape[0], -1),
                                dim=-1))
            total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def save_checkpoint(self):
        """
        Save training checkpoint with accelerator state
        (Handles model, optimizer, and RNG states)
        """
        if self.accelerator.is_main_process:  # Only main process saves
            # Clean up old checkpoints if total limit is configured
            if self.config.get('checkpoints_total_limit'):
                # Sort checkpoints by epoch number
                checkpoints = sorted(os.listdir(self.checkpoint_path),
                                     key=lambda x: int(x.split('_')[-1]))
                # Remove excess checkpoints beyond limit
                if len(checkpoints) > self.config.checkpoints_total_limit:
                    for chk in checkpoints[:-self.config.
                                           checkpoints_total_limit]:
                        shutil.rmtree(os.path.join(self.checkpoint_path, chk))
            # Create checkpoint path with current epoch
            save_path = os.path.join(
                self.checkpoint_path,
                f'checkpoint_{self.config.current_epoch}')
            # Save full training state using accelerator
            self.accelerator.save_state(save_path)

    def load_state(self):
        """Load the latest checkpoint state"""
        checkpoints = os.listdir(self.checkpoint_path)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))
        save_path = os.path.join(self.checkpoint_path, checkpoints[-1])
        self.accelerator.load_state(save_path)
        self.accelerator.print(f"Loaded checkpoint from {save_path}")

    def save_best_model(self):
        """
        Save best-performing model weights and configuration
        """
        if self.accelerator.is_main_process:  # Only main process saves
            # Save model state dict (using .module for DDP-wrapped models)
            torch.save(
                self.model.state_dict(),
                os.path.join(self.config.output_path,
                             f"{self.config.predictive_backbone}.pt"))
            # Save configuration to output directory
            self.config.save(self.config.output_path)
            self.logger.info(f"Saved best model to {self.config.output_path}")

    def train(self, is_resume=False):
        """
        Main training loop with early stopping
        """
        if is_resume:
            self.load_state()
        # Initialize training state variables
        self.config.best_valid_loss = self.config.get("best_valid_loss")
        self.config.current_epoch = self.config.get("current_epoch")
        self.config.num_steps = self.config.get("num_steps")
        self.config.early_stop_cnt = self.config.get("early_stop_cnt")
        for epoch in range(self.config.current_epoch + 1,
                           self.config.num_epoch):
            train_loss = self.train_epoch(epoch)
            self.save_checkpoint()
            valid_loss = self.evaluate(epoch)
            # Update best validation loss and save checkpoints if improved
            if self.config.best_valid_loss is None or valid_loss < self.config.best_valid_loss:
                self.config.best_valid_loss = valid_loss
                self.save_best_model()
                self.config.early_stop_cnt = 0  # Reset early stop counter
            else:
                self.config.early_stop_cnt += 1  # Increment counter for no improvement
            # Log additional training state metrics
            self.accelerator.log(
                {
                    "valid/valid_loss": valid_loss,
                    "valid/early_stop_cnt": self.config.early_stop_cnt,
                    "epoch": epoch
                },
                step=self.config.num_steps)
            self.accelerator.print(
                f'valid_loss:{valid_loss}, early_stop_cnt:{self.config.early_stop_cnt}'
            )
            # Trigger early stopping if patience exceeded
            if self.config.early_stop_cnt >= self.config.patience:
                self.logger.info("Early stopping triggered!")
                break
        # After training, copy best model to pretrained directory
        predictive_checkpoint_path = os.path.join(
            'pretrained_predictive_model', self.config.dataset,
            f'{self.config.predictive_backbone}.pt')
        os.makedirs(os.path.dirname(predictive_checkpoint_path), exist_ok=True)
        shutil.copyfile(
            os.path.join(self.config.output_path,
                         f"{self.config.predictive_backbone}.pt"),
            predictive_checkpoint_path)
        self.logger.info(f"Copied best model to {predictive_checkpoint_path}")


def main(config):
    """
    Main entry point to start training
    Args:
        config: Configuration object
    """
    if config.resume:
        config = read_config_from_yaml(
            os.path.join(config.checkpoint_path, 'config.yml'))
        config.resume = True
    config.print()  # Print configuration for verification
    trainer = PredictiveTrainer(config)
    trainer.train(is_resume=config.resume)


if __name__ == '__main__':
    # Load default configuration from YAML
    default_config = read_config_from_yaml('config/default.yml')

    args = parse_args()

    # Update default config with command-line arguments
    config = default_config
    config.update({
        'num_epoch': 500,
        'checkpoint_path': args.checkpoint_path,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'predictive_backbone': args.predictive_backbone,
        'patience': args.patience,
        'log_steps': args.log_steps,
        'resume': args.resume
    })
    main(config)
