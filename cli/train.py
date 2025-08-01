import argparse
import os

from RSB.common.config import read_config_from_yaml
from RSB.trainer import RSB_Trainer


def parse_arguments(config):
    """
    Parse command line arguments, providing descriptions for each parameter.

    Args:
        config: The default configuration object loaded from YAML.

    Returns:
        Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Training configuration for the RSB model.')

    # --- General Configuration ---
    parser.add_argument('--seed',
                        type=int,
                        default=config.seed,
                        help='Random seed for reproducibility.')
    parser.add_argument(
        '--run_name',
        type=str,
        default=config.get("run_name"),
        help=
        'Name for this specific training run (optional, overrides default naming).'
    )
    parser.add_argument(
        '--bridge_type',
        type=str,
        default=config.get("bridge_type"),
        choices=['VP', 'VE'],
        help=
        'Type of bridge function to use: Variance Preserving (VP) or Variance Exploding (VE).'
    )

    # --- Training Data and Process ---
    parser.add_argument(
        '--dataset',
        type=str,
        default=config.get("dataset", "vocebank+demand"),
        help=
        'Name of the dataset to use (e.g., vocebank+demand). Must be defined in config/dataset.yml.'
    )
    parser.add_argument('--learning_rate',
                        type=float,
                        default=config.get("learning_rate", 1e-4),
                        help='Initial learning rate for the optimizer.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=config.get("batch_size", 16),
                        help='Number of samples per batch during training.')
    parser.add_argument('--num_epoch',
                        type=int,
                        default=config.get("num_epoch", 1000),
                        help='Maximum number of training epochs.')
    parser.add_argument(
        '--run_dir',
        type=str,
        default=config.get("run_dir", "runs"),
        help=
        'Base directory where run outputs (logs, checkpoints) will be saved.')
    parser.add_argument(
        '--log_steps',
        type=int,
        default=config.get("log_steps", 10),
        help='Frequency (in training steps) to log metrics and information.')
    parser.add_argument(
        '--log_with',
        type=str,
        default=config.get("log_with", "wandb"),
        choices=['none', 'wandb'],
        help=
        'Tool to use for experiment tracking and logging (e.g., Weights & Biases).'
    )

    # --- Execution Control ---
    parser.add_argument(
        '--resume',
        action='store_true',
        help=
        'If set, resume training from the latest checkpoint found in --checkpoint_path.'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='(Placeholder/Unused) Flag for potential dummy runs or testing.')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help=
        'Path to the checkpoint directory to resume training from (required if --resume is used).'
    )

    # --- RSB-Specific Training Options ---
    parser.add_argument('--training_method',
                        type=str,
                        default=config.training_method,
                        choices=[
                            'none', 'optimal', 'condition',
                            'optimal&condition', 'regularization'
                        ],
                        help='Specifies the RSB training strategy: '
                        'none (standard), '
                        'optimal (optimal path), '
                        'condition (conditioning), '
                        'optimal&condition (combined), '
                        'regularization (applies specified regularization).')
    parser.add_argument(
        '--training_target',
        type=str,
        default=config.training_target,
        choices=['data', 'noise', 'score', 'vector'],
        help='Defines the primary target for the training loss: '
        'data (clean data), '
        'noise (noise component), '
        'score (score function), '
        'vector (specific vector field).')
    parser.add_argument(
        '--regularization_weight',  # Note: Typo preserved from original ('regulization' instead of 'regularization')
        type=str,
        default=config.regularization_weight,
        choices=['quadratic', 'linear'],
        help=
        'Type of regularization to apply if --training_method=regularization is selected: '
        'quadratic or linear weight.')

    return parser.parse_args()


def start_training(config):
    """
    Initialize the trainer and start the training process.

    Args:
        config: The final configuration object (defaults updated with CLI args).
    """
    # If resuming, load the configuration from the specified checkpoint path
    if config.resume:
        config = read_config_from_yaml(
            os.path.join(config.checkpoint_path, 'config.yml'))
        config.resume = True  # Ensure resume flag is set in loaded config

    # Instantiate the trainer with the configuration
    model = RSB_Trainer(config)
    # Start the training loop
    model.train(is_resume=config.resume)


def main():
    """Main execution flow of the training script."""
    # 1. Read the original default configuration from YAML file
    original_config = read_config_from_yaml("config/default.yml")

    # 2. Parse command line arguments, using defaults from the configuration
    args = parse_arguments(original_config)

    # 3. Build a dictionary of command line arguments that were actually provided
    #    (excluding those that are None, e.g., not passed)
    update_dict = {k: v for k, v in vars(args).items() if v is not None}

    # 4. Update the original configuration with the provided command line arguments
    config = original_config
    config.update(
        update_dict)  # Assuming the config object has an update method

    # 5. Start the training process with the final configuration
    start_training(config)


if __name__ == '__main__':
    main()
