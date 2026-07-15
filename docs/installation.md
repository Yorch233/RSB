# Install and configure RSB

This guide prepares a host to train or run RSB. Run every command from the repository root. The setup is project-local:
UV manages `.venv`, RSB writes host settings to `.config/rsb.yml`, and generated runs and results stay outside the
tracked source tree.

## Requirements

- Linux deployment host
- Python 3.12
- [UV](https://docs.astral.sh/uv/)
- NVIDIA GPU for training and practical RSB inference
- Torch and Torchaudio 2.5.1 or later, installed through the project dependencies
- A paired dataset containing `{train,valid,test}/{clean,noisy}`
- A Weights & Biases login when WandB logging is enabled

Metric-only deployment can run without CUDA. NCSN++ CUDA JIT requires a compatible NVIDIA driver, CUDA toolkit,
compiler, Ninja, and `CUDA_HOME`; the portable PyTorch backend remains available when JIT compilation is unavailable.

## Create the virtual environment

Clone the repository if it is not already present:

```bash
git clone https://github.com/Yorch233/RSB.git
cd RSB
```

Create the project environment and verify Python:

```bash
uv sync
uv run python --version
```

`uv sync` creates or updates `.venv` from `pyproject.toml`. The version command must report Python 3.12. The project
does not require `uv.lock` or `.python-version` for deployment.

## Register the deployment dataset

An existing paired dataset must have the following structure:

```text
<dataset>/
  train/clean/*.wav
  train/noisy/*.wav
  valid/clean/*.wav
  valid/noisy/*.wav
  test/clean/*.wav
  test/noisy/*.wav
```

Register the dataset under a stable ID:

```bash
uv run rsb dataset add \
  --id voicebank \
  --path /path/to/Voicebank+Demand \
  --select
```

Registration stores the path; it does not copy, rewrite, or delete audio. Use the [dataset guide](datasets.md) when
the paired dataset must first be synthesized or when posterior means must be generated.

## Configure the host

Launch the configuration wizard:

```bash
uv run rsb config
```

The wizard detects the available GPUs, validates registered datasets, prints the final configuration, and writes the
accepted settings to `.config/rsb.yml`.

| Setting | Choices and default | Deployment effect |
|---|---|---|
| Dataset registry | One or more ID/path pairs | Makes paired datasets available to all commands. |
| Selected dataset | Any registered ID | Becomes the default when `--dataset` is omitted. |
| Mixed precision | `none`, `fp16`, `bf16`; default `none` | Controls the Lightning training precision. |
| Multi-GPU | Yes/No; default Yes when multiple GPUs are detected | Enables distributed training. |
| GPU IDs | `all` or comma-separated IDs; default `all` | Restricts the visible training devices. |
| Logger | `wandb`, `none`; default `wandb` | Enables or disables remote experiment logging. |
| Logging interval | Positive steps; default `10` | Controls training-loss logging frequency. |
| State-save interval | Positive steps; default `1000` | Controls resumable checkpoint frequency. |
| Checkpoint limit | Positive count; default `3` | Limits retained intermediate checkpoints. |
| Run directory | Path; default `<project>/runs` | Stores training runs and checkpoints. |
| Result directory | Path; default `<project>/results` | Stores inference WAVs and metrics. |

Standard deployments must use the default output `.config/rsb.yml`, which is the project configuration loaded by
training and workflow commands. `--output PATH` is available for an explicitly managed export, but other commands do
not automatically load that alternate path. `--force` overwrites and saves without the final confirmation.

Tracked defaults inherit in this order:

```text
config/data_representation.yml
  └─ config/dataset.yml
       └─ config/default.yml
            └─ .config/rsb.yml
```

The project configuration contains host and registry settings. `config/default.yml` contains generative training
defaults. New configurations and runs use `version: 1.0.0`; legacy `0.1.0` and unversioned checkpoint configurations
are migrated in memory during loading.

## Verify the installation

```bash
uv run rsb --help
uv run rsb dataset list
```

The first command must show the RSB command groups. The second must show the selected dataset and its resolved path.
Do not start training until all six paired directories exist and clean/noisy filenames match within every split.

## Continue the deployment

Follow these guides in order:

1. [Dataset preparation and posterior means](datasets.md)
2. [Predictive, SB, and RSB training](training.md)
3. [Predictive and generative inference](inference.md)
4. [Metrics and third-party evaluation](metrics.md)

## Troubleshooting

### UV selects the wrong Python version

Confirm that Python 3.12 is installed and available to UV, then recreate the project environment with that interpreter.
Do not bypass the `requires-python` constraint with Python 3.11 or 3.13.

### The configuration wizard reports no GPU

Run the wizard on the GPU deployment host, not on a CPU-only login node. Metric-only commands do not require the
wizard to detect CUDA, but training does.

### CUDA JIT compilation fails

Verify the compiler, Ninja, CUDA toolkit, driver, and `CUDA_HOME`. RSB asks whether to continue with portable PyTorch
operators; accepting the default Yes is safe but can change throughput. The selected backend and JIT status are saved
in the run config and WandB parameters.

### The dataset is rejected

Check the exact `{train,valid,test}/{clean,noisy}` directory names and ensure each clean/noisy split contains the same
WAV filename set. Dataset roots are directories, not individual split paths.

### WandB blocks startup

Authenticate with WandB before training or select `none` as the logger in `rsb config`. Do not place API keys in the
repository or run configuration.

Never commit `.venv`, `.config/rsb.yml`, credentials, machine paths, datasets, checkpoints, generated WAVs, or result
artifacts.
