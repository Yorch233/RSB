# RSB - Regularized Schrödinger Bridge

Research codebase for speech enhancement using Schrödinger Bridge diffusion models.
Python 3.12 + PyTorch 2.5.1. No package manager, no tests, no linter.

## Setup

```bash
pip install -r requirements.txt  # requirements.txt not in repo; use wandb/*/files/requirements.txt as reference
accelerate config  # configure distributed training before first run
```

Key dependencies: `accelerate`, `torch`, `torchaudio`, `wandb`, `pesq`, `pystoi`, `safetensors`, `huggingface_hub`.

## Commands

All training uses `accelerate launch`. Inference and metrics use `python -m`.

```bash
# Train RSB model
accelerate launch -m cli.train --dataset voicebank+demand --training_method regularization --training_target data

# Train predictive model (prerequisite for RSB)
accelerate launch -m cli.train_predictive --dataset voicebank+demand --predictive_backbone ncsnpp_base

# Inference
python -m cli.inference --audio_path /path/to/noisy --output_dir /path/to/output --model_dir /path/to/run/dir --num_step 50

# Evaluate metrics
python -m cli.calc_metric --clean_dir /path/to/clean --noisy_dir /path/to/noisy --enhanced_dir /path/to/enhanced
```

## Configuration

YAML files in `config/` use inheritance: `data_representation.yml` -> `dataset.yml` -> `default.yml`.
Override via CLI args or create new YAML inheriting from `default.yml`.

Must edit before first run:
- `config/dataset.yml`: dataset paths (each needs `train/{clean,noisy}`, `valid/{clean,noisy}`, `test/{clean,noisy}`)
- `config/default.yml`: `run_dir`, `log_with`, training hyperparameters

## Architecture

```
cli/               # Entry points (train, inference, calc_metric)
RSB/
  modeling_rsb.py  # RSB model class (main model)
  sdes.py          # Schrödinger Bridge SDEs (SB_VESDE, SB_VPSDE)
  solver.py        # SDE/ODE numerical solvers
  trainer.py       # Training loop (RSB_Trainer)
  common/
    config.py      # Config loader with YAML inheritance
    register.py    # Registry pattern (Register class)
    notifier.py    # WeChat notification via AutoDL
  backbone/
    registry.py    # BackboneRegister
    ncsnpp/        # NCSN++ backbone (default: ncsnpp_base)
  dataset/
    ComplexSpecDataset.py  # STFT-based dataset, STFTUtil singleton
    AudioFolder.py         # Audio file loading
  evaluate/
    registry.py    # MetricRegister
    metrics.py     # PESQ, ESTOI, SI-SNR, DNSMOS, etc.
config/            # YAML configs with inheritance
pretrained_predictive_model/  # Checkpoints organized by dataset name
```

## Key Patterns

- **Registry**: `BackboneRegister.register("name")` and `MetricRegister.register("name")` decorators. Fetch with `BackboneRegister.fetch("name")`.
- **STFTUtil**: Class-level singleton. Must call `STFTUtil.initial()` before use (auto-called on first dataset load). Parameters in `config/data_representation.yml`.
- **Config**: `read_config_from_yaml(path)` returns `Config` object. Supports `inherit` key for YAML chaining.
- **SDE types**: `VE` (Variance Exploding, default) and `VP` (Variance Preserving).
- **Training methods**: `none`, `optimal`, `condition`, `optimal&condition`, `regularization` (default, proposed method).
- **Training targets**: `data` (default), `noise`, `score`, `vector`.
- **Posterior mean**: When `load_posterior_mean=True`, expects `mean/<source>/` folders in dataset dirs. Sources: `NCSN++M`, `MetricGAN+`, `SEMamba`, `MP-SENet`.

## Gotchas

- No `requirements.txt` at repo root. Check `wandb/run-*/files/requirements.txt` for pinned versions.
- `run_dir` defaults to `/root/autodl-tmp/runs` (AutoDL cloud path). Change for local runs.
- Config has `autodl_token` and `wechat_notify` fields for AutoDL cloud integration - ignore locally.
- NCSN++ backbone pads spectrogram length to multiple of 64 (`pad_spec` in `modeling_ncsnpp.py:35`).
- `cli/inference.py` imports `from RSB.modeling_RSB import RSB` (capital RSB) but file is `modeling_rsb.py` - case-sensitive filesystems will break.
- wandb run IDs follow pattern `abc` + `MMDDHHmmSS`.
- Checkpoints saved every `save_state_steps` (default 1000) steps, with `checkpoints_total_limit` (default 3) kept.
