# Train predictive, SB, and RSB models

RSB deployment uses two training stages. The predictive model estimates offline posterior means. Generative training
then runs either vanilla Schrödinger Bridge (`none`) or Regularized Schrödinger Bridge (`regularization`). Both commands
save complete Lightning state for resume and export the best validation model as `model.safetensors`.

Run training only after completing [installation and host configuration](installation.md) and registering a paired
dataset.

## Train the predictive model

```bash
uv run rsb train predictive --dataset voicebank
```

The default method is `NCSN++M`. A normal run is written to `runs/rsb_predictive_MMDDhhmm/`. Its `config.yml` records
both `predictive_method: NCSN++M` and the concrete backbone needed for resume and inference.

### Predictive parameters

| Parameter | Choices/default | Purpose |
|---|---|---|
| `--method` | `NCSN++M`; default `NCSN++M` | Selects the predictive method and derived backbone. |
| `--dataset` | Registered ID; selected dataset by default | Selects train and validation data. |
| `--run-name` | Text; generated timestamp name by default | Overrides the run directory and WandB run name. |
| `--run-dir` | Directory; configured `runs` root | Overrides the run root. |
| `--max-epoch` | Positive integer; default `1000` | Sets the maximum epochs. |
| `--learning-rate` | Non-negative number; default `1e-4` | Sets the optimizer learning rate. |
| `--batch-size` | Positive integer; default `16` | Sets per-process batch size. |
| `--optimizer` | `Adam`, `AdamW`; default `Adam` | Selects the Torch optimizer. |
| `--ema` / `--no-ema` | EMA on by default | Enables or disables exponential moving averages. |
| `--ema-rate` | `0` to `<1`; default `0.999` | Sets EMA decay. |
| `--patience` | Positive integer; default `50` | Stops on validation-loss stagnation. |
| `--num-workers` | Non-negative integer; default `0` | Sets DataLoader workers per process. |
| `--seed` | Integer; default `10` | Seeds data and training operations. |
| `--logger` | `wandb`, `none`; configured logger by default | Controls experiment logging. |
| `--log-steps` | Positive integer; configured default `10` | Sets training-loss logging frequency. |
| `--precision` | Lightning precision string; wizard mapping by default | Overrides `none`/`fp16`/`bf16` mapping, for example with `32-true` or `bf16-mixed`. |
| `--accelerator` | Lightning accelerator; default `gpu` | Overrides accelerator selection. |
| `--devices` | Lightning device expression; wizard GPU IDs by default | Overrides device selection. |
| `--strategy` | Lightning strategy; `auto` or `ddp` from the wizard | Overrides distributed strategy. |
| `--num-nodes` | Positive integer; default `1` | Sets distributed host count. |
| `--resume` | Flag; off | Enables complete Lightning-state resume. |
| `--checkpoint-path` | Run, checkpoints directory, or `last.ckpt` | Selects resume state and requires `--resume`. |
| `--yes` | Flag; off | Skips operator-backend and launch confirmations. |

Leave advanced Lightning overrides unset for normal deployments so `.config/rsb.yml` controls precision and devices.

## Generate posterior means for RSB

```bash
uv run rsb dataset generate-mean \
  --run rsb_predictive_MMDDhhmm \
  --dataset voicebank
```

Regularized training requires matching train, valid, and test means. See [datasets.md](datasets.md) for generation,
manifest, overwrite, and read-only dataset behavior. Vanilla SB does not use posterior means.

## Train vanilla SB

```bash
uv run rsb train generative \
  --dataset voicebank \
  --schedule VE \
  --training-method none
```

`none` disables regularized perturbation and rejects `--posterior-mean-from`. This is the explicit vanilla
Schrödinger Bridge reproduction mode.

## Train RSB

```bash
uv run rsb train generative \
  --config config/default.yml \
  --dataset voicebank \
  --schedule VE \
  --training-method regularization \
  --posterior-mean-from NCSN++M \
  --regularization-weight quadratic
```

Before creating the run, RSB checks that the selected mean source exists for train, valid, and test and exactly
matches each clean split by filename and count. Validation PESQ selects `model.safetensors`; validation SI-SDR drives
early stopping. Fifty fixed validation samples are used for sampled model selection and five test samples are logged
without affecting checkpoint or early-stopping decisions.

### Generative parameters

| Parameter | Choices/default | Purpose |
|---|---|---|
| `--config` | YAML path; `config/default.yml` | Selects generative model and objective defaults. |
| `--dataset` | Registered ID; selected dataset by default | Selects train and validation data. |
| `--schedule` | `VE`, `VP`; default `VE` | Selects the bridge schedule. |
| `--training-method` | `none`, `regularization`; default `regularization` | Selects vanilla SB or RSB. |
| `--posterior-mean-from` | Mean directory name; default `NCSN++M` for RSB | Selects `<split>/mean/<source>`. |
| `--regularization-weight` | `quadratic`, `cosine`, `linear`; default `quadratic` | Selects time-dependent regularization weighting. |
| `--training-target` | `data`, `noise`, `score`, `vector`; default `data` | Selects the SDE network target. |
| `--generative-backbone` | Registered backbone; default `ncsnpp_base` | Selects the generative architecture. |
| `--loss-weight-type` | Configured string; default `constant` | Selects the SDE loss weighting rule. |
| `--reduction` | `mean`, `sum`; default `sum` | Selects per-sample loss reduction. |
| `--time-loss-weight` | Non-negative number; default `1e-3` | Weights the waveform-domain loss. |
| `--t-min` / `--t-max` | Numbers with `t_min < t_max`; defaults `1e-4` / `1` | Sets the training-time interval. |
| `--patience` | Positive integer; default `20` | Sets validation SI-SDR early-stopping patience. |

Generative training also accepts the shared run, optimization, EMA, worker, seed, logger, Lightning, and `--yes`
options described for predictive training. CLI values override the selected YAML; host-specific missing values are
filled from `.config/rsb.yml`.

## Use a custom generative configuration

Create a small YAML that inherits the released defaults:

```yaml
inherit: "/path/to/RSB/config/default.yml"
version: "1.0.0"
batch_size: 8
num_epoch: 500
```

Launch it with:

```bash
uv run rsb train generative \
  --config /path/to/custom.yml \
  --dataset voicebank
```

Do not add predictive method or predictive backbone fields to a generative configuration. Those fields belong only
to predictive run artifacts.

## Resume training

Both training commands accept the same resume contract:

```bash
uv run rsb train generative \
  --resume \
  --checkpoint-path runs/rsb_generative_MMDDhhmm
```

`--checkpoint-path` may point to the run directory, its `checkpoints/` directory, or
`checkpoints/last.ckpt`. `--resume` and `--checkpoint-path` must be supplied together. Resume loads the persisted run
configuration and complete Lightning model/optimizer/scheduler state before applying explicit CLI overrides.

## Run artifacts

```text
runs/<run-name>/
  config.yml
  model.safetensors
  checkpoints/
    last.ckpt
    step=*.ckpt
```

`config.yml` contains the effective versioned parameters, run ID, selected operator backend, and JIT status.
`model.safetensors` is the validation-selected inference model. `last.ckpt` is the complete resumable state.

## Launch behavior

Before training, RSB attempts the optimized NCSN++ CUDA JIT operators. When unavailable, it asks whether to continue
with portable PyTorch autograd operators; Yes is the default. It then prints the effective parameters and asks for a
final training confirmation. Use `--yes` only in reviewed non-interactive deployment jobs.

## Troubleshooting

### RSB reports missing posterior means

Generate all three splits with `dataset generate-mean`. Verify the source spelling and the WAV filename set under
every `<split>/mean/<source>` directory. Use `--training-method none` only when intentionally deploying vanilla SB.

### Training starts on the wrong GPUs

Rerun `uv run rsb config` and verify multi-GPU mode and GPU IDs. Avoid mixing wizard device settings with advanced
Lightning overrides unless the deployment launcher requires them.

### CUDA JIT is unavailable

Fix the CUDA build toolchain for best throughput, or accept the PyTorch backend. Compare runs only when their saved
backend/JIT status is understood because operator choice can affect performance characteristics.

### WandB does not resume the expected run

Resume from the original run directory or `last.ckpt`. Do not create a new `--run-name` for an existing run. Confirm
that the saved `run_id` remains present in `config.yml`.

### A checkpoint cannot be resumed

Only complete Lightning `last.ckpt` state is resumable. `model.safetensors` is intended for inference and Hub loading,
not optimizer-state recovery.
