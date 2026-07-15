# RSB Deployment Agent Guide

RSB is the official PyTorch implementation of Regularized Schrodinger Bridge for speech enhancement. This file tells
an execution agent how to install, configure, train, infer, and evaluate the project on behalf of a human operator.
Treat [`README.md`](README.md) as the human overview and the files below as the command references:

- [`docs/installation.md`](docs/installation.md): host requirements, environment creation, configuration, and checks
- [`docs/datasets.md`](docs/datasets.md): paired dataset creation, registry management, and posterior means
- [`docs/training.md`](docs/training.md): predictive training, vanilla SB, RSB, resume, and run artifacts
- [`docs/inference.md`](docs/inference.md): predictive/generative inference, Hugging Face loading, and manifests
- [`docs/metrics.md`](docs/metrics.md): run-linked and third-party evaluation

Preserve the `For Human`, `For Agent`, and `Skip This README` hand-off prompts in `README.md`.

## Deployment objective

The standard deployment produces an auditable sequence of versioned artifacts:

```text
paired dataset
  → predictive run
  → offline posterior means
  → vanilla SB or regularized RSB run
  → manifested inference WAVs
  → per-file and aggregate metrics
```

Do not skip artifact-producing stages by copying untracked files into run or result directories. Use the CLI so every
checkpoint, posterior-mean set, inference result, and metric output retains its provenance.

## Host and environment rules

- Run commands from the repository root.
- Use Python 3.12 and UV; create the environment with `uv sync`.
- Torch and Torchaudio must be version 2.5.1 or later.
- Training requires a CUDA-capable deployment host. Metric-only work can run without CUDA.
- NCSN++ first attempts its optimized CUDA JIT operators. If they are unavailable, explain the throughput trade-off
  and ask whether to use the portable PyTorch autograd backend; Yes is the default.
- Training prints the effective parameters and asks for confirmation. Use `--yes` only when the human has reviewed
  the complete non-interactive job configuration.
- Never write credentials, remote host details, machine paths, dataset paths, checkpoints, generated audio, or tokens
  into tracked files.

## Dataset contract

A registered dataset root must contain:

```text
train/{clean,noisy}/*.wav
valid/{clean,noisy}/*.wav
test/{clean,noisy}/*.wav
```

Clean and noisy filenames must match within each split. RSB additionally requires
`<split>/mean/<source>/*.wav` to match the corresponding clean filenames and count for train, valid, and test.
Registration stores only the dataset ID and path and must never copy, modify, or delete the source dataset.

## Configure the deployment

```bash
uv sync
uv run rsb dataset add --id voicebank --path /path/to/dataset --select
uv run rsb config
```

`rsb config` writes ignored host settings to `.config/rsb.yml`: dataset registry and selection, precision, GPU mode
and IDs, logger, logging/checkpoint intervals, and run/result roots. Use the project default path. Before continuing,
confirm that `uv run rsb dataset list` shows the intended selected dataset.

Tracked configuration inheritance is:

```text
config/data_representation.yml
  └─ config/dataset.yml
       └─ config/default.yml
            └─ .config/rsb.yml
```

`config/default.yml` is the generative training default and must not be treated as a predictive-run configuration.
Predictive runs derive and persist their method and concrete backbone when the run is created.

## Execute the training sequence

Train the predictive model:

```bash
uv run rsb train predictive --dataset voicebank
```

Record the generated `rsb_predictive_MMDDhhmm` name, then generate all posterior-mean splits:

```bash
uv run rsb dataset generate-mean \
  --run rsb_predictive_MMDDhhmm \
  --dataset voicebank
```

For vanilla Schrödinger Bridge:

```bash
uv run rsb train generative \
  --dataset voicebank \
  --training-method none
```

For Regularized Schrödinger Bridge:

```bash
uv run rsb train generative \
  --dataset voicebank \
  --training-method regularization \
  --posterior-mean-from NCSN++M
```

Only `none` and `regularization` are valid training methods. `none` does not use posterior means. `regularization`
defaults to `NCSN++M` and must validate every mean split before a run is created. Do not invent a latest-run selection;
capture and reuse the exact run name printed by each command.

## Execute inference and metrics

```bash
uv run rsb inference generative \
  --run rsb_generative_MMDDhhmm \
  --dataset voicebank \
  --sampler SDE \
  --num-steps 50

uv run rsb metric \
  --dir results/rsb_generative_MMDDhhmm/SDE_N=50 \
  --metrics pesq,estoi,si_sdr
```

For reproducibility, report the run, dataset ID, split, sampler, step count, skip type, seed, model hash, and output
directory. When `--run` is omitted, generative inference materializes the default
[`Yorch233/RSB`](https://huggingface.co/Yorch233/RSB) checkpoint. Do not imply that another local run was used.

Use `rsb metric --run RUN --result VARIANT` when selecting among multiple manifested results. Use aligned
`--clean/--noisy/--enhanced` directories only for third-party evaluation.

## Artifact acceptance checks

A completed training run must contain:

```text
runs/<run-name>/
  config.yml
  model.safetensors
  checkpoints/last.ckpt
```

Intermediate `checkpoints/step=*.ckpt` files may also be present. `model.safetensors` is the validation-selected
inference model; `last.ckpt` is the complete resumable Lightning state.

A completed inference result must contain `inference.json` and exactly the expected WAV filename set. A completed
evaluation must contain both `metrics.csv` and `metrics.json` in the selected result directory.

New runs use `version: 1.0.0`. Legacy `0.1.0` and unversioned configurations are migrated in memory during loading;
do not rewrite the source checkpoint merely to make inference work.

## Deployment reporting

When handing the result back to the human, state:

1. the registered dataset ID and split used;
2. the exact predictive and/or generative run names;
3. whether CUDA JIT or portable PyTorch operators were selected;
4. the sampler, step count, skip type, and seed for generative inference;
5. the run, result, checkpoint, manifest, and metric artifact paths;
6. any skipped stage, fallback, warning, or incomplete artifact.

Do not report a deployment as complete when a required manifest, checkpoint, posterior-mean split, WAV, or metrics
file is missing. Consult the troubleshooting section of the corresponding deployment guide before changing command
parameters or overwriting artifacts.
