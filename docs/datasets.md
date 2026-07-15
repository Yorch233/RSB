# Prepare and manage datasets

RSB trains and evaluates on paired clean/noisy speech. This guide covers dataset synthesis, registry management,
layout validation, and offline posterior-mean generation. Registration is non-destructive: it stores an ID and path
in the local project configuration without copying audio.

## Required layout

```text
<dataset>/
  train/{clean,noisy}/*.wav
  valid/{clean,noisy}/*.wav
  test/{clean,noisy}/*.wav
```

Within each split, clean and noisy directories must contain the same WAV filenames. Regularized training adds a mean
source below every split:

```text
<dataset>/<split>/mean/NCSN++M/*.wav
```

The mean filenames must exactly match the corresponding clean filenames.

## Create a paired dataset

Use this command when clean speech and noise/reverberation sources have not already been exported as aligned pairs:

```bash
uv run rsb dataset create \
  --task enhancement \
  --clean vctk /path/to/clean-audio \
  --noise chime /path/to/noise-audio \
  --output-dir /path/to/output-dataset
```

The exporter writes the paired splits plus `create_configuraton.json`, which records source selections, synthesis
settings, measured statistics, and test-set metrics.

### `dataset create` parameters

| Parameter | Accepted values | Default | Purpose |
|---|---|---|---|
| `--task` | Repeatable `enhancement`, `dereverberation` | Required | Selects one or both corruption tasks. |
| `--clean TYPE PATH` | `vctk`, `wsj0`, `timit` and a source root | Required | Selects the clean corpus and its location. |
| `--noise TYPE PATH` | `none`, `chime`, `qut`, `wham` and a source root | Required | Selects the noise corpus and its location. |
| `--output-dir` / `--output_dir` | Directory | Required | Receives the exported paired dataset. |
| `--sample-rate` | Positive integer | `16000` | Output sampling rate in hertz. |
| `--snr-min` | Number in dB | `-6` | Minimum mixture SNR. |
| `--snr-max` | Number in dB | `14` | Maximum mixture SNR. |
| `--t60-min` | Positive seconds | `0.4` | Minimum reverberation time. |
| `--t60-max` | Positive seconds | `1.0` | Maximum reverberation time. |
| `--seed` | Integer | `100` | Makes synthesis and source selection reproducible. |
| `--overwrite` | Flag | Off | Replaces a non-empty output directory. |

`wsj0` uses its fixed source partitions; its validation and test splits correspond to `si_et_05` and `si_dt_05` as
defined by the project data loader.

Inspect the exported pair counts:

```bash
uv run rsb dataset inspect /path/to/output-dataset
```

## Register a dataset

```bash
uv run rsb dataset add \
  --id voicebank \
  --path /path/to/Voicebank+Demand \
  --select
```

| Parameter | Default | Purpose |
|---|---|---|
| `--id` | Required | Stable identifier accepted by later `--dataset` options. |
| `--path` | Required | Root containing all six paired split directories. |
| `--select` | Off | Makes this ID the project default. |

List registered datasets:

```bash
uv run rsb dataset list
```

Update an ID, path, or selection:

```bash
uv run rsb dataset edit --id voicebank --path /new/dataset/path --select
uv run rsb dataset edit --id voicebank --new-id voicebank-v1
```

Remove only the registry entry:

```bash
uv run rsb dataset delete --id voicebank
```

Add `--yes` or `-y` to skip deletion confirmation. The command never deletes dataset files.

## Generate offline posterior means

Regularized RSB uses a completed predictive run to generate distortion-optimal targets:

```bash
uv run rsb dataset generate-mean \
  --run rsb_predictive_MMDDhhmm \
  --dataset voicebank
```

By default, all three splits are processed. Each output directory receives WAV files and a manifest recording the
predictive run, model hash, dataset ID, split, and file count.

### `dataset generate-mean` parameters

| Parameter | Accepted values | Default | Purpose |
|---|---|---|---|
| `--run` | Predictive run name or directory | Required | Selects the predictive checkpoint. |
| `--dataset` | Registered dataset ID | Selected dataset | Selects the writable dataset. |
| `--split` | Repeatable `train`, `valid`, `test` | All splits | Restricts generation to selected splits. |
| `--device` | `auto`, Torch device, or CUDA index | `auto` | Selects the inference device. |
| `--num-workers` | Non-negative integer | `0` | Configures input workers. |
| `--overwrite` | Flag | Off | Replaces existing posterior-mean WAVs. |
| `--progress` / `--no-progress` | Boolean pair | Progress on | Controls progress rendering. |

If predictive test inference already exists and its manifest matches the same dataset, split, run, and model hash,
RSB reuses those WAVs instead of recomputing them.

## Troubleshooting

### A dataset ID already exists

Use `dataset edit` to change the existing entry, or choose a new ID. IDs must be unique and may contain letters,
numbers, `.`, `_`, and `-`.

### Clean and noisy counts differ

Compare filenames rather than counts alone. Every clean WAV must have a noisy WAV with the same basename. Remove or
regenerate incomplete pairs before registration.

### Posterior-mean generation reports a manifest conflict

The target directory was generated from different inputs or a different checkpoint. Use another dataset copy or pass
`--overwrite` only after confirming the existing means are no longer needed.

### Generative training rejects the mean source

For every train, valid, and test split, check that `mean/<source>` exists and contains exactly the clean WAV filename
set. The source name is taken directly from `--posterior-mean-from`; it is not a hardcoded model-choice list.

### A dataset is read-only

Inference and metrics can use a read-only registered dataset. `dataset generate-mean` cannot; copy or mount a writable
training dataset before generating posterior means.
