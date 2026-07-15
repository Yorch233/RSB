# Run RSB inference

Inference enhances a registered dataset split with either a predictive checkpoint or a generative SB/RSB checkpoint.
Commands preserve input filenames, write WAV outputs below the configured result root, and record model and dataset
provenance in `inference.json`. The manifest makes interrupted runs resumable and allows the metric command to verify
that a result directory matches its source run.

## Command overview

```text
uv run rsb inference predictive [OPTIONS]
uv run rsb inference generative [OPTIONS]
```

Predictive inference is useful for evaluating the posterior-mean model directly. Generative inference runs the
reverse SDE or probability-flow ODE for vanilla SB and RSB checkpoints.

## Predictive inference

```bash
uv run rsb inference predictive \
  --run rsb_predictive_MMDDhhmm \
  --dataset voicebank
```

The default test result directory is:

```text
results/<predictive-run>/predictive/
  *.wav
  inference.json
```

### Predictive parameters

| Parameter | Accepted values | Default | Purpose |
|---|---|---|---|
| `--run` | Predictive run name or directory | Required | Selects `config.yml` and `model.safetensors`. |
| `--dataset` | Registered dataset ID | Selected dataset | Selects the paired dataset. |
| `--split` | `train`, `valid`, `test` | `test` | Selects the noisy input split. |
| `--device` | `auto`, Torch device, or CUDA index | `auto` | Selects the inference device. |
| `--num-workers` | Non-negative integer | `0` | Configures input workers. |
| `--overwrite` | Flag | Off | Replaces existing or conflicting output WAVs. |
| `--progress` / `--no-progress` | Boolean pair | Progress on | Controls progress rendering. |

`--run` accepts either a run name below the configured run root or a direct directory path. It never selects the
latest run automatically.

## Generative SB/RSB inference

```bash
uv run rsb inference generative \
  --run rsb_generative_MMDDhhmm \
  --dataset voicebank \
  --sampler SDE \
  --num-steps 50
```

The result directory includes the solver and step count:

```text
results/<generative-run>/SDE_N=50/
  *.wav
  inference.json
```

### Generative parameters

| Parameter | Accepted values | Default | Purpose |
|---|---|---|---|
| `--run` | Generative run name or directory | `Yorch233/RSB` from Hugging Face | Selects the local or released checkpoint. |
| `--dataset` | Registered dataset ID | Selected dataset | Selects the paired dataset. |
| `--split` | `train`, `valid`, `test` | `test` | Selects the noisy input split. |
| `--sampler` | `SDE`, `ODE` | `SDE` | Selects stochastic SDE or probability-flow ODE sampling. |
| `--num-steps` | Positive integer | `5` | Sets reverse-sampling discretization steps. |
| `--skip-type` | `time_uniform`, `time_quadratic` | `time_uniform` | Selects timestep spacing. |
| `--seed` | Integer | `10` | Seeds stochastic SDE sampling. |
| `--device` | `auto`, Torch device, or CUDA index | `auto` | Selects the inference device. |
| `--num-workers` | Non-negative integer | `0` | Configures input workers. |
| `--overwrite` | Flag | Off | Replaces existing or conflicting output WAVs. |
| `--progress` / `--no-progress` | Boolean pair | Progress on | Controls progress rendering. |

For paper reproduction, explicitly record `--sampler`, `--num-steps`, `--skip-type`, and `--seed`. SDE results can
change with the seed; ODE inference is deterministic for the same checkpoint, inputs, and numerical settings.

## Use the released Hugging Face checkpoint

Omit `--run` to use [`Yorch233/RSB`](https://huggingface.co/Yorch233/RSB):

```bash
uv run rsb inference generative \
  --dataset voicebank \
  --sampler SDE \
  --num-steps 50
```

RSB downloads `config.yml` and `model.safetensors`, materializes a local run, and migrates the checkpoint
configuration from `0.1.0` to the current `1.0.0` schema in the local materialized copy. The source Hub repository is
not modified.

## Evaluate predictive and generative outputs

```bash
uv run rsb metric \
  --dir results/rsb_predictive_MMDDhhmm/predictive \
  --metrics pesq,estoi,si_sdr

uv run rsb metric \
  --dir results/rsb_generative_MMDDhhmm/SDE_N=50 \
  --metrics pesq,estoi,si_sdr
```

The metric command writes `metrics.csv` and `metrics.json` into the selected result directory. See
[metrics.md](metrics.md) for run selection, optional metrics, and third-party evaluation.

## Manifest and resume behavior

`inference.json` records at least the dataset ID, split, run name, model hash, sample rate, file count, device, and
sampling parameters. On a repeated command:

- matching parameters reuse the manifest and generate only missing WAVs;
- conflicting parameters stop before mixing incompatible outputs;
- `--overwrite` authorizes replacement of existing WAVs and a conflicting manifest.

Do not manually copy WAVs between result variants while retaining the original manifest.

## Troubleshooting

### A run cannot be resolved

Pass the exact run name below the configured run root or an absolute run directory. RSB intentionally does not choose
the newest run. Confirm that the directory contains `config.yml` and `model.safetensors`.

### The dataset ID is not registered

Run `uv run rsb dataset list`, then register or select the correct path. Inference reads `<split>/noisy` and validates
its filenames against `<split>/clean`.

### Existing results conflict with the requested settings

Choose the result directory that matches the intended sampler and step count. Use `--overwrite` only when replacing
the previous result is deliberate; otherwise preserve it and run a different variant.

### CUDA runs out of memory

Use a different `--device`, reduce concurrent deployment load, or run on CPU for a functional check. `--num-workers`
controls input workers, not GPU batch size; inference processes the registered files without a training batch option.

### Hugging Face download fails

Verify network access and Hub availability, or download/materialize the checkpoint ahead of time and pass the local
run directory with `--run`.

### Metrics reject the inference directory

Check that `inference.json` exists and every expected WAV is present with no extra WAVs. Rerun the matching inference
command to fill missing files before calculating metrics.
