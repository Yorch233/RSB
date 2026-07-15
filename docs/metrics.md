# Calculate speech-enhancement metrics

The `rsb metric` command evaluates manifested RSB inference results, selects a result associated with a run, or scores
aligned third-party clean/noisy/enhanced directories. It writes per-file values to `metrics.csv` and aggregate mean
and standard deviation to `metrics.json`.

## Command overview

```text
uv run rsb metric --dir RESULT_DIR [OPTIONS]
uv run rsb metric --run RUN [--result VARIANT] [OPTIONS]
uv run rsb metric --clean CLEAN --noisy NOISY --enhanced ENHANCED [OPTIONS]
```

Choose exactly one input mode. Run-linked modes validate provenance through `inference.json`; third-party mode
validates aligned filename sets directly.

## Evaluate a result directory

```bash
uv run rsb metric \
  --dir results/rsb_generative_MMDDhhmm/SDE_N=50 \
  --metrics pesq,estoi,si_sdr
```

`--dir` must contain `inference.json` and the complete WAV set produced by the corresponding inference command.

## Select a result from a run

```bash
uv run rsb metric \
  --run rsb_generative_MMDDhhmm \
  --result SDE_N=50 \
  --metrics pesq,estoi,si_sdr
```

`--run` accepts a run name or directory. If `--result` is omitted and exactly one manifested variant exists, it is
selected automatically. With multiple variants, an interactive terminal presents a menu; non-interactive deployment
must pass `--result` explicitly.

## Evaluate third-party results

```bash
uv run rsb metric \
  --clean /path/to/test/clean \
  --noisy /path/to/test/noisy \
  --enhanced /path/to/third-party/enhanced \
  --metrics pesq,estoi,si_sdr
```

All three directories must contain the same non-empty WAV filename set. Metric artifacts are written into the
enhanced directory.

## Parameters

| Parameter | Accepted values | Default | Purpose |
|---|---|---|---|
| `--dir` | Manifested result directory | None | Selects a specific RSB inference result. |
| `--run` | Run name or directory | None | Selects from manifested results belonging to a run. |
| `--result` | Variant directory name such as `predictive` or `SDE_N=50` | Interactive/sole result | Disambiguates `--run`. |
| `--clean` | Directory | None | Third-party reference-clean WAVs. |
| `--noisy` | Directory | None | Third-party degraded WAVs used to derive noise for energy ratios. |
| `--enhanced` | Directory | None | Third-party enhanced WAVs and output location. |
| `--metrics` / `--metric` | Name, comma-separated names, or repeated options | `pesq,estoi,si_sdr` | Selects output metrics. |
| `--sample-rate` | Positive integer | `16000` | Sets third-party evaluation sampling rate. Run-linked mode uses its manifest/config. |
| `--max-workers` | Non-negative integer | `0` | Sets worker threads; `0` uses the executor default. |
| `--overwrite` | Flag | Off | Recalculates existing metric artifacts. |

## Available metrics

| CLI name | Outputs | Input requirement |
|---|---|---|
| `pesq` | `PESQ` | Clean and enhanced audio. |
| `estoi` | `ESTOI` | Clean and enhanced audio. |
| `si_sdr` | `SI_SDR` | Clean and enhanced audio. |
| `composite` | `CSIG`, `CBAK`, `COVL`, `LLR`, `WSS` | Clean and enhanced audio. |
| `dnsmos` | `DNSMOS_SIG`, `DNSMOS_BAK`, `DNSMOS_OVRL`, `DNSMOS_P808` | Enhanced WAV path and bundled ONNX models. |
| `si_sir` | `SI_SIR` | Clean, noisy, and enhanced audio. |
| `si_sar` | `SI_SAR` | Clean, noisy, and enhanced audio. |

Metric names may be repeated or comma-separated:

```bash
uv run rsb metric \
  --dir results/rsb_generative_MMDDhhmm/SDE_N=50 \
  --metrics pesq,estoi \
  --metrics si-sdr
```

Hyphens are normalized to underscores, so `si-sdr` and `si_sdr` select the same calculator.

## Output artifacts

`metrics.csv` contains one row per WAV and one column per selected output. `metrics.json` records:

- source type and paths or run provenance;
- run name, run ID, model hash, dataset ID, split, and variant when available;
- sample rate, requested metrics, and number of files;
- mean and standard deviation for every output column.

If both files already exist and `--overwrite` is not set, the command returns the existing artifacts. A partial pair
is treated as an error to avoid mixing results from different evaluations.

## Troubleshooting

### Exactly one input mode is required

Do not combine `--dir`, `--run`, or third-party paths. Third-party mode requires `--clean`, `--noisy`, and
`--enhanced` together.

### Multiple run results are available

Pass `--result predictive`, `--result SDE_N=50`, or another exact manifested variant name in automated deployments.

### Run-linked provenance does not match

The inference manifest conflicts with the selected run, model hash, dataset, or split. Re-run inference from the
correct checkpoint rather than editing the manifest manually.

### WAV filename sets differ

Align clean, noisy, and enhanced basenames. Extra files are rejected as well as missing files so summary metrics
cannot silently cover different sample sets.

### PESQ fails for individual files

Confirm supported sampling rates, non-empty finite audio, and sufficient utterance length. Run-linked audio uses the
sample rate recorded by inference; third-party mode uses `--sample-rate`.

### DNSMOS cannot load ONNX models

Verify that the installed package contains `RSB/metrics/dnsmos/models/` and that `onnxruntime` is installed. DNSMOS is
optional; omit it when only intrusive reproduction metrics are required.
