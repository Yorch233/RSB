import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchaudio
from torch import Tensor, nn

from RSB.training import generative as generative_training
from RSB.utils.config import CURRENT_CONFIG_VERSION, BaseConfiguer, Config
from RSB.workflows import artifacts, evaluation, inference, posterior
from RSB.workflows.artifacts import RunArtifact


def make_audio_dataset(root: Path, sample_rate: int = 16_000) -> Path:
    waveform = torch.linspace(-0.25, 0.25, sample_rate // 4).reshape(1, -1)
    for split in artifacts.SPLITS:
        for signal in ("clean", "noisy"):
            directory = root / split / signal
            directory.mkdir(parents=True)
            torchaudio.save(directory / "sample.wav", waveform, sample_rate)
    return root


def make_run(root: Path, run_type: str, dataset_root: Path) -> RunArtifact:
    root.mkdir(parents=True)
    model_path = root / "model.safetensors"
    model_path.write_bytes(b"model")
    values = {
        "run_type": run_type,
        "run_name": root.name,
        "run_id": "run-id",
        "dataset": "example",
        "datasets": {"example": str(dataset_root)},
        "sample_rate": 16_000,
        "n_fft": 510,
        "num_frames": 256,
        "hop_length": 128,
        "spec_abs_exponent": 0.5,
        "spec_factor": 0.33,
        "window": "sqrthann",
    }
    if run_type == "predictive":
        values.update({"predictive_method": "NCSN++M", "predictive_backbone": "fake"})
    else:
        values.update({"training_method": "regularization", "generative_backbone": "fake"})
    BaseConfiguer.dump(values, root / "config.yml")
    return RunArtifact(root, Config(values), run_type, model_path, artifacts.file_sha256(model_path))


class IdentityPredictive(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value


class IdentityBridge:
    def to(self, device: torch.device) -> "IdentityBridge":
        del device
        return self

    def eval(self) -> "IdentityBridge":
        return self

    def enhance(self, audio: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        del kwargs
        empty = torch.empty(0)
        return audio, empty, empty


def patch_workflow_resolution(monkeypatch, run: RunArtifact, dataset_root: Path, results_root: Path) -> None:
    monkeypatch.setattr(inference, "resolve_run", lambda reference, expected_type=None: run)
    monkeypatch.setattr(inference, "resolve_dataset", lambda config, dataset_id=None: ("example", dataset_root))
    monkeypatch.setattr(inference, "result_directory", lambda selected_run, variant: results_root / variant)
    monkeypatch.setattr(posterior, "resolve_run", lambda reference, expected_type=None: run)
    monkeypatch.setattr(posterior, "resolve_dataset", lambda config, dataset_id=None: ("example", dataset_root))
    monkeypatch.setattr(posterior, "result_directory", lambda selected_run, variant: results_root / variant)
    monkeypatch.setattr(evaluation, "resolve_run", lambda reference: run)
    monkeypatch.setattr(evaluation, "resolve_dataset", lambda config, dataset_id=None: ("example", dataset_root))
    monkeypatch.setattr(evaluation, "result_directory", lambda selected_run, variant: results_root / variant)


def test_resolve_run_accepts_name_and_infers_legacy_type(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    run_root = tmp_path / "runs"
    run = make_run(run_root / "legacy", "predictive", dataset_root)
    config = dict(run.config.dict())
    config.pop("run_type")
    BaseConfiguer.dump(config, run.path / "config.yml")
    local_config = tmp_path / "rsb.yml"
    BaseConfiguer.dump({"run_dir": str(run_root)}, local_config)
    monkeypatch.setattr(artifacts, "USER_CONFIG_PATH", local_config)

    resolved = artifacts.resolve_run("legacy", "predictive")

    assert resolved.path == run.path
    assert resolved.run_type == "predictive"
    assert resolved.config.version == CURRENT_CONFIG_VERSION
    with pytest.raises(ValueError, match="Expected a generative run"):
        artifacts.resolve_run(run.path, "generative")


def test_predictive_inference_is_manifested_and_resumable(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    run = make_run(tmp_path / "runs" / "predictive", "predictive", dataset_root)
    results_root = tmp_path / "results" / run.name
    patch_workflow_resolution(monkeypatch, run, dataset_root, results_root)
    calls = []

    def load_model(selected_run, device):
        calls.append((selected_run, device))
        return IdentityPredictive()

    monkeypatch.setattr(inference, "load_predictive_model", load_model)
    output = inference.run_predictive_inference(run.path, device="cpu", progress=False)
    resumed = inference.run_predictive_inference(run.path, device="cpu", progress=False)

    assert resumed == output
    assert (output / "sample.wav").is_file()
    manifest = json.loads((output / "inference.json").read_text())
    assert manifest["artifact_type"] == "predictive_inference"
    assert manifest["files"] == ["sample.wav"]
    assert len(calls) == 1


def test_generative_inference_uses_sampler_result_directory(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    run = make_run(tmp_path / "runs" / "generative", "generative", dataset_root)
    results_root = tmp_path / "results" / run.name
    patch_workflow_resolution(monkeypatch, run, dataset_root, results_root)
    monkeypatch.setattr(inference.RSB, "from_pretrained", lambda *args, **kwargs: IdentityBridge())

    output = inference.run_generative_inference(
        run.path,
        sampler="ODE",
        num_steps=7,
        device="cpu",
        progress=False,
    )

    assert output.name == "ODE_N=7"
    manifest = json.loads((output / "inference.json").read_text())
    assert manifest["sampler"] == "ODE"
    assert manifest["num_steps"] == 7
    assert manifest["seed"] == 10


def test_generate_mean_writes_all_splits_and_reuses_test_results(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    run = make_run(tmp_path / "runs" / "predictive", "predictive", dataset_root)
    results_root = tmp_path / "results" / run.name
    patch_workflow_resolution(monkeypatch, run, dataset_root, results_root)
    monkeypatch.setattr(posterior, "load_predictive_model", lambda selected_run, device: IdentityPredictive())
    predictive_results = results_root / "predictive"
    predictive_results.mkdir(parents=True)
    source_test = dataset_root / "test" / "noisy" / "sample.wav"
    (predictive_results / "sample.wav").write_bytes(source_test.read_bytes())
    artifacts.write_json(
        predictive_results / "inference.json",
        {
            "artifact_type": "predictive_inference",
            "run_name": run.name,
            "model_sha256": run.model_sha256,
            "dataset_id": "example",
            "split": "test",
        },
    )

    manifests = posterior.generate_posterior_means(run.path, device="cpu", progress=False)

    assert len(manifests) == 3
    for split in artifacts.SPLITS:
        output = dataset_root / split / "mean" / "NCSN++M"
        assert (output / "sample.wav").is_file()
        assert json.loads((output / "manifest.json").read_text())["split"] == split
    assert (dataset_root / "test" / "mean" / "NCSN++M" / "sample.wav").read_bytes() == source_test.read_bytes()


def _write_matching_posterior_means(dataset_root: Path, source: str) -> None:
    for split in artifacts.SPLITS:
        output = dataset_root / split / "mean" / source
        output.mkdir(parents=True)
        source_file = dataset_root / split / "clean" / "sample.wav"
        (output / source_file.name).write_bytes(source_file.read_bytes())


def _posterior_mean_config(dataset_root: Path, source: str) -> Config:
    return Config(
        {
            "training_method": "regularization",
            "posterior_mean_from": source,
            "dataset": "example",
            "datasets": {"example": str(dataset_root)},
        }
    )


def test_validate_posterior_means_accepts_dataset_defined_source_without_manifest(tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    _write_matching_posterior_means(dataset_root, "custom-predictor")

    provenance = posterior.validate_posterior_means(_posterior_mean_config(dataset_root, "custom-predictor"))

    assert provenance == {
        "source": "custom-predictor",
        "splits": {
            split: {
                "directory": str(dataset_root / split / "mean" / "custom-predictor"),
                "num_files": 1,
            }
            for split in artifacts.SPLITS
        },
    }


def test_validate_posterior_means_requires_every_split(tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    for split in ("train", "valid"):
        output = dataset_root / split / "mean" / "NCSN++M"
        output.mkdir(parents=True)
        (output / "sample.wav").write_bytes((dataset_root / split / "clean" / "sample.wav").read_bytes())

    with pytest.raises(ValueError, match="Missing posterior-mean directory for test"):
        posterior.validate_posterior_means(_posterior_mean_config(dataset_root, "NCSN++M"))


@pytest.mark.parametrize("source", ["one/two", ".."])
def test_validate_posterior_means_rejects_non_directory_name(tmp_path: Path, source: str) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")

    with pytest.raises(ValueError, match="must name one directory"):
        posterior.validate_posterior_means(_posterior_mean_config(dataset_root, source))


def test_validate_posterior_means_rejects_filename_and_count_mismatch(tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    _write_matching_posterior_means(dataset_root, "NCSN++M")
    output = dataset_root / "valid" / "mean" / "NCSN++M"
    (output / "unexpected.wav").write_bytes(b"extra")

    with pytest.raises(ValueError, match=r"clean=1, mean=2"):
        posterior.validate_posterior_means(_posterior_mean_config(dataset_root, "NCSN++M"))


def test_generative_training_validates_posterior_means_before_creating_run(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    config = _posterior_mean_config(dataset_root, "missing-source")
    monkeypatch.setattr(
        generative_training,
        "prepare_generative_run",
        lambda *args, **kwargs: pytest.fail("a run must not be created when posterior means are invalid"),
    )

    with pytest.raises(ValueError, match="Missing posterior-mean directory for train"):
        generative_training.start_generative_training(config)


def test_evaluate_results_writes_core_metric_artifacts(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    run = make_run(tmp_path / "runs" / "predictive", "predictive", dataset_root)
    results_root = tmp_path / "results" / run.name
    patch_workflow_resolution(monkeypatch, run, dataset_root, results_root)
    output = results_root / "predictive"
    output.mkdir(parents=True)
    (output / "sample.wav").write_bytes((dataset_root / "test" / "noisy" / "sample.wav").read_bytes())
    artifacts.write_json(
        output / "inference.json",
        {
            "dataset_id": "example",
            "split": "test",
            "run_name": run.name,
            "model_sha256": run.model_sha256,
            "sample_rate": 16_000,
        },
    )

    class FakeMetric:
        def calculate(self, **kwargs):
            del kwargs
            return {"PESQ": 1.0, "ESTOI": 0.9, "SI_SDR": 8.0}

    monkeypatch.setattr(evaluation.MetricRegister, "fetch", lambda names: dict.fromkeys(names, FakeMetric))
    csv_path, json_path = evaluation.evaluate_directory(output)

    assert list(pd.read_csv(csv_path).columns) == ["filename", "PESQ", "ESTOI", "SI_SDR"]
    summary = json.loads(json_path.read_text())
    assert summary["summary"]["PESQ"]["mean"] == 1.0
    assert summary["metrics"] == ["pesq", "estoi", "si_sdr"]


def test_evaluate_external_writes_metrics_without_an_rsb_manifest(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_audio_dataset(tmp_path / "dataset")
    clean = dataset_root / "test" / "clean"
    noisy = dataset_root / "test" / "noisy"
    enhanced = tmp_path / "third-party"
    enhanced.mkdir()
    (enhanced / "sample.wav").write_bytes((noisy / "sample.wav").read_bytes())

    class FakeMetric:
        def calculate(self, **kwargs):
            del kwargs
            return {"SI_SDR": 9.0}

    monkeypatch.setattr(evaluation.MetricRegister, "fetch", lambda names: dict.fromkeys(names, FakeMetric))
    csv_path, json_path = evaluation.evaluate_external(clean, noisy, enhanced, metrics=("si_sdr",))

    assert csv_path.is_file()
    summary = json.loads(json_path.read_text())
    assert summary["source_type"] == "external"
    assert summary["summary"]["SI_SDR"]["mean"] == 9.0
