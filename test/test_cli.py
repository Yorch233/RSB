import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from RSB.cli import train as train_cli
from RSB.cli.app import app
from RSB.utils.config import BaseConfiguer
from RSB.utils.paths import DEFAULT_CONFIG_PATH

runner = CliRunner()


def set_project_training_config(monkeypatch, tmp_path: Path) -> Path:
    config_path = tmp_path / ".config" / "rsb.yml"
    config_path.parent.mkdir()
    BaseConfiguer.dump(
        {
            "inherit": str(DEFAULT_CONFIG_PATH),
            "dataset": "voicebank",
            "datasets": {"voicebank": str(tmp_path / "dataset")},
            "run_dir": str(tmp_path / "runs"),
            "logger": "none",
            "mixed_precision": "none",
            "multi_gpu": False,
            "gpu_ids": [0],
            "log_steps": 10,
            "save_state_steps": 1000,
            "checkpoints_total_limit": 3,
            "ncsnpp_operator_backend": "pytorch_native",
            "ncsnpp_cuda_jit_status": "not_attempted",
        },
        config_path,
    )
    monkeypatch.setattr(train_cli.common, "USER_CONFIG_PATH", config_path)
    return config_path


def test_cli_app_keeps_lowercase_name() -> None:
    assert app.info.name == "rsb"


def test_root_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "RSB CLI" in result.stdout
    assert "████" not in result.stdout
    assert "Usage:" in result.stdout
    for command in ("config", "checkpoint", "inference", "train", "dataset", "metric", "webui"):
        assert command in result.stdout


def test_train_help_lists_predictive_command() -> None:
    result = runner.invoke(app, ["train", "predictive", "--help"])

    assert result.exit_code == 0
    assert "--method" in result.stdout
    assert "NCSN++M" in result.stdout
    assert "--max-epoch" in result.stdout
    assert "--logger" in result.stdout
    assert "--ema" in result.stdout
    assert "--yes" in result.stdout
    assert "CONFIG_PATH" not in result.stdout


def test_train_commands_are_split_without_changing_public_cli() -> None:
    assert train_cli.predictive.predictive.__module__ == "RSB.cli.train.predictive"
    assert train_cli.generative.generative.__module__ == "RSB.cli.train.generative"


def test_train_help_lists_generative_options() -> None:
    result = runner.invoke(app, ["train", "generative", "--help"])

    assert result.exit_code == 0
    for option in (
        "--schedule",
        "--training-method",
        "--config",
        "--max-epoch",
        "--logger",
        "--ema",
        "--yes",
    ):
        assert option in result.stdout
    assert "CONFIG_PATH" not in result.stdout


def test_dataset_help_lists_registry_commands() -> None:
    result = runner.invoke(app, ["dataset", "--help"])

    assert result.exit_code == 0
    for command in ("list", "add", "edit", "delete", "create", "inspect", "generate-mean"):
        assert command in result.stdout


def test_inference_help_is_model_symmetric_and_metric_accepts_a_directory() -> None:
    inference_result = runner.invoke(app, ["inference", "--help"])
    generative_result = runner.invoke(app, ["inference", "generative", "--help"])
    metric_result = runner.invoke(app, ["metric", "--help"])

    assert inference_result.exit_code == 0
    assert generative_result.exit_code == 0
    assert metric_result.exit_code == 0
    for command in ("predictive", "generative"):
        assert command in inference_result.stdout
    assert "--dir" in metric_result.stdout
    assert "--metrics" in metric_result.stdout
    assert "--predictive-run" not in generative_result.stdout


def test_generative_inference_uses_default_hub_run(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def run_inference(run, **kwargs) -> Path:
        captured.update({"run": run, **kwargs})
        return tmp_path / "results"

    monkeypatch.setattr("RSB.cli.inference.run_generative_inference", run_inference)
    result = runner.invoke(
        app,
        ["inference", "generative", "--dataset", "voicebank", "--device", "cpu", "--no-progress"],
    )

    assert result.exit_code == 0
    assert captured["run"] is None
    assert captured["dataset_id"] == "voicebank"
    assert captured["seed"] == 10


def test_metric_accepts_repeated_and_comma_separated_names(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(
        "RSB.cli.metric.evaluate_directory",
        lambda directory, **kwargs: (
            captured.update({"directory": directory, **kwargs}) or directory / "metrics.csv",
            directory / "metrics.json",
        ),
    )

    result = runner.invoke(
        app,
        ["metric", "--dir", str(tmp_path), "--metrics", "pesq,estoi", "--metrics", "si-sdr"],
    )

    assert result.exit_code == 0
    assert captured["metrics"] == ("pesq", "estoi", "si_sdr")


def test_metric_selects_a_named_result_from_a_run(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    result_dir = tmp_path / "results" / "demo" / "SDE_N=50"
    result_dir.mkdir(parents=True)
    (result_dir / "inference.json").write_text("{}")
    run = SimpleNamespace(name="demo", config=SimpleNamespace())
    captured = {}
    monkeypatch.setattr("RSB.cli.metric.resolve_run", lambda reference: run)
    monkeypatch.setattr("RSB.cli.metric.resolve_results_root", lambda config: tmp_path / "results")
    monkeypatch.setattr(
        "RSB.cli.metric.evaluate_directory",
        lambda directory, **kwargs: (
            captured.update({"directory": directory, **kwargs}) or directory / "metrics.csv",
            directory / "metrics.json",
        ),
    )

    result = runner.invoke(app, ["metric", "--run", "demo", "--result", "SDE_N=50"])

    assert result.exit_code == 0
    assert captured["directory"] == result_dir


def test_metric_accepts_three_third_party_directories(monkeypatch, tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("clean", "noisy", "enhanced")]
    for path in paths:
        path.mkdir()
    captured = {}
    monkeypatch.setattr(
        "RSB.cli.metric.evaluate_external",
        lambda clean, noisy, enhanced, **kwargs: (
            captured.update({"paths": (clean, noisy, enhanced), **kwargs}) or enhanced / "metrics.csv",
            enhanced / "metrics.json",
        ),
    )

    result = runner.invoke(
        app,
        ["metric", "--clean", str(paths[0]), "--noisy", str(paths[1]), "--enhanced", str(paths[2])],
    )

    assert result.exit_code == 0
    assert captured["paths"] == tuple(paths)


def test_train_rejects_unregistered_dataset_id(monkeypatch, tmp_path: Path) -> None:
    set_project_training_config(monkeypatch, tmp_path)

    result = runner.invoke(app, ["train", "predictive", "--dataset", "missing"])

    assert result.exit_code != 0
    assert "not registered" in result.output
    assert "voicebank" in result.output


def test_generative_cli_overrides_yaml_configuration(monkeypatch, tmp_path: Path) -> None:
    captured = []
    set_project_training_config(monkeypatch, tmp_path)

    def fake_start(config, **kwargs) -> None:
        captured.append((config, kwargs))

    monkeypatch.setattr("RSB.training.generative.start_generative_training", fake_start)
    result = runner.invoke(
        app,
        [
            "train",
            "generative",
            "--max-epoch",
            "7",
            "--learning-rate",
            "0.0002",
            "--dataset",
            "voicebank",
            "--run-name",
            "generative-test",
            "--logger",
            "none",
            "--no-ema",
            "--schedule",
            "VP",
            "--training-method",
            "regularization",
            "--posterior-mean-from",
            "custom-predictor",
            "--regularization-weight",
            "cosine",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    config, kwargs = captured[0]
    assert config.num_epoch == 7
    assert config.learning_rate == 0.0002
    assert config.run_name == "generative-test"
    assert config.logger == "none"
    assert config.ema is False
    assert config.bridge_type == "VP"
    assert config.training_method == "regularization"
    assert config.posterior_mean_from == "custom-predictor"
    assert config.regularization_weight == "cosine"
    assert config.ncsnpp_operator_backend == "pytorch_native"
    assert config.ncsnpp_cuda_jit_status == "not_attempted"
    assert "Training parameters" in result.stdout
    assert kwargs["checkpoint_path"] is None


def test_predictive_cli_uses_default_method_and_overrides(monkeypatch, tmp_path: Path) -> None:
    captured = []
    set_project_training_config(monkeypatch, tmp_path)

    def fake_start(config, **kwargs) -> None:
        captured.append((config, kwargs))

    monkeypatch.setattr("RSB.training.predictive.start_predictive_training", fake_start)
    result = runner.invoke(
        app,
        [
            "train",
            "predictive",
            "--max-epoch",
            "3",
            "--logger",
            "none",
            "--no-ema",
            "--run-name",
            "",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    config, kwargs = captured[0]
    assert config.num_epoch == 3
    assert config.logger == "none"
    assert config.ema is False
    assert config.run_name is None
    assert config.ncsnpp_operator_backend == "pytorch_native"
    assert "Training parameters" in result.stdout
    assert kwargs["method"] == "NCSN++M"


def test_predictive_cli_requires_final_confirmation_without_yes(monkeypatch, tmp_path: Path) -> None:
    captured = []
    set_project_training_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "RSB.training.predictive.start_predictive_training",
        lambda config, **kwargs: captured.append((config, kwargs)),
    )

    result = runner.invoke(app, ["train", "predictive"], input="2\n")

    assert result.exit_code != 0
    assert "Training parameters" in result.stdout
    assert "Start training with these parameters" in result.stdout
    assert not captured


def test_training_preflight_is_skipped_in_distributed_worker(monkeypatch, tmp_path: Path) -> None:
    config_path = set_project_training_config(monkeypatch, tmp_path)
    config, overrides = train_cli._load_config(config_path)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        "RSB.training.preflight.prepare_training_launch",
        lambda *args, **kwargs: pytest.fail("distributed workers must not prompt"),
    )

    worker_config, worker_overrides = train_cli._prepare_training_command(
        config,
        overrides,
        checkpoint_path=None,
        assume_yes=False,
    )

    assert worker_config is config
    assert worker_overrides is overrides


def test_generative_cli_uses_requested_defaults(monkeypatch, tmp_path: Path) -> None:
    captured = []
    set_project_training_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "RSB.training.generative.start_generative_training",
        lambda config, **kwargs: captured.append((config, kwargs)),
    )

    result = runner.invoke(app, ["train", "generative", "--yes"])

    assert result.exit_code == 0
    config, _ = captured[0]
    assert config.bridge_type == "VE"
    assert config.training_method == "regularization"
    assert config.posterior_mean_from == "NCSN++M"
    assert config.regularization_weight == "quadratic"


def test_generative_cli_uses_custom_run_config(monkeypatch, tmp_path: Path) -> None:
    captured = []
    set_project_training_config(monkeypatch, tmp_path)
    custom_config = tmp_path / "custom-run.yml"
    BaseConfiguer.dump(
        {
            "inherit": str(DEFAULT_CONFIG_PATH),
            "num_epoch": 7,
            "learning_rate": 0.0002,
            "batch_size": 3,
            "run_dir": str(tmp_path / "custom-runs"),
            "logger": "none",
        },
        custom_config,
    )
    monkeypatch.setattr(
        "RSB.training.generative.start_generative_training",
        lambda config, **kwargs: captured.append((config, kwargs)),
    )

    result = runner.invoke(app, ["train", "generative", "--config", str(custom_config), "--yes"])

    assert result.exit_code == 0
    config, _ = captured[0]
    assert config.num_epoch == 7
    assert config.learning_rate == 0.0002
    assert config.batch_size == 3
    assert config.run_dir == str(tmp_path / "custom-runs")
    assert config.logger == "none"
    assert config.dataset == "voicebank"
    assert config.datasets == {"voicebank": str(tmp_path / "dataset")}


def test_generative_none_disables_posterior_mean(monkeypatch, tmp_path: Path) -> None:
    captured = []
    set_project_training_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "RSB.training.generative.start_generative_training",
        lambda config, **kwargs: captured.append((config, kwargs)),
    )

    result = runner.invoke(app, ["train", "generative", "--training-method", "none", "--yes"])

    assert result.exit_code == 0
    config, kwargs = captured[0]
    assert config.training_method == "none"
    assert config.posterior_mean_from is None
    assert kwargs["overrides"]["posterior_mean_from"] is None


def test_generative_none_rejects_posterior_mean_source(monkeypatch, tmp_path: Path) -> None:
    set_project_training_config(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        ["train", "generative", "--training-method", "none", "--posterior-mean-from", "NCSN++M"],
    )

    assert result.exit_code != 0
    assert "not applicable" in result.stderr


def test_generative_cli_rejects_unsupported_training_method(monkeypatch, tmp_path: Path) -> None:
    set_project_training_config(monkeypatch, tmp_path)

    result = runner.invoke(app, ["train", "generative", "--training-method", "unsupported", "--yes"])

    assert result.exit_code != 0
    assert "regularization" in result.stderr


def test_dataset_create_exports_pairs_and_statistics(tmp_path: Path) -> None:
    clean_dir = tmp_path / "source_clean"
    noise_dir = tmp_path / "source_noise"
    output_dir = tmp_path / "dataset"
    (clean_dir / "test").mkdir(parents=True)
    noise_dir.mkdir()
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sf.write(clean_dir / "test" / "speech.wav", 0.2 * np.sin(2 * np.pi * 220 * time), sample_rate)
    sf.write(noise_dir / "noise.wav", 0.2 * np.sin(2 * np.pi * 997 * time), sample_rate)

    result = runner.invoke(
        app,
        [
            "dataset",
            "create",
            "--task",
            "enhancement",
            "--clean",
            "vctk",
            str(clean_dir),
            "--noise",
            "chime",
            str(noise_dir),
            "--output_dir",
            str(output_dir),
            "--snr-min",
            "0",
            "--snr-max",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "test" / "clean" / "000000_speech.wav").is_file()
    assert (output_dir / "test" / "noisy" / "000000_speech.wav").is_file()
    configuration = json.loads((output_dir / "create_configuraton.json").read_text())
    assert configuration["summary"]["num_files"] == 1
    assert configuration["summary"]["test_metrics"]["num_pairs"] == 1
    assert configuration["summary"]["average_measured_snr_db"] is not None
    assert configuration["attribution"]["project"] == "StoRM"
    assert configuration["export"]["clean_dataset"]["type"] == "vctk"
    assert configuration["export"]["noise_dataset"]["type"] == "chime"


def test_webui_status_explains_optional_component() -> None:
    result = runner.invoke(app, ["webui", "status"])

    assert result.exit_code == 0
    assert "not bundled" in result.stdout
