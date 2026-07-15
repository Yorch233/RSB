import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from RSB.cli import config as config_cli
from RSB.cli.app import app
from RSB.data.dataset_registry import DATASET_SIGNALS, DATASET_SPLITS, validate_dataset_root
from RSB.utils.config import CURRENT_CONFIG_VERSION, read_config_from_yaml
from RSB.utils.paths import PROJECT_ROOT, RESULTS_DIR, RUNS_DIR

runner = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def read_pty_until(descriptor: int, needle: str, transcript: bytearray, timeout: float = 10.0) -> None:
    """Read a pseudo-terminal until a prompt appears or fail with its transcript."""
    expected = needle.encode()
    start = len(transcript)
    deadline = time.monotonic() + timeout
    while expected not in transcript[start:]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(f"Timed out waiting for {needle!r}:\n{transcript.decode(errors='replace')}")
        readable, _, _ = select.select([descriptor], [], [], remaining)
        if not readable:
            continue
        try:
            chunk = os.read(descriptor, 4096)
        except OSError:
            chunk = b""
        if not chunk:
            pytest.fail(f"PTY closed while waiting for {needle!r}:\n{transcript.decode(errors='replace')}")
        transcript.extend(chunk)


def make_dataset(root: Path) -> Path:
    for split in DATASET_SPLITS:
        for signal in DATASET_SIGNALS:
            (root / split / signal).mkdir(parents=True)
    return root


def wizard_configuration(dataset_root: Path) -> dict:
    return {
        "dataset": "default",
        "datasets": {"default": str(dataset_root)},
        "mixed_precision": "bf16",
        "multi_gpu": True,
        "gpu_ids": [0, 2],
        "logger": "wandb",
        "log_steps": 20,
        "save_state_steps": 500,
        "checkpoints_total_limit": 4,
        "run_dir": "runs",
    }


def test_validate_dataset_root_requires_all_pair_directories(tmp_path: Path) -> None:
    dataset_root = make_dataset(tmp_path / "complete")
    assert validate_dataset_root(dataset_root) == dataset_root.resolve()

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="train/clean"):
        validate_dataset_root(incomplete)


@pytest.mark.parametrize(
    ("value", "multi_gpu", "expected"),
    [("all", True, "all"), ("all", False, "all"), ("0,2", True, [0, 2]), ("3", False, [3])],
)
def test_parse_gpu_ids(value: str, multi_gpu: bool, expected: object) -> None:
    assert config_cli.parse_gpu_ids(value, gpu_count=4, multi_gpu=multi_gpu) == expected


def test_parse_gpu_ids_rejects_invalid_cardinality() -> None:
    with pytest.raises(ValueError, match="at least two"):
        config_cli.parse_gpu_ids("0", gpu_count=4, multi_gpu=True)
    with pytest.raises(ValueError, match="exactly one"):
        config_cli.parse_gpu_ids("0,1", gpu_count=4, multi_gpu=False)


def test_dataset_manager_edits_an_entry_and_returns_to_list(monkeypatch, tmp_path: Path) -> None:
    original = make_dataset(tmp_path / "original")
    replacement = make_dataset(tmp_path / "replacement")
    selections = iter([0, 2])
    answers = iter(["renamed", str(replacement)])
    monkeypatch.setattr(config_cli, "rich_select", lambda prompt, choices, default=0: next(selections))
    monkeypatch.setattr(config_cli, "_select", lambda prompt, choices, default: "Edit")
    monkeypatch.setattr(
        config_cli,
        "prompt_validated_text",
        lambda label, validator, **kwargs: validator(next(answers)),
    )

    datasets = config_cli._manage_datasets({"original": str(original)})

    assert datasets == {"renamed": str(replacement.resolve())}


def test_dataset_manager_removes_an_entry_and_returns_to_empty_list(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_dataset(tmp_path / "dataset")
    selections = iter([0, 1])
    monkeypatch.setattr(config_cli, "rich_select", lambda prompt, choices, default=0: next(selections))
    monkeypatch.setattr(config_cli, "_select", lambda prompt, choices, default: "Remove")
    monkeypatch.setattr(config_cli, "prompt_yes_no", lambda question, default=False: True)

    datasets = config_cli._manage_datasets({"voicebank": str(dataset_root)})

    assert datasets == {}


def test_config_wizard_aborts_before_prompting_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(config_cli.torch.cuda, "device_count", lambda: 0)

    result = runner.invoke(app, ["config", "--force"])

    assert result.exit_code == 1
    assert "No CUDA GPUs were detected" in result.stdout
    assert "Paired dataset directory" not in result.stdout


def test_config_command_writes_inherited_rsb_yml(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_dataset(tmp_path / "dataset")
    configuration = wizard_configuration(dataset_root)
    monkeypatch.setattr(config_cli, "collect_configuration", lambda defaults: configuration)
    destination = tmp_path / ".config" / "rsb.yml"

    result = runner.invoke(app, ["config", "--output", str(destination), "--force"])

    assert result.exit_code == 0
    loaded = read_config_from_yaml(destination)
    assert loaded.dataset == "default"
    assert loaded.datasets == {"default": str(dataset_root)}
    assert loaded.mixed_precision == "bf16"
    assert loaded.multi_gpu is True
    assert loaded.gpu_ids == [0, 2]
    assert loaded.logger == "wandb"
    assert loaded.bridge_type == "VE"
    assert loaded.version == CURRENT_CONFIG_VERSION
    assert "Configuration saved" in result.stdout


def test_config_command_runs_complete_non_tty_wizard(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_dataset(tmp_path / "dataset")
    destination = tmp_path / ".config" / "rsb.yml"
    monkeypatch.setattr(config_cli.torch.cuda, "device_count", lambda: 2)
    answers = "\n".join(
        [
            "1",
            "1",
            "voicebank",
            str(dataset_root),
            "3",
            "1",
            "3",
            "1",
            "0,1",
            "2",
            "5",
            "7",
            "2",
            "",
        ]
    )

    result = runner.invoke(
        app,
        ["config", "--output", str(destination), "--force"],
        input=f"{answers}\n",
    )

    assert result.exit_code == 0
    loaded = read_config_from_yaml(destination)
    assert loaded.dataset == "voicebank"
    assert loaded.datasets == {"voicebank": str(dataset_root.resolve())}
    assert loaded.mixed_precision == "bf16"
    assert loaded.multi_gpu is True
    assert loaded.gpu_ids == [0, 1]
    assert loaded.logger == "none"
    assert loaded.log_steps == 5
    assert loaded.save_state_steps == 7
    assert loaded.checkpoints_total_limit == 2
    assert loaded.run_dir == str(RUNS_DIR)
    assert loaded.results_dir == str(RESULTS_DIR)
    assert "> Mixed precision" not in result.stdout
    assert "RSB Configuration Wizard" in result.stdout
    assert result.stdout.startswith("▱ RSB Configuration Wizard\n│\n◆ Detected GPUs: 2\n│")
    assert "◇ Configure the dataset registry?" in result.stdout
    assert "＋ Add dataset" in result.stdout
    assert "✓ Save and continue" in result.stdout
    assert "◇ What ID should identify this dataset?" in result.stdout
    assert "◇ Where is the paired dataset directory?" in result.stdout
    assert "◇ Which dataset should training use?" in result.stdout
    assert "│  Selected: voicebank" in result.stdout
    assert "│  Answer:" in result.stdout
    assert dataset_root.name in result.stdout
    assert "◇ Mixed precision?\n│  1. ● none\n│  2. ○ fp16\n│  3. ○ bf16" in result.stdout
    assert "◇ Mixed precision?\n│  Selected: bf16\n│" in result.stdout
    assert "◇ Enable multi-GPU training?" in result.stdout
    assert "Default: Yes" in result.stdout
    assert "◇ GPU IDs?" in result.stdout
    assert "│  Use comma-separated GPU IDs, or 'all'." in result.stdout
    assert "│  Enter the GPU selection, or press Enter to use the default." in result.stdout
    assert "◇ Log training loss every N steps?" in result.stdout
    assert "│  Enter a positive integer, or press Enter to use the default." in result.stdout
    assert "◇ Run directory?" in result.stdout
    assert "│  Enter a value, or press Enter to use the default." in result.stdout
    assert "> │" not in result.stdout
    assert ": │" not in result.stdout
    assert "████" not in result.stdout
    assert "╭" not in result.stdout
    assert "─" not in result.stdout


@pytest.mark.skipif(os.name != "posix", reason="pseudo-terminal interaction requires POSIX")
def test_config_command_supports_real_pty_arrow_navigation(tmp_path: Path) -> None:
    import pty

    dataset_root = make_dataset(tmp_path / "dataset")
    destination = tmp_path / ".config" / "rsb.yml"
    descriptor, child_terminal = pty.openpty()
    script = """
import sys
from pathlib import Path

from RSB.cli import config as config_cli

config_cli.torch.cuda.device_count = lambda: 2
config_cli.configure(Path(sys.argv[1]), force=True)
"""
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", script, str(destination)],
        stdin=child_terminal,
        stdout=child_terminal,
        stderr=child_terminal,
        cwd=PROJECT_ROOT,
        close_fds=True,
    )
    os.close(child_terminal)

    transcript = bytearray()
    try:
        read_pty_until(descriptor, "Configure the dataset registry", transcript)
        os.write(descriptor, b"\r")
        read_pty_until(descriptor, "Which dataset registry item", transcript)
        os.write(descriptor, b"\r")
        read_pty_until(descriptor, "What ID should identify this dataset", transcript)
        os.write(descriptor, b"voicebank\n")
        read_pty_until(descriptor, "Where is the paired dataset directory", transcript)
        invalid_attempt_start = len(transcript)
        os.write(descriptor, f"{tmp_path / 'missing'}\n".encode())
        read_pty_until(descriptor, "Dataset directory does not exist", transcript)
        read_pty_until(descriptor, "│  > ", transcript)
        os.write(descriptor, f"{dataset_root}\n".encode())
        read_pty_until(descriptor, "Which dataset registry item", transcript)
        os.write(descriptor, b"\r")
        read_pty_until(descriptor, "Which dataset should training use", transcript)
        os.write(descriptor, b"\r")
        read_pty_until(descriptor, "Mixed precision", transcript)
        os.write(descriptor, b"\x1b[B\x1b[B\r")
        read_pty_until(descriptor, "Enable multi-GPU training", transcript)
        os.write(descriptor, b"\r")
        read_pty_until(descriptor, "GPU IDs", transcript)
        os.write(descriptor, b"\n")
        read_pty_until(descriptor, "Experiment logger", transcript)
        os.write(descriptor, b"\x1b[B\r")
        read_pty_until(descriptor, "Log training loss every N steps", transcript)
        os.write(descriptor, b"\n")
        read_pty_until(descriptor, "Save resumable state every N steps", transcript)
        os.write(descriptor, b"\n")
        read_pty_until(descriptor, "Intermediate checkpoint limit", transcript)
        os.write(descriptor, b"\n")
        read_pty_until(descriptor, "Run directory", transcript)
        os.write(descriptor, b"\n")
        read_pty_until(descriptor, "Configuration saved", transcript)
    except BaseException:
        process.terminate()
        process.wait(timeout=10)
        raise
    finally:
        os.close(descriptor)

    assert process.wait(timeout=10) == 0
    loaded = read_config_from_yaml(destination)
    assert loaded.dataset == "voicebank"
    assert loaded.datasets == {"voicebank": str(dataset_root.resolve())}
    assert loaded.mixed_precision == "bf16"
    assert loaded.logger == "none"

    decoded = transcript.decode(errors="replace")
    rendered_with_cr = ANSI_ESCAPE.sub("", decoded)
    assert "◇ Mixed precision?\r\n│  1." in rendered_with_cr
    assert "\x1b[u\x1b[J" in decoded
    assert "What ID should identify this dataset" not in transcript[invalid_attempt_start:].decode(errors="replace")

    rendered = rendered_with_cr.replace("\r", "")
    assert "◇ Mixed precision?\n│  1. ● none\n│  2. ○ fp16\n│  3. ○ bf16" in rendered
    assert re.search(r"◇ Mixed precision\?\n│  Selected: bf16\n│", rendered)
    assert "████" not in rendered
    assert "╭" not in rendered
    assert "─" not in rendered


def test_collect_configuration_reuses_existing_values(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    observed: dict[str, object] = {}

    monkeypatch.setattr(config_cli.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        config_cli,
        "_configure_datasets",
        lambda current: (observed.setdefault("datasets", current["datasets"]), current["dataset"]),
    )

    def select(prompt: str, choices: list[str], default: int) -> str:
        observed[prompt] = default
        return choices[default]

    def devices(gpu_count: int, multi_gpu: bool, default: str | list[int]) -> str | list[int]:
        observed["gpu"] = (gpu_count, multi_gpu, default)
        return default

    def prompt_text(label: str, default: str | None = None, description: str | None = None) -> str:
        del description
        observed[label] = default
        return default or ""

    monkeypatch.setattr(config_cli, "_select", select)
    monkeypatch.setattr(config_cli, "prompt_yes_no", lambda question, default=False: default)
    monkeypatch.setattr(config_cli, "_prompt_devices", devices)
    monkeypatch.setattr(config_cli, "_prompt_positive_int", lambda prompt, default: default)
    monkeypatch.setattr(config_cli, "prompt_text", prompt_text)

    configuration = config_cli.collect_configuration(
        {
            "dataset": "saved",
            "datasets": {"saved": str(dataset_root)},
            "mixed_precision": "bf16",
            "multi_gpu": True,
            "gpu_ids": [1, 3],
            "logger": "none",
            "log_steps": 25,
            "save_state_steps": 500,
            "checkpoints_total_limit": 7,
            "run_dir": "custom-runs",
        }
    )

    assert observed["datasets"] == {"saved": str(dataset_root)}
    assert observed["Mixed precision"] == 2
    assert observed["Experiment logger"] == 1
    assert observed["gpu"] == (4, True, [1, 3])
    assert configuration["run_dir"] == str(PROJECT_ROOT / "custom-runs")


def test_collect_configuration_defaults_runs_to_project_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_cli.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(config_cli, "_configure_datasets", lambda current: ({}, None))
    monkeypatch.setattr(config_cli, "_select", lambda prompt, choices, default: choices[default])
    monkeypatch.setattr(config_cli, "prompt_yes_no", lambda question, default=False: default)
    monkeypatch.setattr(config_cli, "_prompt_devices", lambda gpu_count, multi_gpu, default: default)
    monkeypatch.setattr(config_cli, "_prompt_positive_int", lambda prompt, default: default)
    monkeypatch.setattr(config_cli, "prompt_text", lambda label, default=None, description=None: default or "")

    configuration = config_cli.collect_configuration()

    assert configuration["run_dir"] == str(RUNS_DIR)
    assert configuration["results_dir"] == str(RESULTS_DIR)


def test_configure_checks_expanded_destination_before_overwrite(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    destination = home / "existing.yml"
    destination.write_text("logger: none\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(config_cli, "prompt_yes_no", lambda question, default=False: False)

    with pytest.raises(typer.Abort):
        config_cli.configure(Path("~/existing.yml"), force=False)


def test_configure_requires_final_confirmation(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "rsb.yml"
    monkeypatch.setattr(config_cli, "collect_configuration", lambda defaults: wizard_configuration(tmp_path))
    monkeypatch.setattr(config_cli, "prompt_yes_no", lambda question, default=False: False)

    with pytest.raises(typer.Abort):
        config_cli.configure(destination, force=False)

    assert not destination.exists()
