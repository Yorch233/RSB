from pathlib import Path

import yaml
from typer.testing import CliRunner

from RSB.cli import dataset as dataset_cli
from RSB.cli.app import app
from RSB.data.dataset_registry import (
    DATASET_SIGNALS,
    DATASET_SPLITS,
    add_dataset_entry,
    edit_dataset_entry,
    load_dataset_registry,
    remove_dataset_entry,
)
from RSB.utils.config import CURRENT_CONFIG_VERSION

runner = CliRunner()


def make_dataset(root: Path) -> Path:
    for split in DATASET_SPLITS:
        for signal in DATASET_SIGNALS:
            (root / split / signal).mkdir(parents=True)
    return root


def test_dataset_registry_add_edit_and_remove(tmp_path: Path) -> None:
    first = make_dataset(tmp_path / "first")
    second = make_dataset(tmp_path / "second")

    datasets = add_dataset_entry({}, "voicebank", first)
    assert datasets == {"voicebank": str(first.resolve())}

    datasets = edit_dataset_entry(datasets, "voicebank", new_id="vb-demand", path=second)
    assert datasets == {"vb-demand": str(second.resolve())}
    assert remove_dataset_entry(datasets, "vb-demand") == {}


def test_dataset_cli_manages_project_configuration(monkeypatch, tmp_path: Path) -> None:
    first = make_dataset(tmp_path / "first")
    second = make_dataset(tmp_path / "second")
    config_path = tmp_path / ".config" / "rsb.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("logger: none\n", encoding="utf-8")
    monkeypatch.setattr(dataset_cli, "USER_CONFIG_PATH", config_path)

    added = runner.invoke(app, ["dataset", "add", "--id", "voicebank", "--path", str(first)])
    assert added.exit_code == 0
    assert load_dataset_registry(config_path).selected_id == "voicebank"

    listed = runner.invoke(app, ["dataset", "list"])
    assert listed.exit_code == 0
    assert "voicebank" in listed.stdout
    assert first.name in listed.stdout

    edited = runner.invoke(
        app,
        ["dataset", "edit", "--id", "voicebank", "--new-id", "vb", "--path", str(second), "--select"],
    )
    assert edited.exit_code == 0
    assert load_dataset_registry(config_path).datasets == {"vb": str(second.resolve())}

    deleted = runner.invoke(app, ["dataset", "delete", "--id", "vb", "--yes"])
    assert deleted.exit_code == 0
    assert load_dataset_registry(config_path).datasets == {}
    assert load_dataset_registry(config_path).selected_id is None
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["logger"] == "none"
    assert raw["version"] == CURRENT_CONFIG_VERSION


def test_dataset_cli_rejects_duplicate_ids(monkeypatch, tmp_path: Path) -> None:
    dataset_root = make_dataset(tmp_path / "dataset")
    config_path = tmp_path / ".config" / "rsb.yml"
    monkeypatch.setattr(dataset_cli, "USER_CONFIG_PATH", config_path)

    first = runner.invoke(app, ["dataset", "add", "--id", "shared", "--path", str(dataset_root)])
    duplicate = runner.invoke(app, ["dataset", "add", "--id", "shared", "--path", str(dataset_root)])

    assert first.exit_code == 0
    assert duplicate.exit_code != 0
    assert "already registered" in duplicate.output
