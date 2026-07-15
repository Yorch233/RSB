from pathlib import Path

import pytest

from RSB.utils.config import CURRENT_CONFIG_VERSION, LEGACY_CONFIG_VERSION, BaseConfiguer, read_config_from_yaml
from RSB.workflows import artifacts, hub


def make_legacy_snapshot(root: Path) -> Path:
    snapshot = root / "snapshots" / "revision-sha"
    snapshot.mkdir(parents=True)
    BaseConfiguer.dump(
        {
            "version": LEGACY_CONFIG_VERSION,
            "run_name": "RSB_VE_05231038",
            "run_id": "legacy-run-id",
            "dataset": "voicebank",
            "datasets": {"voicebank": "/training-host/Voicebank+Demand"},
            "generative_backbone": "ncsnpp_base",
            "bridge_type": "VE",
            "training_method": "regularization",
            "training_target": "data",
            "regularization_type": "quadratic",
            "load_posterior_mean": True,
            "log_with": "wandb",
        },
        snapshot / "config.yml",
    )
    blob = root / "blobs" / "model-weights"
    blob.parent.mkdir()
    blob.write_bytes(b"model-weights")
    (snapshot / "model.safetensors").symlink_to(blob)
    return snapshot


def test_materialize_generative_run_migrates_legacy_config(monkeypatch, tmp_path: Path) -> None:
    snapshot = make_legacy_snapshot(tmp_path)
    observed = {}

    def download(**kwargs) -> str:
        observed.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(hub, "snapshot_download", download)
    run_path = hub.materialize_generative_run(cache_dir=tmp_path / "hub")
    config = read_config_from_yaml(run_path)

    assert observed["repo_id"] == hub.DEFAULT_GENERATIVE_MODEL_ID
    assert observed["allow_patterns"] == ["config.yml", "model.safetensors"]
    assert run_path == tmp_path / "hub" / "rsb-runs" / "Yorch233--RSB" / "revision-sha"
    assert (run_path / "model.safetensors").read_bytes() == b"model-weights"
    assert not (run_path / "model.safetensors").is_symlink()
    assert config.run_type == "generative"
    assert config.run_name == "Yorch233--RSB"
    assert config.source_run_name == "RSB_VE_05231038"
    assert config.training_method == "regularization"
    assert config.regularization_weight == "quadratic"
    assert config.posterior_mean_from == "NCSN++M"
    assert config.logger == "wandb"
    assert config.hub_model_id == "Yorch233/RSB"
    assert config.hub_revision == "revision-sha"
    assert config.version == CURRENT_CONFIG_VERSION
    assert "regularization_type" not in config.dict()
    assert "load_posterior_mean" not in config.dict()
    assert "datasets" not in config.dict()


def test_hub_migration_keeps_current_config_version(tmp_path: Path) -> None:
    migrated = hub.migrate_generative_config(
        {"version": CURRENT_CONFIG_VERSION, "training_method": "none"},
        model_id="example/rsb",
        revision="main",
        run_path=tmp_path,
    )

    assert migrated["version"] == CURRENT_CONFIG_VERSION


def test_materialize_generative_run_requires_model_and_config(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "revision-sha"
    snapshot.mkdir(parents=True)
    BaseConfiguer.dump({}, snapshot / "config.yml")
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: str(snapshot))

    with pytest.raises(ValueError, match="model.safetensors"):
        hub.materialize_generative_run(cache_dir=tmp_path / "hub")


def test_resolve_run_uses_default_hub_checkpoint(monkeypatch, tmp_path: Path) -> None:
    run_path = tmp_path / "cached-run"
    run_path.mkdir()
    BaseConfiguer.dump(
        {"run_type": "generative", "run_name": "Yorch233--RSB", "training_method": "regularization"},
        run_path / "config.yml",
    )
    (run_path / "model.safetensors").write_bytes(b"model")
    monkeypatch.setattr(artifacts, "materialize_generative_run", lambda: run_path)

    resolved = artifacts.resolve_run(None, "generative")
    explicit = artifacts.resolve_run(hub.DEFAULT_GENERATIVE_MODEL_ID, "generative")

    assert resolved.path == run_path
    assert explicit.path == run_path
    assert resolved.name == "Yorch233--RSB"
    with pytest.raises(ValueError, match="local run"):
        artifacts.resolve_run(None, "predictive")
