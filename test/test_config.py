import pytest
import yaml

from RSB.utils.config import CURRENT_CONFIG_VERSION, LEGACY_CONFIG_VERSION, migrate_config, read_config_from_yaml
from RSB.utils.paths import DEFAULT_CONFIG_PATH


def test_default_configuration_inheritance() -> None:
    config = read_config_from_yaml(DEFAULT_CONFIG_PATH)

    assert config.sample_rate == 16_000
    assert config.n_fft == 510
    assert config.bridge_type == "VE"
    assert config.training_method == "regularization"
    assert config.posterior_mean_from == "NCSN++M"
    assert config.regularization_weight == "quadratic"
    assert config.mixed_precision == "none"
    assert config.logger == "wandb"
    assert config.seed == 10
    assert config.valid_metric_samples == 50
    assert config.test_metric_samples == 5
    assert config.metric_num_steps == 20
    assert config.version == CURRENT_CONFIG_VERSION
    assert config.get("load_posterior_mean") is None


def test_default_configuration_omits_derived_and_duplicate_fields() -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert raw["generative_backbone"] == "ncsnpp_base"
    assert "predictive_backbone" not in raw
    assert "evaluate_batch_size" not in raw
    assert "sample_rate" not in raw
    assert "num_nodes" not in raw


def test_legacy_configuration_migrates_to_current_schema() -> None:
    migrated = migrate_config(
        {
            "version": LEGACY_CONFIG_VERSION,
            "training_method": "regularization",
            "regularization_type": "quadratic",
            "log_with": "wandb",
            "load_posterior_mean": True,
        }
    )

    assert migrated["version"] == CURRENT_CONFIG_VERSION
    assert migrated["training_method"] == "regularization"
    assert migrated["regularization_weight"] == "quadratic"
    assert migrated["posterior_mean_from"] == "NCSN++M"
    assert migrated["logger"] == "wandb"
    assert "regularization_type" not in migrated
    assert "load_posterior_mean" not in migrated


def test_unsupported_configuration_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported RSB config version"):
        migrate_config({"version": "2.0.0"})


def test_default_configuration_excludes_machine_specific_fields() -> None:
    config = read_config_from_yaml(DEFAULT_CONFIG_PATH)
    for field in (
        "dataset",
        "datasets",
        "precision",
        "accelerator",
        "devices",
        "gpu_ids",
        "multi_gpu",
        "strategy",
        "wandb_log",
        "log_steps",
        "save_state_steps",
        "checkpoints_total_limit",
        "run_dir",
        "wechat_notify",
        "autodl_token",
    ):
        assert config.get(field) is None
