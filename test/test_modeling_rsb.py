from pathlib import Path

import pytest
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from RSB.backbone import BackboneRegister
from RSB.data import STFTUtil
from RSB.modeling_rsb import RSB
from RSB.pipelines.generative import RSBTrainingPipeline
from RSB.utils.config import Config


@BackboneRegister.register("test_rsb_backbone")
class TestRSBBackbone(nn.Module):
    __test__ = False

    def __init__(self, input_channels: int, **kwargs) -> None:
        super().__init__()
        del input_channels, kwargs
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, inputs: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return inputs[:, :1] * self.scale


def make_model(**kwargs) -> RSB:
    return RSB(backbone="test_rsb_backbone", bridge_type="VE", device="cpu", **kwargs)


def test_rsb_owns_sde_samplers_and_spectral_sampling() -> None:
    model = make_model()
    observation = torch.ones(1, 1, 4, 4, dtype=torch.complex64)

    enhanced, trajectory, predictions = model.sample(observation, num_steps=1)

    assert model.sde is model.sde_sampler.sde
    assert model.sde is model.ode_sampler.sde
    assert model.sampler is model.sde_sampler
    assert enhanced.shape == observation.shape
    assert trajectory.shape[1] == 2
    assert predictions.shape[1] == 1


def test_rsb_hub_mixin_round_trip(tmp_path: Path) -> None:
    model = make_model()
    model.generator.scale.data.fill_(2.5)

    model.save_pretrained(tmp_path)
    restored = RSB.from_pretrained(tmp_path, map_location="cpu")

    assert isinstance(restored, PyTorchModelHubMixin)
    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "config.json").is_file()
    assert torch.equal(restored.generator.scale, model.generator.scale)


def test_rsb_loads_run_directory_config_yml(tmp_path: Path) -> None:
    model = make_model()
    model.save_pretrained(tmp_path)
    (tmp_path / "config.json").unlink()
    Config(
        {
            "generative_backbone": "test_rsb_backbone",
            "training_method": "none",
            "training_target": "data",
            "loss_weight_type": "constant",
            "bridge_type": "VE",
        }
    ).save(tmp_path)

    restored = RSB.from_pretrained(tmp_path, map_location="cpu")

    assert restored.backbone_name == "test_rsb_backbone"


def test_rsb_training_pipeline_owns_regularized_loss(monkeypatch) -> None:
    monkeypatch.setattr(STFTUtil, "istft", lambda spectrum: spectrum.real)
    pipeline = RSBTrainingPipeline(
        make_model(training_method="regularization"),
        learning_rate=1e-3,
        reduction="mean",
        time_loss_weight=1e-3,
    )
    clean = torch.ones(2, 1, 4, 4, dtype=torch.complex64)
    noisy = torch.full_like(clean, 2)
    posterior_mean = torch.full_like(clean, 1.5)

    losses = pipeline._shared_step((clean, noisy, posterior_mean))

    assert set(losses) == {"loss", "prediction_loss", "time_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    assert pipeline.configure_optimizers().param_groups[0]["lr"] == 1e-3


def test_rsb_training_pipeline_matches_reference_time_loss_reduction() -> None:
    predicted = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    target = torch.zeros_like(predicted)

    loss = RSBTrainingPipeline._time_loss(predicted, target)

    assert loss == torch.tensor(10.5)


def test_rsb_training_pipeline_uses_split_and_frequency_metric_names(monkeypatch) -> None:
    pipeline = RSBTrainingPipeline(make_model())
    losses = {
        "loss": torch.tensor(3.0),
        "prediction_loss": torch.tensor(2.0),
        "time_loss": torch.tensor(1.0),
    }
    logged = []
    monkeypatch.setattr(pipeline, "_shared_step", lambda batch: losses)
    monkeypatch.setattr(pipeline, "log", lambda name, value, **kwargs: logged.append((name, value, kwargs)))

    pipeline.training_step((), 0)
    pipeline.validation_step((), 0)

    assert [name for name, _, _ in logged] == [
        "train/loss",
        "train/loss_per_epoch",
        "train/prediction_loss_per_epoch",
        "train/time_loss_per_epoch",
        "valid/loss_per_epoch",
        "valid/prediction_loss_per_epoch",
        "valid/time_loss_per_epoch",
    ]
    assert logged[0][2]["on_step"] is True
    assert logged[0][2]["on_epoch"] is False
    assert all(entry[2]["on_epoch"] is True for entry in logged[1:])


def test_rsb_training_pipeline_requires_offline_posterior_mean(monkeypatch) -> None:
    monkeypatch.setattr(STFTUtil, "istft", lambda spectrum: spectrum.real)
    pipeline = RSBTrainingPipeline(make_model(training_method="regularization"))
    clean = torch.ones(2, 1, 4, 4, dtype=torch.complex64)
    noisy = torch.full_like(clean, 2)

    with pytest.raises(RuntimeError, match="offline posterior mean"):
        pipeline._shared_step((clean, noisy))


def test_rsb_rejects_unsupported_training_method() -> None:
    with pytest.raises(ValueError, match="training_method must be one of"):
        make_model(training_method="unsupported")
