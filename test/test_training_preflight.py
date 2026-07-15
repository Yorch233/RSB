import pytest

from RSB.cli import tui
from RSB.training import preflight
from RSB.utils.config import Config


def _hide_summary(monkeypatch) -> list[dict]:
    summaries = []
    monkeypatch.setattr(tui, "print_config_summary", lambda config, **kwargs: summaries.append(config.copy()))
    return summaries


def test_preflight_defaults_to_cuda_jit_and_records_available_status(monkeypatch) -> None:
    config = Config({"batch_size": 8})
    summaries = _hide_summary(monkeypatch)
    monkeypatch.setattr(preflight, "enable_cuda_jit", lambda: None)
    monkeypatch.setattr(tui, "prompt_yes_no", lambda *args, **kwargs: pytest.fail("--yes must skip prompts"))

    preflight.prepare_training_launch(config, assume_yes=True)

    assert config.ncsnpp_operator_backend == "cuda_jit"
    assert config.ncsnpp_cuda_jit_status == "available"
    assert summaries == [config.dict()]


def test_preflight_accepts_native_fallback_and_final_confirmation(monkeypatch) -> None:
    config = Config({"batch_size": 8})
    prompts = []
    native_selected = []
    summaries = _hide_summary(monkeypatch)

    def fail_cuda_jit() -> None:
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(preflight, "enable_cuda_jit", fail_cuda_jit)
    monkeypatch.setattr(preflight, "use_pytorch_native", lambda: native_selected.append(True))
    monkeypatch.setattr(
        tui,
        "prompt_yes_no",
        lambda question, **kwargs: prompts.append((question, kwargs["default"])) or True,
    )

    preflight.prepare_training_launch(config)

    assert config.ncsnpp_operator_backend == "pytorch_native"
    assert config.ncsnpp_cuda_jit_status == "failed"
    assert native_selected == [True]
    assert [default for _, default in prompts] == [True, True]
    assert "native PyTorch" in prompts[0][0]
    assert "Start training" in prompts[1][0]
    assert summaries == [config.dict()]


def test_preflight_stops_when_native_fallback_is_declined(monkeypatch) -> None:
    config = Config({})
    _hide_summary(monkeypatch)
    monkeypatch.setattr(preflight, "enable_cuda_jit", lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    monkeypatch.setattr(tui, "prompt_yes_no", lambda *args, **kwargs: False)

    with pytest.raises(preflight.TrainingCancelled, match="fallback was declined"):
        preflight.prepare_training_launch(config)


def test_preflight_honors_persisted_native_backend_without_retrying_jit(monkeypatch) -> None:
    config = Config(
        {
            "ncsnpp_operator_backend": "pytorch_native",
            "ncsnpp_cuda_jit_status": "failed",
        }
    )
    native_selected = []
    _hide_summary(monkeypatch)
    monkeypatch.setattr(preflight, "enable_cuda_jit", lambda: pytest.fail("persisted native backend must be reused"))
    monkeypatch.setattr(preflight, "use_pytorch_native", lambda: native_selected.append(True))

    preflight.prepare_training_launch(config, assume_yes=True)

    assert native_selected == [True]
    assert config.ncsnpp_cuda_jit_status == "failed"
