import pytest

from RSB.metrics import CompositeMetric, DistortionMetric, MetricRegister
from RSB.utils.register import Register


def test_register_decorator_and_fetch() -> None:
    registry = Register()

    @registry.register("example")
    class Example:
        pass

    assert registry.fetch("example") is Example


def test_register_rejects_missing_name() -> None:
    registry = Register()

    with pytest.raises(KeyError):
        registry.fetch("missing")


def test_metric_registry_keeps_distortion_and_composite_classes_distinct() -> None:
    assert MetricRegister.fetch("distortion") is DistortionMetric
    assert MetricRegister.fetch("composite") is CompositeMetric
    assert DistortionMetric is not CompositeMetric
