"""Training entry points."""

from RSB.training.generative import start_generative_training
from RSB.training.predictive import start_predictive_training

__all__ = ["start_generative_training", "start_predictive_training"]
