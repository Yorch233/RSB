"""Training command group and compatibility exports."""

import typer

from RSB.cli.train import common, generative, predictive
from RSB.cli.train.common import Logger, Optimizer, _load_config, _prepare_training_command, _validate_resume
from RSB.cli.train.generative import RegularizationWeight, Schedule, TrainingMethod, TrainingTarget
from RSB.cli.train.predictive import PredictiveMethod

app = typer.Typer(help="Train predictive and generative RSB models.")
app.command("predictive")(predictive.predictive)
app.command("generative")(generative.generative)

__all__ = [
    "Logger",
    "Optimizer",
    "PredictiveMethod",
    "RegularizationWeight",
    "Schedule",
    "TrainingMethod",
    "TrainingTarget",
    "_load_config",
    "_prepare_training_command",
    "_validate_resume",
    "app",
    "common",
    "generative",
    "predictive",
]
