"""YAML-backed dynamic configuration helpers."""

from __future__ import annotations

import inspect
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml
from tabulate import tabulate

ConfigData = dict[str, Any]
ConfigClass = TypeVar("ConfigClass", bound=type[Any])
LEGACY_CONFIG_VERSION = "0.1.0"
CURRENT_CONFIG_VERSION = "1.0.0"
_SUPPORTED_CONFIG_VERSIONS = {LEGACY_CONFIG_VERSION, CURRENT_CONFIG_VERSION}


def _represent_ordered_dict(dumper: yaml.Dumper, data: OrderedDict[str, Any]) -> yaml.Node:
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, _represent_ordered_dict)


class BaseConfiguer:
    """Load and dump inherited YAML configuration files."""

    @classmethod
    def load(cls, file_path: str | os.PathLike[str]) -> OrderedDict[str, Any]:
        """Load YAML while recursively resolving relative ``inherit`` entries."""
        path = Path(file_path)
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError("file_path must point to a YAML file")

        with path.open(encoding="utf-8") as stream:
            origin = yaml.safe_load(stream) or {}
        if not isinstance(origin, dict):
            raise TypeError(f"Configuration root in {path} must be a mapping")

        merged: OrderedDict[str, Any] = OrderedDict()
        inherited = origin.get("inherit")
        if inherited is not None:
            if not isinstance(inherited, (str, list)):
                raise TypeError(f"The inherit field in {path} must be a string or list")
            inherited_paths = [inherited] if isinstance(inherited, str) else inherited
            for inherited_path in inherited_paths:
                resolved = Path(inherited_path)
                if not resolved.is_absolute():
                    resolved = path.parent / resolved
                merged.update(cls.load(resolved))

        for key, value in origin.items():
            if key == "inherit":
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._merge_mapping(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> OrderedDict[str, Any]:
        merged = OrderedDict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = BaseConfiguer._merge_mapping(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def dump(cls, data: Mapping[str, Any], output_path: str | os.PathLike[str]) -> None:
        """Write configuration data as YAML."""
        del cls
        with Path(output_path).open("w", encoding="utf-8") as stream:
            yaml.dump(dict(data), stream, default_flow_style=False, sort_keys=False)


def read_yml(yml_path: str | os.PathLike[str]) -> OrderedDict[str, Any]:
    """Read an inherited YAML mapping."""
    return BaseConfiguer.load(yml_path)


def read_config_from_yaml(config_path: str | os.PathLike[str]) -> Config:
    """Read a YAML file, or ``config.yml`` within a run directory."""
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.yml"
    if path.suffix not in {".yml", ".yaml"}:
        raise ValueError(f"config_path must point to a YAML file, not {path!s}")
    if not path.exists():
        raise ValueError(f"The config file {path} does not exist")
    return Config(migrate_config(read_yml(path)))


def migrate_config(values: Mapping[str, Any]) -> ConfigData:
    """Upgrade a supported persisted configuration to the current schema.

    Configurations created before versioned run metadata are treated as 0.1.0.
    Migration is in-memory only; callers decide whether to save the upgraded
    configuration beside the source checkpoint.
    """
    migrated = dict(values)
    source_version = str(migrated.get("version", LEGACY_CONFIG_VERSION))
    if source_version not in _SUPPORTED_CONFIG_VERSIONS:
        supported = ", ".join(sorted(_SUPPORTED_CONFIG_VERSIONS))
        raise ValueError(f"Unsupported RSB config version {source_version!r}; supported versions: {supported}")

    if source_version == LEGACY_CONFIG_VERSION:
        training_method = migrated.get("training_method", "none")
        migrated["training_method"] = training_method

        legacy_regularization = migrated.pop("regularization_type", None)
        if legacy_regularization is not None:
            migrated.setdefault("regularization_weight", legacy_regularization)
        if training_method == "regularization":
            migrated.setdefault("regularization_weight", "quadratic")
            migrated.setdefault("posterior_mean_from", "NCSN++M")

        legacy_logger = migrated.pop("log_with", None)
        if legacy_logger is not None:
            migrated.setdefault("logger", legacy_logger)
        migrated.pop("load_posterior_mean", None)

    migrated["version"] = CURRENT_CONFIG_VERSION
    return migrated


class Config:
    """Attribute-access wrapper around a dynamic configuration mapping."""

    _MAX_LENGTH = 100

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize configuration attributes from a mapping."""
        self.__dict__.update(config)

    def dict(self) -> ConfigData:
        """Return public configuration items as a dictionary."""
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def get(self, field: str, default: Any = None) -> Any:
        """Get a configuration value with a fallback."""
        return getattr(self, field, default)

    def update(self, config: Mapping[str, Any]) -> None:
        """Update configuration attributes from a mapping."""
        self.__dict__.update(config)

    def save(self, save_path: str | os.PathLike[str] | None = None, file_name: str = "config.yml") -> None:
        """Save configuration to a run directory."""
        destination = Path(save_path if save_path is not None else self.run_path)
        destination.mkdir(parents=True, exist_ok=True)
        self.update({"version": CURRENT_CONFIG_VERSION})
        BaseConfiguer.dump(self.dict(), destination / file_name)

    def print(self) -> None:
        """Print configuration as a compact table."""
        rows = [[str(key), self._truncate(str(value))] for key, value in self.dict().items()]
        print("Configuration:")
        print(tabulate(rows, headers=["Param", "Value"], tablefmt="pretty"))

    def _truncate(self, sentence: str) -> str:
        if len(sentence) <= self._MAX_LENGTH:
            return sentence
        return f"{sentence[: self._MAX_LENGTH - 4]}..."


def config_from_yaml(config_path: str, key_value: str | None = None) -> Callable[[ConfigClass], ConfigClass]:
    """Inject YAML values as constructor defaults for a decorated class."""

    def decorator(cls: ConfigClass) -> ConfigClass:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file {config_path} not found")
        configs: Mapping[str, Any] = read_yml(path)
        if key_value is not None:
            configs = configs[key_value]

        signature = inspect.signature(cls.__init__)
        parameters = signature.parameters
        defaults = {
            name: _parse_config_value(parameter.annotation, configs[name])
            for name, parameter in parameters.items()
            if name != "self" and name in configs
        }
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            merged = {
                name: parameter.default
                for name, parameter in parameters.items()
                if parameter.default is not inspect.Parameter.empty and name != "self"
            }
            merged.update(defaults)
            merged.update(kwargs)
            missing = [
                name
                for name, parameter in parameters.items()
                if name != "self" and parameter.default is inspect.Parameter.empty and name not in merged
            ]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")
            original_init(self, *args, **merged)

        cls.__init__ = new_init
        return cls

    return decorator


def _parse_config_value(expected_type: Any, value: Any) -> Any:
    """Convert a YAML value to its annotated constructor type."""
    if expected_type in {Any, inspect.Parameter.empty}:
        return value
    if expected_type is torch.device:
        return torch.device(value)
    if inspect.isclass(expected_type) and issubclass(expected_type, dict):
        return dict(value)
    return expected_type(value) if value is not None else None
