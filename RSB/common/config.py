# RSB/common/config.py
import inspect
import os
import os.path
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate.logging import get_logger
from tabulate import tabulate

# Add a custom YAML representer to preserve order in dictionaries
represent_dict_order = lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)

logger = get_logger(__name__)


class BaseConfiguer:
    """Base class for loading and dumping configuration files."""

    def __init__(self):
        pass

    @classmethod
    def load(cls, file_path: str):
        """
        Load configuration from a YAML file, supporting inheritance.

        Args:
            file_path (str): Path to the YAML config file.

        Returns:
            OrderedDict: Loaded configuration.
        """
        if '.yaml' not in file_path and '.yml' not in file_path:
            raise ValueError(
                f'The value of `file_path` should be a path to a yaml file.')
        root_path = os.path.dirname(os.path.abspath(file_path))

        def get_yaml_data(path: str):
            """Read and parse YAML file."""
            with open(path, encoding='utf-8') as file:
                return yaml.safe_load(file.read())

        def overwrite(ori: dict, con: dict) -> dict:
            """Recursively overwrite values from ori into con."""
            _con = OrderedDict()
            _con.update(con)
            for k, v in ori.items():
                if isinstance(v, dict) and k in _con:
                    _con[k] = overwrite(ori[k], _con[k])
                else:
                    _con[k] = v
            return _con

        _origin = get_yaml_data(file_path)
        _config = OrderedDict()

        # Handle inheritance
        if 'inherit' in _origin:
            inherit_ = _origin['inherit']
            if isinstance(inherit_, str) or isinstance(inherit_, list):
                if isinstance(inherit_, str):
                    inherit_paths = [inherit_]
                else:
                    inherit_paths = inherit_
                for inherit_path in inherit_paths:
                    if not os.path.isabs(inherit_path):
                        inherit_path = os.path.join(root_path, inherit_path)
                    _config.update(cls.load(inherit_path))
            else:
                raise TypeError(
                    f'The field of `inherit` in "{os.path.abspath(file_path)}" should be a string or dictionary.'
                )

        # Merge configurations
        for key, value in _origin.items():
            if key not in 'inherit':
                if isinstance(value, dict) and key in _config:
                    _config[key] = overwrite(_origin[key], _config[key])
                else:
                    _config[key] = value
        return _config

    @classmethod
    def dump(cls, data: any, output_path: str):
        """
        Dump configuration data to a YAML file.

        Args:
            data (any): Configuration data to write.
            output_path (str): Path to the output file.
        """
        with open(output_path, "w", encoding='utf-8') as fo:
            yaml.dump(data, fo, default_flow_style=False)


def read_yml(yml_path):
    """
    Load a YAML file using BaseConfiguer.

    Args:
        yml_path (str): Path to the YAML file.

    Returns:
        dict: Loaded configuration.
    """
    yml = BaseConfiguer.load(yml_path)
    return yml


def read_config_from_yaml(config_path: str):
    """
    Read configuration from a YAML file and return a Config object.

    Args:
        config_path (str): Path to the YAML file or directory containing it.

    Returns:
        Config: Configuration object.
    """
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'config.yml')
    if not config_path.endswith('yml') and not config_path.endswith('yaml'):
        raise ValueError(
            f'The value of `config_path` should be a path to a yaml file, not \'{config_path}\'.'
        )
    if not os.path.exists(config_path):
        raise ValueError(f'The config file `{config_path}` does not exist.')

    config = read_yml(config_path)
    return Config(config)


class Config:
    """Wrapper class for configuration dictionaries."""

    _MAX_LENGTH = 100

    def __init__(self, config):
        assert isinstance(config, dict), "Config must be a dictionary."
        self.__dict__.update(config)

    def dict(self):
        """Return public configuration items as a dictionary."""
        return {
            k: v
            for k, v in self.__dict__.items() if not k.startswith('_')
        }

    def get(self, field, default=None):
        """Get a configuration value with a default fallback."""
        return getattr(self, field, default)

    def update(self, config):
        """Update configuration with a new dictionary."""
        self.__dict__.update(config)

    def save(self, save_path=None, file_name='config.yml'):
        """
        Save configuration to a YAML file.

        Args:
            save_path (str, optional): Directory to save the file. Defaults to self.run_path.
            file_name (str, optional): Name of the output file. Defaults to 'config.yml'.
        """
        if save_path is None:
            save_path = self.run_path
        BaseConfiguer.dump(data=self.dict(),
                           output_path=os.path.join(save_path, file_name))

    def handleOvergLength(self, sentence: str, max_length: int) -> str:
        """Truncate long strings for display purposes."""
        sentence = sentence if len(
            sentence) < max_length else sentence[:max_length - 1 - 3] + '...'
        return sentence

    def print(self):
        """Print configuration in a formatted table."""
        con = self.dict()
        table_data = []
        for key, value in con.items():
            table_data.append([
                str(key),
                self.handleOvergLength(str(value), self._MAX_LENGTH)
            ])
        print('Configuration:')
        print(
            tabulate(table_data, headers=["Param", "Value"],
                     tablefmt="pretty"))


def config_from_yaml(config_path: str, key_value: str = None):
    """
    Decorator to inject default parameters from a YAML file into a class constructor.

    Args:
        config_path (str): Path to the YAML configuration file.
        key_value (str, optional): Optional key to extract a sub-dictionary from the config.

    Returns:
        function: Decorator function.
    """

    def decorator(cls):
        # Read the YAML configuration file
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"YAML file {config_path} not found")

        configs = read_yml(config_path)

        if key_value is not None:
            configs = configs[key_value]

        # Inspect the class __init__ signature
        init_signature = inspect.signature(cls.__init__)
        parameters = init_signature.parameters

        # Generate final default values with type conversion
        final_defaults = {}
        for name, param in parameters.items():
            if name == 'self':
                continue
            if name in configs:
                final_defaults[name] = _parse_config_value(
                    param.annotation, configs[name])

        # Override the class __init__ method
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Merge arguments: explicit > YAML > original defaults
            merged_kwargs = {}

            # Fill in original defaults
            for name, param in parameters.items():
                if param.default != inspect.Parameter.empty and name not in merged_kwargs:
                    merged_kwargs[name] = param.default

            # Apply YAML defaults
            merged_kwargs.update(final_defaults)
            # Apply user-provided arguments
            merged_kwargs.update(kwargs)

            # Check for missing required arguments
            required_params = [
                name for name, param in parameters.items()
                if param.default == inspect.Parameter.empty and name not in (
                    'self')
            ]
            missing = [p for p in required_params if p not in merged_kwargs]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")

            # Call the original __init__
            original_init(self, *args, **merged_kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


def _parse_config_value(expected_type: Any, value: Any) -> Any:
    """
    Safely convert configuration values to the expected type.

    Args:
        expected_type (Any): Expected type of the value.
        value (Any): Raw value from YAML.

    Returns:
        Any: Type-converted value.
    """
    if expected_type is torch.device:
        return torch.device(value)
    if inspect.isclass(expected_type) and issubclass(expected_type, dict):
        return dict(value)  # Ensure dictionary is serializable
    return expected_type(value) if value is not None else None