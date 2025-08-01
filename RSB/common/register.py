# RSB/common/register.py
import warnings
from typing import Any, Callable, Dict, List, Union


class Register(dict):
    """
    A custom dictionary subclass designed to register and manage artifacts.
    Artifacts can be registered by type and name, allowing for easy retrieval.
    """

    def __init__(self, types_list: List[str] = None) -> None:
        """
        Initializes the ArtifactRegister object. Optionally initializes with a list of types.

        Parameters:
            types_list (List[str], optional): A list of types to initialize the register with. Defaults to None.
        """
        super().__init__()
        self._dict: Dict[str, Dict[str, Any]] = {}

        if types_list is not None:
            for artifact_type in types_list:
                self._dict[artifact_type] = {}

    def register(self, artifact_name: str) -> Callable[[Any], Any]:
        """
        Decorator method to register an artifact by its type and name.

        Parameters:
            artifact_name (str): The name of the artifact.

        Returns:
            Callable[[Any], Any]: A decorator function.
        """

        def decorator(artifact: Any) -> Any:
            self._dict[artifact_name] = artifact
            return artifact

        return decorator

    def fetch(self, name_or_name_list: Union[str, List[str]]) -> Any:
        """
        Retrieves a registered artifact by its type and name.

        Parameters:
            artifact_type (str): The type of the artifact.
            artifact_name (str): The name of the artifact.

        Returns:
            Any: The registered artifact.

        Raises:
            KeyError: If the artifact type or name is not registered.
        """
        if isinstance(name_or_name_list, list):
            artifacts = {}
            for name in name_or_name_list:
                if name in self._dict:
                    artifacts[name] = self._dict[name]
                else:
                    warnings.warn(
                        f"Unregistered artifact_name: '{name}'. Ignoring this artifact."
                    )
            if len(artifacts) == 0:
                raise ValueError("No registered artifacts found.")
            return artifacts
        elif isinstance(name_or_name_list, str):
            if name_or_name_list in self._dict:
                return self._dict[name_or_name_list]
            else:
                raise KeyError(
                    f"Unregistered artifact_name: '{name_or_name_list}'.")
