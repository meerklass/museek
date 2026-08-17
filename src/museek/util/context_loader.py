import pickle
from enum import Enum
from typing import Any

from ivory.plugin.abstract_plugin import AbstractPlugin


class ContextLoader:
    """Class to load a context file externally."""

    def __init__(self, context_path: str):
        """Initialise with the path to the context pickle `context_path`."""
        self._context_path = context_path

        with open(self._context_path, "rb") as input_file:
            self.context = pickle.load(input_file)

    def get_result(self, location: Enum) -> Any:
        """
        Return the result stored in `self.context` under `location`.
        Note: `self.context[location]` needs to be a `Result` object.
        """
        return self.context[location].result

    def requirements_dict(self, plugin: AbstractPlugin) -> dict[str, Any]:
        """Return the requirements of `plugin` as a dictionary."""
        result_dict = {}
        for requirement in plugin.requirements:
            result_dict[requirement.variable] = self.context[
                requirement.location
            ].result
        return result_dict
