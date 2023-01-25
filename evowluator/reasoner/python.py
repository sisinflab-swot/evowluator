from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

from pyutils.proc.task import find_executable
from .base import Reasoner, ReasoningTask
from ..config import Paths


class PythonReasoner(Reasoner, ABC):
    """Abstract Python reasoner interface."""

    @classmethod
    def is_template(cls) -> bool:
        return cls == PythonReasoner

    @property
    @abstractmethod
    def script_path(self) -> str:
        """Path to the main script of the reasoner."""
        pass

    @abstractmethod
    def script_args(self, task: ReasoningTask, inputs: List[str], output: str) -> List[str]:
        """Args to pass to the script."""
        pass

    @property
    def interpreter(self) -> str:
        """Python interpreter to use."""
        return 'python3'

    # Overrides

    @property
    def path(self) -> str:
        abs_interpreter = Paths.absolute(self.interpreter)

        if os.path.isfile(abs_interpreter):
            return abs_interpreter

        return find_executable(self.interpreter)

    def args(self, task: ReasoningTask, inputs: List[str], output: str) -> List[str]:
        return [Paths.absolute(self.script_path)] + self.script_args(task, inputs, output)
