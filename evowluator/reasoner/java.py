from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pyutils.proc.task import find_executable, java_args

from evowluator.config import Paths
from evowluator.evaluation.mode import EvaluationMode
from .base import Reasoner, ReasoningTask


class JavaReasoner(Reasoner, ABC):
    """Abstract Java reasoner interface."""

    @classmethod
    def is_template(cls) -> bool:
        return cls == JavaReasoner

    @property
    @abstractmethod
    def jar_path(self) -> str:
        """Path to the Jar file of the reasoner."""
        pass

    @property
    @abstractmethod
    def vm_opts(self) -> List[str] | None:
        """Options to pass to the Java VM."""
        pass

    @abstractmethod
    def jar_args(self, task: ReasoningTask, mode: EvaluationMode,
                 inputs: List[str], output: str | None) -> List[str] | None:
        """Args to pass to the Jar."""
        pass

    # Overrides

    @property
    def path(self) -> str:
        return find_executable('java')

    def args(self, task: ReasoningTask, mode: EvaluationMode,
             inputs: List[str], output: str | None) -> List[str]:
        return java_args(Paths.absolute(self.jar_path),
                         jar_args=self.jar_args(task, mode, inputs, output),
                         jvm_opts=self.vm_opts)
