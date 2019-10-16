from abc import ABC, abstractmethod
from typing import List

from pyutils.proc.task import Jar, Task

from .base import Reasoner


class JavaReasoner(Reasoner, ABC):
    """Abstract Java reasoner interface."""

    @classmethod
    def is_template(cls) -> bool:
        return cls == JavaReasoner

    @property
    @abstractmethod
    def path(self) -> str:
        """Path to the Jar file of the reasoner."""
        pass

    @property
    @abstractmethod
    def vm_opts(self) -> List[str]:
        """Options to pass to the Java VM."""
        pass

    def _task(self, args: List[str]) -> Task:
        return Jar(self._absolute_path(self.path), jar_args=args, vm_opts=self.vm_opts)
