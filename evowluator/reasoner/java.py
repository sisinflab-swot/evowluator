from abc import ABC, abstractmethod
from typing import List, Optional

from evowluator.pyutils.proc import Benchmark, EnergyProfiler, Jar, Task
from evowluator.test.enum import TestMode
from .base import Reasoner


class JavaReasoner(Reasoner, ABC):
    """Abstract Java reasoner interface."""

    # Override

    @classmethod
    def is_template(cls) -> bool:
        return cls == JavaReasoner

    @property
    @abstractmethod
    def vm_opts(self) -> List[str]:
        """Options to pass to the Java VM."""
        pass

    # Protected methods

    def _run(self, args: List[str], timeout: Optional[float], mode: str) -> Task:
        path = self.absolute_path

        if mode == TestMode.MEMORY:
            task = Jar(path, jar_args=args, vm_opts=['-Xms1m'] + self.vm_opts)
        else:
            task = Jar(path, jar_args=args, vm_opts=self.vm_opts)

        if mode == TestMode.MEMORY:
            task = Benchmark(task)
        elif mode == TestMode.ENERGY:
            task = EnergyProfiler(task, sampling_interval=500)

        task.run(timeout=timeout)
        return task
