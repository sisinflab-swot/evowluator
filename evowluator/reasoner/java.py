from abc import ABC, abstractmethod
from typing import List, Optional

from pyutils.proc.bench import Benchmark, EnergyProfiler, PowermetricsProbe
from pyutils.proc.task import Jar, Task

from evowluator.test.test_mode import TestMode
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
        path = self._absolute_path(self.path)
        task = Jar(path, jar_args=args, vm_opts=self.vm_opts)

        if mode == TestMode.PERFORMANCE:
            task = Benchmark(task)
        elif mode == TestMode.ENERGY:
            task = EnergyProfiler(task, self.energy_probe, sampling_interval=500)

        task.run(timeout=timeout)
        return task
