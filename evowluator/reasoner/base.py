from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List

from pyutils import inspect
from pyutils.proc.energy import EnergyProfiler
from pyutils.types.unit import MemoryUnit, TimeUnit
from .results import EvaluationTask, Output, Results
from .task import ReasoningTask
from ..data.syntax import Syntax


class Reasoner(ABC):
    """Abstract reasoner interface."""

    __ALL: List[Reasoner] = None
    __TIME_REGEX = re.compile(rf'^(.+):[ \t]*([\d.]+)[ \t]*({"|".join(TimeUnit.all())})[ \t]*$',
                              re.MULTILINE)

    @classmethod
    def all(cls) -> List[Reasoner]:
        """Returns all the available reasoners."""
        if cls.__ALL is None:
            cls.__ALL = list(sorted((s() for s in inspect.subclasses(cls) if not s.is_template()),
                                    key=lambda r: r.name))
        return cls.__ALL

    @classmethod
    def local(cls) -> List[Reasoner]:
        """Returns all the available local reasoners."""
        return [r for r in cls.all() if not isinstance(r, RemoteReasoner)]

    @classmethod
    def with_name(cls, name: str) -> Reasoner:
        """Returns the reasoner that has the specified name."""
        try:
            name = name.lower()
            return next(r for r in cls.all() if r.name.lower() == name)
        except StopIteration:
            raise ValueError(f'No reasoner named "{name}"')

    @classmethod
    def from_names(cls, names: List[str]) -> List[Reasoner]:
        """Returns the reasoners that have the specified names."""
        try:
            return [cls.with_name(n) for n in names]
        except KeyError as e:
            raise KeyError(f'No such reasoner: {str(e)}')

    @classmethod
    def supporting_task(cls, task: ReasoningTask) -> List[Reasoner]:
        """Returns the reasoners that support the specified reasoning task."""
        return [r for r in cls.all() if task in r.supported_tasks]

    # Override

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the reasoner."""
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        """
        Path to the reasoner binary, either absolute or relative
        to the `bin` subdirectory of the framework.
        """
        pass

    @property
    def supported_syntaxes(self) -> List[Syntax]:
        """
        OWL syntaxes supported by the reasoner.
        By default, this property returns all the syntaxes supported by evOWLuator.
        """
        return Syntax.all()

    @property
    def supported_tasks(self) -> List[ReasoningTask]:
        """
        Reasoning tasks supported by the reasoner.
        By default, this property returns the ontology classification and consistency tasks.
        """
        return ReasoningTask.standard()

    @property
    def preferred_syntax(self) -> Syntax:
        """
        Default syntax for the reasoner, used if the user does not specify a syntax explicitly
        via command line options. By default, this property returns the first entry
        in the :attr:`supported_syntaxes` list.
        """
        return self.supported_syntaxes[0]

    @abstractmethod
    def args(self, task: ReasoningTask, inputs: List[str], output: str) -> List[str]:
        """
        Command line arguments to pass to the reasoner executable for each task.

        :param task: Reasoning task.
        :param inputs: Input arguments.
        :param output: Output argument.
        :return: Command line arguments.
        """
        pass

    def setup(self) -> None:
        """Called at the beginning of the evaluation."""
        pass

    def pre_run(self, task: ReasoningTask, inputs: List[str], output: str) -> None:
        """
        Called before running each reasoning task.

        :param task: Reasoning task.
        :param inputs: Input arguments.
        :param output: Output arguments.
        """
        pass

    def post_run(self, task: ReasoningTask, inputs: List[str], output: str) -> None:
        """
        Called after running each reasoning task.

        :param task: Reasoning task.
        :param inputs: Input arguments.
        :param output: Output arguments.
        """
        pass

    def teardown(self) -> None:
        """Called at the end of the evaluation."""
        pass

    @classmethod
    def is_template(cls) -> bool:
        """
        The returned boolean signals whether the class is a template used for multiple reasoners.
        This is useful if, for example, several reasoners expose the same command line interface,
        in which case most of the logic can be reused by wrapping it in a template class
        that can be further subclassed.

        :return: True if this class is a template, False otherwise.
        """
        return cls == Reasoner

    def parse_results(self, rtask: ReasoningTask, etask: EvaluationTask, output: str) -> Results:
        """
        Parses the results of a reasoning task.

        Overriding this method is necessary if a reasoner offers no control over
        its output format.

        :param rtask: Reasoning task.
        :param etask: Evaluation task.
        :param output: Output argument as passed to the reasoner.
        :return: Results of the reasoning task.
        """
        output_fmt = rtask.expected_output_format
        if output_fmt == Output.Format.STRING:
            output = etask.stdout

        return Results(output=Output(output, output_fmt),
                       time_stats=self._parse_time(etask),
                       memory=self._parse_memory(etask),
                       energy=self._parse_energy(etask))

    # Protected methods

    def _parse_time(self, task: EvaluationTask) -> Dict[str, float]:
        times = {}

        for match in self.__TIME_REGEX.finditer(task.stdout):
            value = TimeUnit(match.group(3))(match.group(2)).to_value(TimeUnit.MS)
            times[match.group(1).strip().lower()] = value

        return times

    def _parse_memory(self, task: EvaluationTask) -> int:
        return task.max_memory if hasattr(task, 'max_memory') else 0

    def _parse_energy(self, task: EvaluationTask) -> Dict[str, float]:
        if isinstance(task, EnergyProfiler):
            return {p.name: task.score(p) for p in task.probes}
        return {}


class RemoteReasoner(Reasoner, ABC):
    """Abstract class for reasoners running on remote devices."""

    __MEMORY_REGEX = re.compile(rf'Memory:[ \t]*([\d.]+)[ \t]*({"|".join(MemoryUnit.all())})',
                                re.IGNORECASE)
    __SAMPLES_REGEX = re.compile(r'Samples:([\d.;\t ]+)$', re.MULTILINE | re.IGNORECASE)

    @classmethod
    def is_template(cls) -> bool:
        return cls == RemoteReasoner

    def _parse_memory(self, task: EvaluationTask) -> int:
        res = self.__MEMORY_REGEX.search(task.stdout)
        return int(MemoryUnit(res.group(2))(res.group(1)).to_value(MemoryUnit.B)) if res else 0

    def _parse_energy(self, task: EvaluationTask) -> Dict[str, float]:
        res = self.__SAMPLES_REGEX.search(task.stdout)
        samples = [float(r.strip()) for r in res.group(1).split(';')] if res else [0.0]
        return {'energy': sum(samples)}
