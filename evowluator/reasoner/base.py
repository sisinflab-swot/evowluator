from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List

from pyutils import exc
from pyutils.io import fileutils
from pyutils.inspectutils import get_subclasses
from pyutils.proc.bench import Benchmark, EnergyProfiler
from pyutils.proc.task import Task
from .results import EnergyStats, EvaluationTask
from .results import Results
from ..config import Evaluation, Paths
from ..data.syntax import Syntax
from ..evaluation.mode import EvaluationMode
from ..util import owltool
from ..util.strenum import StrEnum


class OutputFormat(StrEnum):
    """Output format."""

    TEXT = 'text'
    """Text format."""

    ONTOLOGY = 'ontology'
    """Ontology format."""


class Reasoner(ABC):
    """Abstract reasoner interface."""

    __ALL: List[Reasoner] = None

    @classmethod
    def all(cls) -> List[Reasoner]:
        """Returns all the available reasoners."""
        if cls.__ALL is None:
            cls.__ALL = list(sorted((s() for s in get_subclasses(cls) if not s.is_template()),
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
    def args(self, task: ReasoningTask, inputs: List[str], output: str | None) -> List[str]:
        """
        Command line arguments to pass to the reasoner executable for each task.

        :param task: Reasoning task.
        :param inputs: Input arguments.
        :param output: Output argument.
        :return: Command line arguments.
        """
        pass

    def output_format_for_task(self, task: ReasoningTask) -> OutputFormat:
        """
        Output format for each reasoning task.

        :param task: Reasoning task.
        :return: Output format.
        """
        return OutputFormat.ONTOLOGY if task == ReasoningTask.CLASSIFICATION else OutputFormat.TEXT

    def setup(self) -> None:
        """Called at the beginning of the evaluation."""
        pass

    def pre_run(self, task: ReasoningTask, inputs: List[str], output: str | None) -> None:
        """
        Called before running each reasoning task.

        :param task: Reasoning task.
        :param inputs: Input arguments.
        :param output: Output arguments.
        """
        pass

    def post_run(self, task: ReasoningTask, inputs: List[str], output: str | None) -> None:
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

    # noinspection PyUnusedLocal
    def parse_results(self, rtask: ReasoningTask, etask: EvaluationTask) -> Results:
        """
        Parses the results of a reasoning task.

        Overriding this method is necessary if a reasoner offers no control over
        its output format.

        :param rtask: Reasoning task.
        :param etask: Evaluation task.
        :return: Results of the reasoning task.
        """
        exc.raise_if_falsy(stdout=etask.stdout)
        return Results(time_stats=self._parse_time(etask),
                       max_memory=self._parse_memory(etask),
                       energy_stats=self._parse_energy(etask))

    # Protected methods

    def _parse_time(self, task: EvaluationTask) -> Dict[str, float]:
        times = {}

        for match in re.finditer(r'^(.*): (.*) ms$', task.stdout, re.MULTILINE):
            times[match.group(1).lower()] = float(match.group(2))

        return times

    def _parse_memory(self, task: EvaluationTask) -> int:
        return task.max_memory if hasattr(task, 'max_memory') else 0

    def _parse_energy(self, task: EvaluationTask) -> EnergyStats:
        if hasattr(task, 'samples') and hasattr(task, 'interval'):
            return EnergyStats(task.samples, task.interval)
        return EnergyStats([], 0)


class RemoteReasoner(Reasoner, ABC):
    """Abstract class for reasoners running on remote devices."""

    @classmethod
    def is_template(cls) -> bool:
        return cls == RemoteReasoner

    def _parse_memory(self, task: EvaluationTask) -> int:
        res = re.search(r'Memory: (.*) B', task.stdout)
        return int(res.group(1)) if res else 0

    def _parse_energy(self, task: EvaluationTask) -> EnergyStats:
        res = re.search(r'Energy sampling interval: (.*) ms', task.stdout)
        interval = int(res.group(1)) if res else 0

        res = re.search(r'Energy samples: (.*)\n', task.stdout)
        samples = [float(r) for r in res.group(1).split(';')] if res else []

        return EnergyStats(samples, interval)


class ReasoningTask:
    """Reasoning task."""

    CLASSIFICATION: ReasoningTask = None
    CONSISTENCY: ReasoningTask = None
    MATCHMAKING: ReasoningTask = None

    @classmethod
    def standard(cls) -> List[ReasoningTask]:
        """Standard reasoning tasks."""
        return [cls.CLASSIFICATION, cls.CONSISTENCY]

    @classmethod
    def all(cls) -> List[ReasoningTask]:
        """All supported reasoning tasks."""
        return cls.standard() + [cls.MATCHMAKING]

    @property
    def name(self) -> str:
        """Name of the reasoning task."""
        name = type(self).__name__

        if name.endswith('Task'):
            name = name[:-4]

        return name.lower()

    @property
    def requires_additional_inputs(self) -> bool:
        """True if the task requires additional input ontologies (other than the root ontology)."""
        return False

    def __repr__(self) -> str:
        return self.name

    def extract_results(self, task: Task, reasoner: Reasoner, output: str | None) -> Results:
        results = reasoner.parse_results(self, task)

        if not results.output:
            results.output = task.stdout

        return results

    def run(self, reasoner: Reasoner, inputs: str | List[str],
            output: str | None = None) -> Results:
        if not isinstance(inputs, list):
            inputs = [inputs]

        for i in inputs:
            exc.raise_if_not_found(i, file_type=exc.FileType.FILE)

        if output:
            fileutils.remove(output)

        # Run reasoner

        reasoner.pre_run(self, inputs, output)
        task = Task(Paths.absolute(reasoner.path), args=reasoner.args(self, inputs, output))

        if Evaluation.MODE == EvaluationMode.PERFORMANCE:
            if not isinstance(reasoner, RemoteReasoner):
                task = Benchmark(task)

            energy_probe = Evaluation.ENERGY_PROBE

            if energy_probe:
                interval = Evaluation.ENERGY_POLLING_INTERVALS.get(energy_probe.name, 1000)
                task = EnergyProfiler(task, energy_probe, interval=interval)

        task.run(timeout=Evaluation.TIMEOUT if Evaluation.TIMEOUT else None).raise_if_failed()
        results = self.extract_results(task, reasoner, output)
        reasoner.post_run(self, inputs, output)

        return results


class ClassificationTask(ReasoningTask):
    """Ontology classification reasoning task."""

    def extract_results(self, task: Task, reasoner: Reasoner, output: str | None) -> Results:
        results = super().extract_results(task, reasoner, output).update_output(output, True)

        if (Evaluation.MODE == EvaluationMode.CORRECTNESS and
                reasoner.output_format_for_task(self) == OutputFormat.ONTOLOGY):
            temp_path = os.path.splitext(output)[0]
            os.rename(output, temp_path)
            owltool.print_tbox(temp_path, output)

        return results


class ConsistencyTask(ReasoningTask):
    """Ontology consistency reasoning task."""

    def extract_results(self, task: Task, reasoner: Reasoner, output: str | None) -> Results:
        results = super().extract_results(task, reasoner, output)

        if re.search(r'(not |in)consistent', results.output, re.IGNORECASE):
            results.output = 'not consistent'
        elif re.search(r'consistent', results.output, re.IGNORECASE):
            results.output = 'consistent'
        else:
            results.output = 'unknown'

        return results


class MatchmakingTask(ReasoningTask):
    """Matchmaking reasoning task."""

    @property
    def requires_additional_inputs(self) -> bool:
        return True

    def extract_results(self, task: Task, reasoner: Reasoner, output: str | None) -> Results:
        return super().extract_results(task, reasoner, output).update_output(output, True)


ReasoningTask.CLASSIFICATION = ClassificationTask()
ReasoningTask.CONSISTENCY = ConsistencyTask()
ReasoningTask.MATCHMAKING = MatchmakingTask()
