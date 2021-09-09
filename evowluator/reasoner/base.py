from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from pyutils import exc
from pyutils.io import fileutils
from pyutils.proc.bench import Benchmark, EnergyProbe, EnergyProfiler
from pyutils.proc.task import Task

from evowluator.config import Paths
from evowluator.data.ontology import Syntax
from evowluator.evaluation.mode import EvaluationMode
from evowluator.util import owltool
from evowluator.util.strenum import StrEnum
from .results import EnergyStats, EvaluationTask
from .results import Results


class OutputFormat(StrEnum):
    """Output format."""

    TEXT = 'text'
    """Text format."""

    ONTOLOGY = 'ontology'
    """Ontology format."""


class Reasoner(ABC):
    """Abstract reasoner interface."""

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
    def is_remote(self) -> bool:
        """True if the reasoner runs on a remote device, false otherwise."""
        return False

    @abstractmethod
    def args(self, task: ReasoningTask, mode: EvaluationMode,
             inputs: List[str], output: str | None) -> List[str]:
        """
        Command line arguments to pass to the reasoner executable for each task and evaluation mode.

        :param task: Reasoning task.
        :param mode: Evaluation mode.
        :param inputs: Input arguments.
        :param output: Output argument.
        :return: Command line arguments.
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

    @property
    def classification_output_format(self) -> OutputFormat:
        """Output format of the classification task."""
        return OutputFormat.ONTOLOGY

    def setup(self) -> None:
        """Called at the beginning of the evaluation."""
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
        if isinstance(task, Benchmark):
            return task.max_memory

        res = re.search(r'Memory: (.*) B', task.stdout)
        return int(res.group(1)) if res else 0

    def _parse_energy(self, task: EvaluationTask) -> EnergyStats:
        if isinstance(task, EnergyProfiler):
            return EnergyStats(task.samples, task.sampling_interval)

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

    def __repr__(self) -> str:
        return self.name

    def extract_results(self, task: Task, reasoner: Reasoner,
                        output: Optional[str], mode: EvaluationMode) -> Results:
        results = reasoner.parse_results(self, task)

        if not results.output:
            results.output = task.stdout

        return results

    def run(self, reasoner: Reasoner, inputs: Union[str, List[str]],
            output: Optional[str] = None,
            mode: EvaluationMode = EvaluationMode.CORRECTNESS,
            energy_probe: Optional[EnergyProbe] = None,
            timeout: float = 0.0) -> Results:
        if not isinstance(inputs, list):
            inputs = [inputs]

        for i in inputs:
            exc.raise_if_not_found(i, file_type=exc.FileType.FILE)

        if output:
            fileutils.remove(output)

        if reasoner.is_remote:
            inputs = [os.path.basename(f) for f in inputs]
            output = os.path.basename(output) if output else None

        # Run reasoner

        task = Task(Paths.absolute(reasoner.path), args=reasoner.args(self, mode, inputs, output))

        if mode == EvaluationMode.PERFORMANCE:
            task = Benchmark(task)
        elif mode == EvaluationMode.ENERGY:
            task = EnergyProfiler(task, energy_probe, sampling_interval=500)

        task.run(timeout=timeout if timeout else None).raise_if_failed()

        return self.extract_results(task, reasoner, output, mode)


class ClassificationTask(ReasoningTask):
    """Ontology classification reasoning task."""

    def extract_results(self, task: Task, reasoner: Reasoner,
                        output: Optional[str], mode: EvaluationMode) -> Results:
        results = super().extract_results(task, reasoner, output, mode).update_output(output, True)

        if (mode == EvaluationMode.CORRECTNESS and
                reasoner.classification_output_format == OutputFormat.ONTOLOGY):
            temp_path = os.path.splitext(output)[0]
            os.rename(output, temp_path)
            owltool.print_tbox(temp_path, output)

        return results


class ConsistencyTask(ReasoningTask):
    """Ontology consistency reasoning task."""

    def extract_results(self, task: Task, reasoner: Reasoner,
                        output: Optional[str], mode: EvaluationMode) -> Results:
        results = super().extract_results(task, reasoner, output, mode)

        if re.search(r'(not |in)consistent', results.output, re.IGNORECASE):
            results.output = 'not consistent'
        elif re.search(r'consistent', results.output, re.IGNORECASE):
            results.output = 'consistent'
        else:
            results.output = 'unknown'

        return results


class MatchmakingTask(ReasoningTask):
    """Matchmaking reasoning task."""

    def extract_results(self, task: Task, reasoner: Reasoner,
                        output: Optional[str], mode: EvaluationMode) -> Results:
        return super().extract_results(task, reasoner, output, mode).update_output(output, True)


ReasoningTask.CLASSIFICATION = ClassificationTask()
ReasoningTask.CONSISTENCY = ConsistencyTask()
ReasoningTask.MATCHMAKING = MatchmakingTask()
