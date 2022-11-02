from __future__ import annotations

import re
from typing import List

from pyutils import inspect
from pyutils.io import file
from pyutils.proc.task import Task
from .results import Results
from .results import Output
from ..config import Evaluation
from ..evaluation.mode import EvaluationMode
from ..util import owltool


class ReasoningTask:
    """Reasoning task."""

    CLASSIFICATION: ReasoningTask = None
    CONSISTENCY: ReasoningTask = None
    MATCHMAKING: ReasoningTask = None

    __ALL: List[ReasoningTask] = None

    @classmethod
    def standard(cls) -> List[ReasoningTask]:
        """Standard reasoning tasks."""
        return [cls.with_name(n) for n in ('classification', 'consistency')]

    @classmethod
    def all(cls) -> List[ReasoningTask]:
        """All supported reasoning tasks."""
        if cls.__ALL is None:
            cls.__ALL = list(sorted((s() for s in inspect.subclasses(cls)), key=lambda r: r.name))
        return cls.__ALL

    @classmethod
    def with_name(cls, name: str) -> ReasoningTask:
        """Returns the reasoning task with the specified name."""
        try:
            name = name.lower()
            return next(r for r in cls.all() if r.name.lower() == name)
        except StopIteration:
            raise ValueError(f'No reasoning task named "{name}"')

    @property
    def name(self) -> str:
        """Name of the reasoning task."""
        name = type(self).__name__

        if name.endswith('Task'):
            name = name[:-4]

        return name.lower()

    @property
    def performance_fields(self) -> List[str]:
        """Output fields for the 'performance' evaluation mode."""
        return ['parsing', 'reasoning', 'memory']

    @property
    def requires_additional_inputs(self) -> bool:
        """True if the task requires additional inputs, other than the root ontology."""
        return False

    def __init__(self) -> None:
        setattr(ReasoningTask, self.name.upper(), self)

    def __repr__(self) -> str:
        return self.name

    def process_results(self, results: Results, task: Task) -> Results:
        """Override if you need to further process the results object."""
        return results


class ClassificationTask(ReasoningTask):
    """Ontology classification reasoning task."""

    def process_results(self, results: Results, task: Task) -> Results:
        super().process_results(results, task)

        if Evaluation.MODE != EvaluationMode.CORRECTNESS:
            return results

        if results.output.format == Output.Format.ONTOLOGY:
            owltool.print_taxonomy(results.output.path, results.output.path)
            results.output.format = Output.Format.TEXT

        return results


class ConsistencyTask(ReasoningTask):
    """Ontology consistency reasoning task."""

    def process_results(self, results: Results, task: Task) -> Results:
        super().process_results(results, task)

        if Evaluation.MODE != EvaluationMode.CORRECTNESS:
            return results

        if results.output.format == Output.Format.ONTOLOGY:
            # TODO: use owltool to detect collapsed taxonomies.
            raise ValueError('Unsupported output format.')
        elif results.output.format == Output.Format.TEXT:
            output = file.contents(results.output.path)
        else:
            output = results.output.data

        if re.search(r'(not |in)consistent', output, re.IGNORECASE):
            output = 'not consistent'
        elif re.search(r'consistent', output, re.IGNORECASE):
            output = 'consistent'
        else:
            output = 'unknown'

        results.output = Output(output, Output.Format.STRING)
        return results


class MatchmakingTask(ReasoningTask):
    """Matchmaking reasoning task."""

    @property
    def requires_additional_inputs(self) -> bool:
        return True
