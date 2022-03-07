from __future__ import annotations

import re
from typing import List

from pyutils.io import fileutils
from pyutils.proc.task import Task
from .results import Output
from .results import Results
from ..config import Evaluation
from ..evaluation.mode import EvaluationMode
from ..util import owltool


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
        """True if the task requires additional inputs, other than the root ontology."""
        return False

    def __repr__(self) -> str:
        return self.name

    def process_results(self, results: Results, task: Task) -> Results:
        return results


class ClassificationTask(ReasoningTask):
    """Ontology classification reasoning task."""

    def process_results(self, results: Results, task: Task) -> Results:
        if Evaluation.MODE != EvaluationMode.CORRECTNESS:
            return results

        if results.output.format == Output.Format.ONTOLOGY:
            owltool.print_taxonomy(results.output.path, results.output.path)
            results.output.format = Output.Format.TEXT

        return results


class ConsistencyTask(ReasoningTask):
    """Ontology consistency reasoning task."""

    def process_results(self, results: Results, task: Task) -> Results:
        if Evaluation.MODE != EvaluationMode.CORRECTNESS:
            return results

        if results.output.format == Output.Format.ONTOLOGY:
            # TODO: use owltool to detect collapsed taxonomies.
            raise ValueError('Unsupported output format.')
        elif results.output.format == Output.Format.TEXT:
            output = fileutils.file_contents(results.output.path)
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


ReasoningTask.CLASSIFICATION = ClassificationTask()
ReasoningTask.CONSISTENCY = ConsistencyTask()
ReasoningTask.MATCHMAKING = MatchmakingTask()
