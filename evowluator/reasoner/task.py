from __future__ import annotations

from typing import List

from pyutils import inspect
from pyutils.proc.task import Task
from .results import Output, Results


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
    def expected_output_format(self) -> Output.Format:
        """Expected output format."""
        return Output.Format.TEXT

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
