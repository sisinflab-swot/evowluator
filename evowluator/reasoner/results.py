from __future__ import annotations

from typing import Dict, Union

from pyutils.io.file import hex_hash as file_hash
from pyutils.proc.bench import Benchmark
from pyutils.proc.energy import EnergyProfiler
from pyutils.proc.task import Task
from pyutils.types.strenum import StrEnum
from pyutils.types.string import hex_hash as string_hash
from pyutils.types.unit import MemoryUnit, TimeUnit

EvaluationTask = Union[Task, Benchmark, EnergyProfiler]


class Output:
    """Reasoner output.

    :ivar data:
        Output data.

    :ivar format:
        Output format.
    """

    class Format(StrEnum):
        """Output format."""

        STRING = 'string'
        """String format."""

        TEXT = 'text_file'
        """Text file format."""

        ONTOLOGY = 'ontology_file'
        """Ontology file format."""

    @property
    def is_file(self) -> bool:
        return self.format in (self.Format.TEXT, self.Format.ONTOLOGY)

    @property
    def path(self) -> str:
        if not self.is_file:
            raise AttributeError('Output is not a file path.')
        return self.data

    def __init__(self, data: str, fmt: Output.Format) -> None:
        self.data = data
        self.format = fmt

    def hash(self) -> str:
        if self.is_file:
            return file_hash(self.data)
        elif len(self.data) <= 40:
            return self.data
        else:
            return string_hash(self.data)


class Results:
    """Contains results of a reasoning task.

    :ivar output: Output of the reasoning task.
    :ivar times: Times in milliseconds.
    :ivar memory: Memory peak in bytes.
    :ivar energy: Energy statistics.
    """

    @property
    def total_time(self) -> float:
        return sum(self.times.values())

    def __init__(self, output: Output | None = None, time_stats: Dict[str, float] | None = None,
                 memory: int = 0, energy: Dict[str, float] | None = None) -> None:
        self.output = output
        self.times = time_stats
        self.memory = int(memory)
        self.energy = energy

    def get(self, what: str) -> int | float | str:
        if what == 'output':
            return self.output.data
        if what == 'total_time':
            return self.total_time
        if what in self.times:
            return self.times[what]
        if what in self.energy:
            return self.energy[what]
        return getattr(self, what)

    def get_readable(self, what: str) -> str:
        if what == 'memory':
            return MemoryUnit.B(self.memory).readable().format()
        if what == 'total_time':
            return TimeUnit.MS(self.total_time).readable().format(2)
        if what == 'output':
            return self.output.data
        if what in self.times:
            return TimeUnit.MS(self.get(what)).readable().format(2)
        try:
            return f'{self.energy[what]:.2f}'
        except KeyError:
            raise AttributeError(f'No value for attribute \'{what}\'')
