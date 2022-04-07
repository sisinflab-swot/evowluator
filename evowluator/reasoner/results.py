from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Union

from pyutils.io.file import hex_hash as file_hash
from pyutils.proc.bench import Benchmark
from pyutils.proc.energy import EnergyProfiler
from pyutils.proc.task import Task
from pyutils.types.strenum import StrEnum
from pyutils.types.string import hex_hash as string_hash
from pyutils.types.unit import MemoryUnit, TimeUnit

EvaluationTask = Union[Task, Benchmark, EnergyProfiler]


class EnergyStats:
    """Contains energy statistics of a reasoning task.

    :ivar samples: Power samples.
    :ivar interval: Sampling interval in milliseconds.
    """

    def __init__(self, samples: Sequence[float], interval: int):
        self.samples = samples
        self.interval = interval

    def __bool__(self) -> bool:
        return True if self.samples and self.interval else False

    def score(self, task_duration: float) -> float:
        """
        Returns an energy impact score for a reasoning task of the specified duration.

        :param task_duration: Duration of the reasoning task in milliseconds.
        :return: Energy impact score.
        """
        if not self.samples:
            return 0.0

        interval = self.interval
        full_samples = list(self.samples)[:-1]

        # The last sample needs special treatment since the reported power estimate is normalized
        # based on the actual execution time. Example:
        #
        # Sampling interval:    500 ms      |-----|
        # Task runtime:        1800 ms      |-----|-----|-----|---  |
        # Samples:                          | 5.0 | 7.0 | 6.0 | 4.0 |
        #
        # We have three "full" samples, which can be used as-is. The last sample only accounts
        # for 60% of the sampling interval, therefore the profiler normalizes it as if it lasted
        # for the entire interval: reported power is therefore reduced by 40% w.r.t. actual
        # instantaneous power, which would be 4.0 * 1.4 = 5.6.
        #
        # We need to account for this if we have more samples than the total reported
        # reasoning time, which may happen if a reasoner has significant output runtime:
        # in this case, we compute the last sample based on actual runtime,
        # and normalize it by the same criterion used by the profiler.
        last_sample = self.samples[-1]

        full_samples_count = len(full_samples)
        full_samples_mean = sum(full_samples) / full_samples_count if full_samples_count else 0.0

        full_interval_count = int(task_duration // interval)
        unsampled_full_intervals = full_interval_count - full_samples_count

        if unsampled_full_intervals > 0:
            # We don't have enough samples to cover the whole duration of the reasoning task.
            # Add samples based on the current average.
            full_samples += [full_samples_mean] * unsampled_full_intervals
        elif unsampled_full_intervals < 0:
            # We have more samples than needed.
            # Discard the excess samples and recompute last_sample.
            last_interval_duration = task_duration % interval
            last_sample = full_samples[full_interval_count] * last_interval_duration / interval
            full_samples = full_samples[:full_interval_count]

        return (sum(full_samples) + last_sample) * interval / 1000.0


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


class Field:
    """Output field."""

    OUTPUT = 'output'
    """Output of the reasoning task."""

    PARSING = 'parsing'
    """Parsing time."""

    REASONING = 'reasoning'
    """Reasoning time."""

    MEMORY = 'memory'
    """Memory peak."""

    @classmethod
    def correctness(cls) -> List[str]:
        return [cls.OUTPUT]

    @classmethod
    def performance(cls) -> List[str]:
        return [cls.PARSING, cls.REASONING, cls.MEMORY]


class Results:
    """Contains results of a reasoning task.

    :ivar output:
        Output of the reasoning task.

    :ivar parsing:
        Parsing time in milliseconds.

    :ivar reasoning:
        Reasoning time in milliseconds.

    :ivar memory:
        Memory peak in bytes.

    :ivar energy:
        Energy statistics.
    """

    @property
    def total_time(self) -> float:
        return self.parsing + self.reasoning

    def __init__(self, output: Output | None = None, time_stats: Dict[str, float] | None = None,
                 memory: int = 0, energy: Dict[str, EnergyStats] | None = None) -> None:
        self.output = output
        self.parsing = float(sum(v for k, v in time_stats.items() if Field.PARSING in k))
        self.reasoning = float(sum(v for k, v in time_stats.items() if Field.PARSING not in k))
        self.memory = int(memory)
        self.energy = energy

    def get(self, what: str) -> int | float | str:
        if what == Field.OUTPUT:
            return self.output.data
        try:
            return self.energy[what].score(self.total_time)
        except KeyError:
            return getattr(self, what)

    def get_readable(self, what: str) -> str:
        if what in (Field.PARSING, Field.REASONING):
            return TimeUnit.MS(self.get(what)).readable().format(2)
        if what == Field.MEMORY:
            return MemoryUnit.B(self.memory).readable().format()
        if what == Field.OUTPUT:
            return self.output.data
        try:
            return f'{self.energy[what].score(self.total_time):.2f}'
        except KeyError:
            raise AttributeError(f'No value for attribute \'{what}\'')
