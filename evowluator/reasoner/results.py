from __future__ import annotations

import filecmp
from typing import Dict, List, Union

from pyutils.io.fileutils import file_hash
from pyutils.stringutils import string_hash
from pyutils.proc.bench import Benchmark, EnergyProfiler
from pyutils.proc.task import Task

EvaluationTask = Union[Task, Benchmark, EnergyProfiler]


class EnergyStats:
    """Contains energy statistics of a reasoning task.

    :ivar samples: Power samples.
    :ivar interval: Sampling interval in milliseconds.
    """

    def __init__(self, samples: List[float], interval: int):
        self.samples = samples
        self.interval = interval

    def score(self, task_duration: float) -> float:
        """
        Returns an energy impact score for a reasoning task of the specified duration.

        :param task_duration: Duration of the reasoning task in milliseconds.
        :return: Energy impact score.
        """
        if not self.samples:
            return 0.0

        interval = self.interval
        full_samples = self.samples.copy()[:-1]

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


class Results:
    """Contains results of a reasoning task.

    :ivar output:
        Output of the reasoning task.
        Can be either the output itself, or the path to the file containing the output,
        depending on :attr:`output_is_file`.

    :ivar output_is_file:
        If True, :attr:`output` is the path to a file containing the output
        of the reasoning task; otherwise, :attr:`output` contains actual reasoner output.

    :ivar time_stats:
        Turnaround time (in milliseconds) of each reasoning phase by name.

    :ivar max_memory:
        Memory peak in bytes.

    :ivar energy_stats:
        Energy statistics.
    """

    @property
    def total_ms(self) -> float:
        return sum(self.time_stats.values())

    @property
    def parsing_ms(self) -> float:
        return sum(v for k, v in self.time_stats.items() if 'parsing' in k)

    @property
    def reasoning_ms(self) -> float:
        return sum(v for k, v in self.time_stats.items() if 'parsing' not in k)

    @property
    def energy_score(self) -> float:
        return self.energy_stats.score(self.total_ms)

    @property
    def has_output(self) -> bool:
        return True if self.output else False

    @property
    def has_time_stats(self) -> bool:
        return True if self.time_stats else False

    @property
    def has_memory_stats(self) -> bool:
        return self.max_memory > 0

    @property
    def has_performance_stats(self) -> bool:
        return self.has_time_stats and self.has_memory_stats

    @property
    def has_energy_stats(self) -> bool:
        return True if self.energy_stats else False

    def __init__(self, output: str | None = None, output_is_file: bool = False,
                 time_stats: Dict[str, float] | None = None, max_memory: int = 0,
                 energy_stats: EnergyStats | None = None) -> None:
        self.output = output
        self.output_is_file = output_is_file
        self.time_stats = time_stats if time_stats else {}
        self.max_memory = max_memory
        self.energy_stats = energy_stats

    def output_hash(self) -> str:
        return file_hash(self.output) if self.output_is_file else string_hash(self.output)

    def output_matches(self, results: Results) -> bool:
        if self.output_is_file != results.output_is_file:
            return False

        try:
            if self.output_is_file:
                return filecmp.cmp(self.output, results.output, shallow=False)
            else:
                return self.output.strip() == results.output.strip()
        except Exception:
            return False

    def update_output(self, output: str, output_is_file: bool = False) -> Results:
        self.output = output
        self.output_is_file = output_is_file
        return self
