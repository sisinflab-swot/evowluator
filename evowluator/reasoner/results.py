from __future__ import annotations

import filecmp
import re
from abc import ABC, abstractmethod
from typing import List, Union

from pyutils import exc
from pyutils.proc.bench import Benchmark, EnergyProfiler
from pyutils.proc.task import Task

EvaluationTask = Union[Task, Benchmark, EnergyProfiler]


class EnergyStats:
    """Contains energy statistics of a reasoning task."""

    @property
    def samples(self) -> List[float]:
        """Power samples."""
        return self._samples

    @property
    def interval(self) -> int:
        """Sampling interval in milliseconds."""
        return self._interval

    def __init__(self, samples: List[float], interval: int):
        self._samples = samples
        self._interval = interval

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


class ReasoningResults(ABC):
    """Contains results of a reasoning task."""

    # Override

    @property
    @abstractmethod
    def output(self) -> str:
        """
        Output of the reasoning task.
        Can be either the output itself, or the path to the file containing the output,
        depending on :attr:`output_is_file`.
        """
        pass

    @property
    @abstractmethod
    def output_is_file(self) -> bool:
        """
        If True, :attr:`output` is the path to a file containing the output
        of the reasoning task; otherwise, :attr:`output` contains actual reasoner output.
        """
        pass

    @property
    @abstractmethod
    def parsing_ms(self) -> float:
        """Parsing time in milliseconds."""
        pass

    @property
    @abstractmethod
    def reasoning_ms(self) -> float:
        """Reasoning time in milliseconds."""
        pass

    @property
    @abstractmethod
    def max_memory(self) -> int:
        """Memory peak in bytes."""
        pass

    @property
    @abstractmethod
    def energy_stats(self) -> EnergyStats:
        """Energy statistics."""
        pass

    # Public

    @property
    def total_ms(self) -> float:
        return self.parsing_ms + self.reasoning_ms

    @property
    def energy_score(self) -> float:
        return self.energy_stats.score(self.total_ms)

    @property
    def has_output(self) -> bool:
        return True if self.output else False

    @property
    def has_time_stats(self) -> bool:
        return self.parsing_ms > 0.0 and self.reasoning_ms > 0.0

    @property
    def has_memory_stats(self) -> bool:
        return self.max_memory > 0

    @property
    def has_performance_stats(self) -> bool:
        return self.has_time_stats and self.has_memory_stats

    @property
    def has_energy_stats(self) -> bool:
        return True if self.energy_stats else False

    def output_matches(self, results: ReasoningResults) -> bool:
        if self.output_is_file != results.output_is_file:
            return False

        try:
            if self.output_is_file:
                return filecmp.cmp(self.output, results.output, shallow=False)
            else:
                return self.output.strip() == results.output.strip()
        except Exception:
            return False


class StandardReasoningResults(ReasoningResults):
    """Contains results of a standard reasoning task."""

    @property
    def output(self) -> str:
        return self._output

    @property
    def output_is_file(self) -> bool:
        return self._output_is_file

    @property
    def parsing_ms(self) -> float:
        return self._parsing_ms

    @property
    def reasoning_ms(self) -> float:
        return self._reasoning_ms

    @property
    def max_memory(self) -> int:
        return self._max_memory

    @property
    def energy_stats(self) -> EnergyStats:
        return self._energy_stats

    def __init__(self, parsing_ms: float, reasoning_ms: float,
                 max_memory: int, energy_stats: EnergyStats,
                 output: str = '', output_is_file: bool = False) -> None:
        self._parsing_ms = parsing_ms
        self._reasoning_ms = reasoning_ms
        self._max_memory = max_memory
        self._energy_stats = energy_stats
        self._output = output
        self._output_is_file = output_is_file

    def with_output(self, output: str, is_file: bool) -> StandardReasoningResults:
        return StandardReasoningResults(output=output, output_is_file=is_file,
                                        parsing_ms=self.parsing_ms, reasoning_ms=self.reasoning_ms,
                                        max_memory=self.max_memory, energy_stats=self.energy_stats)


class MatchmakingResults(ReasoningResults):
    """Contains results of a matchmaking task."""

    @property
    def init_ms(self) -> float:
        """Reasoner initialization time in milliseconds."""
        return self._init_ms

    @property
    def matchmaking_ms(self) -> float:
        """Matchmaking time in milliseconds."""
        return self._matchmaking_ms

    @property
    def output(self) -> str:
        return self._output

    @property
    def output_is_file(self) -> bool:
        return self._output_is_file

    @property
    def parsing_ms(self) -> float:
        return self._parsing_ms

    @property
    def reasoning_ms(self) -> float:
        return self._init_ms + self._matchmaking_ms

    @property
    def max_memory(self) -> int:
        return self._max_memory

    @property
    def energy_stats(self) -> EnergyStats:
        return self._energy_stats

    def __init__(self, parsing_ms: float, init_ms: float,
                 matchmaking_ms: float, max_memory: int, energy_stats: EnergyStats,
                 output: str = '', output_is_file: bool = False) -> None:
        self._init_ms = init_ms
        self._matchmaking_ms = matchmaking_ms
        self._parsing_ms = parsing_ms
        self._max_memory = max_memory
        self._energy_stats = energy_stats
        self._output = output
        self._output_is_file = output_is_file

    def with_output(self, output: str, is_file: bool) -> MatchmakingResults:
        return MatchmakingResults(output=output, output_is_file=is_file,
                                  parsing_ms=self.parsing_ms,
                                  init_ms=self.init_ms,
                                  matchmaking_ms=self.matchmaking_ms,
                                  max_memory=self.max_memory,
                                  energy_stats=self.energy_stats)


class ResultsParser:
    """Parses results of reasoning tasks."""

    # Public methods

    def parse_classification_results(self, task: EvaluationTask) -> StandardReasoningResults:
        """
        Parses the results of a classification task.

        :param task: Classification task.
        :return: Results of the classification task.
        """
        return self._parse_reasoning_stats(task)

    def parse_consistency_results(self, task: EvaluationTask) -> StandardReasoningResults:
        """
        Parses the results of a consistency task.

        :param task: Consistency task.
        :return: Results of the consistency task.
        """
        results = self._parse_reasoning_stats(task)

        res = re.search(r'The ontology is (.*)\.', task.stdout)

        if res:
            res = 'consistent' if res.group(1) == 'consistent' else 'not consistent'
        else:
            res = 'unknown'

        return results.with_output(res, is_file=False)

    def parse_matchmaking_results(self, task: EvaluationTask) -> MatchmakingResults:
        """
        Parses the results of a matchmaking task.

        :param task: Matchmaking task.
        :return: Results of the matchmaking task.
        """
        stdout = task.stdout
        exc.raise_if_falsy(stdout=stdout)

        parsing_ms = 0.0

        regex = re.compile(r'[pP]arsing: (.*) ms')

        for res in regex.finditer(stdout):
            parsing_ms += float(res.group(1)) if res else 0.0

        res = re.search(r'Reasoner initialization: (.*) ms', stdout)
        init_ms = float(res.group(1)) if res else 0.0

        res = re.search(r'Reasoning: (.*) ms', stdout)
        matchmaking_ms = float(res.group(1)) if res else 0.0

        return MatchmakingResults(parsing_ms=parsing_ms,
                                  init_ms=init_ms,
                                  matchmaking_ms=matchmaking_ms,
                                  max_memory=self._parse_memory(task),
                                  energy_stats=self._parse_energy(task))

    # Protected methods

    def _parse_reasoning_stats(self, task: EvaluationTask) -> StandardReasoningResults:
        """Parses performance stats for a reasoning task."""
        stdout = task.stdout
        exc.raise_if_falsy(stdout=stdout)

        res = re.search(r'Parsing: (.*) ms', stdout)
        parsing_ms = float(res.group(1)) if res else 0.0

        res = re.search(r'Reasoning: (.*) ms', stdout)
        reasoning_ms = float(res.group(1)) if res else 0.0

        return StandardReasoningResults(parsing_ms=parsing_ms,
                                        reasoning_ms=reasoning_ms,
                                        max_memory=self._parse_memory(task),
                                        energy_stats=self._parse_energy(task))

    def _parse_memory(self, task: EvaluationTask) -> int:
        """Parses the peak memory for a reasoning task."""
        if isinstance(task, Benchmark):
            return task.max_memory

        res = re.search(r'Memory: (.*) B', task.stdout)
        return int(res.group(1)) if res else 0

    def _parse_energy(self, task: EvaluationTask) -> EnergyStats:
        """Parses the energy score for a reasoning task."""
        if isinstance(task, EnergyProfiler):
            return EnergyStats(task.samples, task.sampling_interval)

        res = re.search(r'Energy sampling interval: (.*) ms', task.stdout)
        interval = int(res.group(1)) if res else 0

        res = re.search(r'Energy samples: (.*)\n', task.stdout)
        samples = [float(r) for r in res.group(1).split(';')] if res else []

        return EnergyStats(samples, interval)
