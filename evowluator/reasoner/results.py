import re
from typing import List, NamedTuple, Union

from evowluator.pyutils import exc
from evowluator.pyutils.proc import Benchmark, EnergyProfiler, Task

TestTask = Union[Task, Benchmark, EnergyProfiler]


class EnergyStats(NamedTuple):
    """Contains energy stats for a reasoning task."""
    samples: List[float]
    interval: int

    def score(self, task_duration: float) -> float:
        """Returns an energy consumption score for a reasoning task of the specified duration."""
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


class ReasoningStats(NamedTuple):
    """Contains stats about a reasoning task."""
    parsing_ms: float
    reasoning_ms: float
    max_memory: int
    energy_stats: EnergyStats

    @property
    def total_ms(self) -> float:
        return self.parsing_ms + self.reasoning_ms

    @property
    def energy_score(self) -> float:
        return self.energy_stats.score(self.total_ms)


class ConsistencyResults(NamedTuple):
    """Contains results for the consistency task."""
    consistent: bool
    stats: ReasoningStats


class MatchmakingResults(NamedTuple):
    """Contains results for the matchmaking task."""
    resource_parsing_ms: float
    request_parsing_ms: float
    init_ms: float
    reasoning_ms: float
    max_memory: int
    energy_stats: EnergyStats

    @property
    def total_ms(self) -> float:
        return (self.resource_parsing_ms + self.request_parsing_ms +
                self.init_ms + self.reasoning_ms)

    @property
    def energy_score(self) -> float:
        return self.energy_stats.score(self.total_ms)


class ResultsParser:
    """Parses reasoning task results."""

    # Public methods

    def parse_classification_results(self, task: TestTask) -> ReasoningStats:
        """Parses the results of the classification task."""
        return self._parse_reasoning_stats(task)

    def parse_consistency_results(self, task: TestTask) -> ConsistencyResults:
        """Parses the results of the consistency task."""
        stats = self._parse_reasoning_stats(task)

        result = re.search(r'The ontology is (.*)\.', task.stdout)
        exc.raise_if_falsy(result=result)
        consistent = (result.group(1) == 'consistent')

        return ConsistencyResults(consistent, stats)

    def parse_matchmaking_results(self, task: TestTask) -> MatchmakingResults:
        """Parses the result of the matchmaking task by parsing stdout."""
        stdout = task.stdout
        exc.raise_if_falsy(stdout=stdout)

        res = re.search(r'Resource parsing: (.*) ms', stdout)
        exc.raise_if_falsy(res=res)
        res_parsing_ms = float(res.group(1))

        res = re.search(r'Request parsing: (.*) ms', stdout)
        exc.raise_if_falsy(res=res)
        req_parsing_ms = float(res.group(1))

        res = re.search(r'Reasoner initialization: (.*) ms', stdout)
        exc.raise_if_falsy(res=res)
        init_ms = float(res.group(1))

        res = re.search(r'Reasoning: (.*) ms', stdout)
        exc.raise_if_falsy(res=res)
        reasoning_ms = float(res.group(1))

        return MatchmakingResults(resource_parsing_ms=res_parsing_ms,
                                  request_parsing_ms=req_parsing_ms,
                                  init_ms=init_ms,
                                  reasoning_ms=reasoning_ms,
                                  max_memory=self._parse_memory(task),
                                  energy_stats=self._parse_energy(task))

    # Protected methods

    def _parse_reasoning_stats(self, task: TestTask) -> ReasoningStats:
        """Parses stats for a reasoning task."""
        stdout = task.stdout
        exc.raise_if_falsy(stdout=stdout)

        res = re.search(r'Parsing: (.*) ms', stdout)
        exc.raise_if_falsy(res=res)
        parsing_ms = float(res.group(1))

        res = re.search(r'Reasoning: (.*) ms', stdout)
        exc.raise_if_falsy(res=res)
        reasoning_ms = float(res.group(1))

        return ReasoningStats(parsing_ms=parsing_ms,
                              reasoning_ms=reasoning_ms,
                              max_memory=self._parse_memory(task),
                              energy_stats=self._parse_energy(task))

    def _parse_memory(self, task: TestTask) -> int:
        """Parses the peak memory for a reasoning task."""
        if isinstance(task, Benchmark):
            return task.max_memory

        res = re.search(r'Memory: (.*) B', task.stdout)
        return int(res.group(1)) if res else 0

    def _parse_energy(self, task: TestTask) -> EnergyStats:
        """Parses the energy score for a reasoning task."""
        if isinstance(task, EnergyProfiler):
            return EnergyStats(task.samples, task.sampling_interval)

        res = re.search(r'Energy sampling interval: (.*) ms', task.stdout)
        interval = int(res.group(1)) if res else 0

        res = re.search(r'Energy samples: (.*)\n', task.stdout)
        samples = [float(r) for r in res.group(1).split(';')] if res else []

        return EnergyStats(samples, interval)
