from subprocess import TimeoutExpired

from evowluator.config import Test as TestConfig
from evowluator.pyutils import echo, fileutils
from evowluator.reasoner.base import ReasoningTask
from .base import StandardPerformanceTest, Test
from .enum import TestMode, TestName


class ConsistencyCorrectnessTest(Test):
    """Consistency correctness test."""

    @property
    def name(self):
        return TestName.Consistency.CORRECTNESS

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CONSISTENCY)

    def setup(self):
        csv_header = ['Ontology']
        csv_header.extend([r.name for r in self._reasoners])
        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        csv_row = [entry.name]

        # Check consistency
        for reasoner in self._reasoners:
            self._logger.log('{}: '.format(reasoner.name), endl=False)

            try:
                results = reasoner.consistency(entry.ontology(reasoner.preferred_syntax).path,
                                               timeout=TestConfig.TIMEOUT)
            except TimeoutExpired:
                result = 'timeout'
                color = echo.Color.RED
            except Exception:
                result = 'error'
                color = echo.Color.RED
            else:
                if results.consistent:
                    result = 'consistent'
                    color = echo.Color.GREEN
                else:
                    result = 'inconsistent'
                    color = echo.Color.RED

            self._logger.log(result, color=color)
            csv_row.append(result)

        self._csv_writer.write_row(csv_row)


class ConsistencyTimeTest(StandardPerformanceTest):
    """Consistency turnaround time test."""

    @property
    def name(self):
        return TestName.Consistency.TIME

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CONSISTENCY)

    @property
    def result_fields(self):
        return ['parsing', 'consistency']

    def run_reasoner(self, reasoner, ontology):

        results = reasoner.consistency(ontology.path,
                                       timeout=TestConfig.TIMEOUT,
                                       mode=TestMode.TIME)

        stats = results.stats
        message = 'Parsing {:.0f} ms | Consistency {:.0f} ms'.format(stats.parsing_ms,
                                                                     stats.reasoning_ms)
        self._logger.log(message)
        return [stats.parsing_ms, stats.reasoning_ms]


class ConsistencyMemoryTest(StandardPerformanceTest):
    """Consistency memory test."""

    @property
    def name(self):
        return TestName.Consistency.MEMORY

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CONSISTENCY)

    @property
    def result_fields(self):
        return ['memory']

    def run_reasoner(self, reasoner, ontology):
        results = reasoner.consistency(ontology.path,
                                       timeout=TestConfig.TIMEOUT,
                                       mode=TestMode.MEMORY)
        max_memory = results.stats.max_memory
        self._logger.log(fileutils.human_readable_bytes(max_memory))
        return [max_memory]


class ConsistencyEnergyTest(StandardPerformanceTest):
    """Consistency energy test."""

    @property
    def name(self):
        return TestName.Consistency.ENERGY

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CONSISTENCY)

    @property
    def result_fields(self):
        return ['energy']

    def run_reasoner(self, reasoner, ontology):
        stats = reasoner.consistency(ontology.path,
                                     timeout=TestConfig.TIMEOUT,
                                     mode=TestMode.ENERGY).stats

        self._logger.log('{:.2f}'.format(stats.energy_score))

        return [stats.energy_score]
