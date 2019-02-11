from abc import ABC, abstractmethod
from subprocess import TimeoutExpired
from typing import List, Optional

from evowluator import config
from evowluator.config import Test as TestConfig
from evowluator.data.ontology import Ontology
from evowluator.pyutils import echo, fileutils
from evowluator.reasoner.base import ReasoningTask
from evowluator.reasoner.results import MatchmakingResults
from .base import ReasoningTest
from .test_mode import TestMode


class MatchmakingMeasurementTest(ReasoningTest, ABC):
    """Abstract test class for measuring stats of matchmaking tasks."""

    # Override

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        """Names for the columns of the CSV results."""
        pass

    @abstractmethod
    def extract_results(self, stats: MatchmakingResults) -> List:
        """Extract and log relevant results."""
        pass

    # Overrides

    @property
    def task(self) -> str:
        return ReasoningTask.MATCHMAKING

    def __init__(self,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None,
                 iterations: int = 1):
        super().__init__(dataset, reasoners, syntax)
        self._iterations = iterations

    def setup(self):
        csv_header = ['Resource', 'Request']

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        syntax = Ontology.Syntax.RDFXML

        if entry.request_count(syntax) == 0:
            self._logger.log('No available requests.')
            return

        resource = entry.ontology(syntax)

        for iteration in range(self._iterations):
            self._logger.log('Run {}:'.format(iteration + 1), color=echo.Color.YELLOW)
            self._logger.indent_level += 1

            for request in entry.requests(syntax):
                self._logger.log('Request: ', color=echo.Color.YELLOW, endl=False)
                self._logger.log(request.name)
                self._logger.indent_level += 1

                csv_row = [entry.name, request.name]

                for reasoner in self._usable_reasoners():
                    self._logger.log('- {}: '.format(reasoner.name), endl=False)
                    try:
                        stats = reasoner.matchmaking(resource.path, request.path,
                                                     timeout=TestConfig.TIMEOUT,
                                                     mode=self.mode)
                        csv_row.extend(self.extract_results(stats))
                    except TimeoutExpired:
                        csv_row.extend(['timeout'] * len(self.result_fields))
                        self._logger.log('timeout')
                    except Exception as e:
                        if config.DEBUG:
                            raise e

                        csv_row.extend(['error'] * len(self.result_fields))
                        self._logger.log('error')

                self._logger.indent_level -= 1
                self._logger.log('')
                self._csv_writer.write_row(csv_row)

            self._logger.indent_level -= 1
            self._logger.log('')


class MatchmakingPerformanceTest(MatchmakingMeasurementTest):

    # Overrides

    @property
    def mode(self) -> str:
        return TestMode.PERFORMANCE

    @property
    def result_fields(self) -> List[str]:
        return ['resource parsing', 'request parsing', 'reasoner init', 'reasoning', 'memory']

    def extract_results(self, stats: MatchmakingResults) -> List:
        self._logger.log('{:.0f} ms'.format(stats.total_ms))

        self._logger.indent_level += 1
        self._logger.log('Resource parsing {:.0f} ms'.format(stats.resource_parsing_ms))
        self._logger.log('Request parsing {:.0f} ms'.format(stats.request_parsing_ms))
        self._logger.log('Reasoner init {:.0f} ms'.format(stats.init_ms))
        self._logger.log('Reasoning {:.0f} ms'.format(stats.reasoning_ms))
        self._logger.log('Memory {}'.format(fileutils.human_readable_bytes(stats.max_memory)))
        self._logger.indent_level -= 1

        return [stats.resource_parsing_ms, stats.request_parsing_ms,
                stats.init_ms, stats.reasoning_ms, stats.max_memory]


class MatchmakingEnergyTest(MatchmakingMeasurementTest):

    # Overrides

    @property
    def mode(self) -> str:
        return TestMode.ENERGY

    @property
    def result_fields(self) -> List[str]:
        return ['energy']

    def extract_results(self, stats: MatchmakingResults) -> List:
        self._logger.log('{:.2f}'.format(stats.energy_score))
        return [stats.energy_score]
