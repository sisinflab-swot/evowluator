from abc import ABC, abstractmethod
from subprocess import TimeoutExpired
from typing import List, Optional

from evowluator import config
from evowluator.config import Test as TestConfig
from evowluator.data.ontology import Ontology
from evowluator.pyutils import echo, fileutils
from evowluator.reasoner.base import Reasoner, ReasoningTask
from .base import Test
from .enum import TestMode, TestName


class MatchmakingPerformanceTest(Test, ABC):
    """Matchmaking performance test."""

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        pass

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.NON_STANDARD)

    @abstractmethod
    def run_reasoner(self, reasoner: Reasoner, resource: str, request: str) -> List[str]:
        """Called every run, for each reasoner and each ontology.

        :return : Values for the CSV result fields.
        """
        pass

    def __init__(self,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None,
                 iterations: int = 1):
        super().__init__(dataset, reasoners, syntax)
        self._iterations = iterations

    def setup(self):
        csv_header = ['Resource', 'Request']

        for reasoner in self._reasoners:
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
                self._logger.log('Request: {}'.format(request.name))
                self._logger.indent_level += 1

                csv_row = [entry.name, request.name]

                for reasoner in self._reasoners:
                    self._logger.log('- {}: '.format(reasoner.name), endl=False)
                    try:
                        csv_row.extend(self.run_reasoner(reasoner, resource.path, request.path))
                    except TimeoutExpired:
                        csv_row.extend(['timeout'] * len(self.result_fields))
                        self._logger.log('timeout')
                    except Exception as e:
                        if config.DEBUG:
                            raise e

                        csv_row.extend(['error'] * len(self.result_fields))
                        self._logger.log('error')

                self._logger.indent_level -= 1
                self._csv_writer.write_row(csv_row)

            self._logger.indent_level -= 1
            self._logger.log('')


class MatchmakingTimeTest(MatchmakingPerformanceTest):
    """Matchmaking time test."""

    @property
    def name(self):
        return TestName.Matchmaking.TIME

    @property
    def result_fields(self):
        return ['resource parsing', 'request parsing', 'reasoner init', 'reasoning']

    def run_reasoner(self, reasoner, resource, request):
        stats = reasoner.matchmaking(resource, request,
                                     timeout=TestConfig.MATCHMAKING_TIMEOUT,
                                     mode=TestMode.TIME)

        self._logger.log(('Resource parsing {:.0f} ms | '
                          'Request parsing {:.0f} ms | '
                          'Reasoner init {:.0f} ms | '
                          'Reasoning {:.0f} ms').format(stats.resource_parsing_ms,
                                                        stats.request_parsing_ms,
                                                        stats.init_ms,
                                                        stats.reasoning_ms))

        return [stats.resource_parsing_ms, stats.request_parsing_ms,
                stats.init_ms, stats.reasoning_ms]


class MatchmakingMemoryTest(MatchmakingPerformanceTest):
    """Matchmaking memory test."""

    @property
    def name(self):
        return TestName.Matchmaking.MEMORY

    @property
    def result_fields(self):
        return ['memory']

    def run_reasoner(self, reasoner, resource, request):
        stats = reasoner.matchmaking(resource, request,
                                     timeout=TestConfig.MATCHMAKING_TIMEOUT,
                                     mode=TestMode.MEMORY)

        self._logger.log(fileutils.human_readable_bytes(stats.max_memory))
        return [stats.max_memory]


class MatchmakingEnergyTest(MatchmakingPerformanceTest):
    """Matchmaking energy test."""

    @property
    def name(self):
        return TestName.Matchmaking.ENERGY

    @property
    def result_fields(self):
        return ['energy']

    def run_reasoner(self, reasoner, resource, request):
        stats = reasoner.matchmaking(resource, request,
                                     timeout=TestConfig.MATCHMAKING_TIMEOUT,
                                     mode=TestMode.ENERGY)

        self._logger.log('{:.2f}'.format(stats.energy_score))

        return [stats.energy_score]
