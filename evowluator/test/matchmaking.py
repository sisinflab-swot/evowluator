import os
from abc import ABC, abstractmethod
from subprocess import TimeoutExpired
from typing import List, Optional

from pyutils.io import echo, fileutils

from evowluator import config
from evowluator.config import Test as TestConfig
from evowluator.data.dataset import Dataset
from evowluator.reasoner.base import ReasoningTask
from evowluator.reasoner.results import MatchmakingResults
from .base import ReasoningTest, ReasoningEnergyTest
from .test_mode import TestMode


class MatchmakingCorrectnessTest(ReasoningTest):
    """Test class for checking the correctness of non-standard reasoning."""

    @property
    def mode(self) -> str:
        return TestMode.CORRECTNESS

    def __init__(self,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None):
        super().__init__(ReasoningTask.MATCHMAKING, dataset, reasoners, syntax)

    def setup(self):
        reasoners = self._usable_reasoners()
        csv_header = ['Resource', 'Request'] + ['{}: match'.format(r.name) for r in reasoners[1:]]
        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        if entry.request_count() == 0:
            self._logger.log('No available requests.\n', color=echo.Color.YELLOW)
            return

        reasoners = self._usable_reasoners()
        reference = reasoners[0]
        reasoners = reasoners[1:]

        r_out = os.path.join(self.temp_dir, 'reasoner.txt')
        ref_out = os.path.join(self.temp_dir, 'reference.txt')
        ref_syntax = self._syntax_for_reasoner(reference)

        for request in entry.requests():
            self.clear_temp()

            self._logger.log('Request: ', color=echo.Color.YELLOW, endl=False)
            self._logger.log(request.name)
            self._logger.indent_level += 1
            self._logger.log('{}: '.format(reference.name), endl=False)

            csv_row = [entry.name, request.name]

            try:
                ref_resource, ref_request = entry.ontology(ref_syntax), request.ontology(ref_syntax)
                ref_result = reference.matchmaking(ref_resource.path, ref_request.path,
                                                   output_file=ref_out, timeout=TestConfig.TIMEOUT,
                                                   mode=self.mode)
            except Exception as e:
                if config.DEBUG:
                    raise e

                self._logger.log('error', color=echo.Color.RED)
                csv_row.append(['unknown'] * len(reasoners))
            else:
                self._logger.log('done', color=echo.Color.GREEN)
                self._logger.indent_level += 1

                for reasoner in reasoners:
                    self._logger.log('{}: '.format(reasoner.name), endl=False)

                    syntax = self._syntax_for_reasoner(reasoner)
                    resource_onto, request_onto = entry.ontology(syntax), request.ontology(syntax)

                    try:
                        r_result = reasoner.matchmaking(resource_onto.path, request_onto.path,
                                                        output_file=r_out,
                                                        timeout=TestConfig.TIMEOUT,
                                                        mode=self.mode)
                    except TimeoutExpired:
                        result = 'timeout'
                        color = echo.Color.RED
                    except Exception as e:
                        if config.DEBUG:
                            raise e
                        result = 'error'
                        color = echo.Color.RED
                    else:
                        if r_result.output_matches(ref_result):
                            result = 'same'
                            color = echo.Color.GREEN
                        else:
                            result = 'different'
                            color = echo.Color.RED

                    self._logger.log(result, color=color)
                    csv_row.append(result)

                self._logger.indent_level -= 1

            self._logger.indent_level -= 1
            self._logger.log('')
            self._csv_writer.write_row(csv_row)
        self._logger.log('')


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

    def __init__(self,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None,
                 iterations: int = 1):
        super().__init__(ReasoningTask.MATCHMAKING, dataset, reasoners, syntax)
        self._iterations = iterations

    def setup(self):
        csv_header = ['Resource', 'Request']

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)

    def run(self, entry: Dataset.Entry) -> None:

        if entry.request_count() == 0:
            self._logger.log('No available requests.\n', color=echo.Color.YELLOW)
            return

        for iteration in range(self._iterations):
            self._logger.log('Run {}:'.format(iteration + 1), color=echo.Color.YELLOW)
            self._logger.indent_level += 1

            for request in entry.requests():
                self._logger.log('Request: ', color=echo.Color.YELLOW, endl=False)
                self._logger.log(request.name)
                self._logger.indent_level += 1

                csv_row = [entry.name, request.name]

                for reasoner in self._usable_reasoners():
                    self._logger.log('{}: '.format(reasoner.name), endl=False)

                    syntax = self._syntax_for_reasoner(reasoner)
                    resource_onto = entry.ontology(syntax)
                    request_onto = request.ontology(syntax)

                    try:
                        stats = reasoner.matchmaking(resource_onto.path, request_onto.path,
                                                     timeout=TestConfig.TIMEOUT,
                                                     mode=self.mode)
                        csv_row.extend(self.extract_results(stats))
                    except TimeoutExpired:
                        csv_row.extend(['timeout'] * len(self.result_fields))
                        self._logger.log('timeout', color=echo.Color.RED)
                    except Exception as e:
                        if config.DEBUG:
                            raise e
                        csv_row.extend(['error'] * len(self.result_fields))
                        self._logger.log('error', color=echo.Color.RED)

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
        return ['parsing', 'init', 'reasoning', 'memory']

    def extract_results(self, results: MatchmakingResults) -> List:
        if not results.has_performance_stats:
            raise ValueError('Missing performance stats.')

        self._logger.log('{:.0f} ms'.format(results.total_ms))

        self._logger.indent_level += 1
        self._logger.log('Parsing: {:.0f} ms'.format(results.parsing_ms))
        self._logger.log('Init: {:.0f} ms'.format(results.init_ms))
        self._logger.log('Matchmaking: {:.0f} ms'.format(results.matchmaking_ms))
        self._logger.log('Memory: {}'.format(fileutils.human_readable_bytes(results.max_memory)))
        self._logger.indent_level -= 1

        return [results.parsing_ms, results.init_ms, results.matchmaking_ms, results.max_memory]


class MatchmakingEnergyTest(ReasoningEnergyTest, MatchmakingMeasurementTest):

    def __init__(self,
                 probe: str,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None,
                 iterations: int = 1):
        super().__init__(task=ReasoningTask.MATCHMAKING, probe=probe,
                         dataset=dataset, reasoners=reasoners, syntax=syntax, iterations=iterations)
