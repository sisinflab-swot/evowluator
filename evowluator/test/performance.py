from abc import ABC, abstractmethod
from subprocess import TimeoutExpired
from typing import List, Optional

from evowluator import config
from evowluator.data.ontology import Ontology
from evowluator.pyutils import echo, fileutils
from evowluator.reasoner.base import Reasoner
from evowluator.reasoner.results import ReasoningStats
from .base import ReasoningTest
from .test_mode import TestMode


class OntologyReasoningMeasurementTest(ReasoningTest, ABC):
    """Abstract test class for measuring stats of reasoning tasks over ontologies."""

    # Override

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        """Names for the columns of the CSV results."""
        pass

    @abstractmethod
    def run_reasoner(self, reasoner: Reasoner, ontology: Ontology) -> ReasoningStats:
        """Called every run, for each reasoner and each ontology."""
        pass

    @abstractmethod
    def extract_results(self, stats: ReasoningStats) -> List:
        """Extract and log relevant results."""
        pass

    # Overrides

    def __init__(self,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None,
                 iterations: int = 1):
        """
        :param dataset : If specified, run the test on the specified dataset.
        :param reasoners : If specified, limit the test to the specified reasoners.
        :param syntax : If true, the test is run on the specified OWL syntax whenever possible.
        :param iterations : Number of iterations per ontology.
        """
        super().__init__(dataset, reasoners, syntax)
        self.iterations = iterations

    def setup(self):
        csv_header = ['Ontology']

        for reasoner in self._reasoners:
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        fail: List[str] = []

        for iteration in range(self.iterations):
            self._logger.log('Run {}:'.format(iteration + 1), color=echo.Color.YELLOW)
            self._logger.indent_level += 1

            csv_row = [entry.name]

            for reasoner in self._reasoners:
                self._logger.log('- {}: '.format(reasoner.name), endl=False)
                ontology = entry.ontology(reasoner.syntax_for_requested(self._syntax))

                # Skip already failed or timed out.
                if reasoner.name in fail:
                    csv_row.extend(['skip'] * len(self.result_fields))
                    self._logger.log('skip')
                    continue

                try:
                    csv_row.extend(self.extract_results(self.run_reasoner(reasoner, ontology)))
                except TimeoutExpired:
                    csv_row.extend(['timeout'] * len(self.result_fields))
                    self._logger.log('timeout')
                    fail.append(reasoner.name)
                except Exception as e:
                    if config.DEBUG:
                        raise e

                    csv_row.extend(['error'] * len(self.result_fields))
                    self._logger.log('error')
                    fail.append(reasoner.name)

            self._logger.indent_level -= 1
            self._logger.log('')
            self._csv_writer.write_row(csv_row)


class OntologyReasoningPerformanceTest(OntologyReasoningMeasurementTest, ABC):
    """Abstract test class for measuring the performance of reasoning tasks over ontologies."""

    # Overrides

    @property
    def mode(self) -> str:
        return TestMode.PERFORMANCE

    @property
    def result_fields(self) -> List[str]:
        return ['parsing', 'reasoning', 'memory']

    def extract_results(self, stats: ReasoningStats) -> List:
        self._logger.log('{:.0f} ms'.format(stats.total_ms))

        self._logger.indent_level += 1
        self._logger.log('Parsing {:.0f} ms'.format(stats.parsing_ms))
        self._logger.log('Classification {:.0f} ms'.format(stats.reasoning_ms))
        self._logger.log('Memory {}'.format(fileutils.human_readable_bytes(stats.max_memory)))
        self._logger.indent_level -= 1

        return [stats.parsing_ms, stats.reasoning_ms, stats.max_memory]


class OntologyReasoningEnergyTest(OntologyReasoningMeasurementTest, ABC):
    """Abstract test class for measuring energy used by reasoning tasks over ontologies."""

    # Overrides

    @property
    def mode(self) -> str:
        return TestMode.ENERGY

    @property
    def result_fields(self) -> List[str]:
        return ['energy']

    def extract_results(self, stats: ReasoningStats) -> List:
        self._logger.log('{:.2f}'.format(stats.energy_score))
        return [stats.energy_score]
