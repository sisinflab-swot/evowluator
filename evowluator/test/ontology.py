import os
from abc import ABC, abstractmethod
from subprocess import TimeoutExpired
from typing import List, Optional

from evowluator import config
from evowluator.pyutils import echo, fileutils
from evowluator.reasoner.results import ReasoningResults
from .base import ReasoningTest
from .test_mode import TestMode


class OntologyReasoningCorrectnessTest(ReasoningTest):
    """Test class for checking the correctness of reasoning tasks over ontologies."""

    @property
    def mode(self) -> str:
        return TestMode.CORRECTNESS

    def setup(self):
        csv_header = ['Ontology']
        reasoners = self._usable_reasoners()

        for reasoner in reasoners[1:]:
            csv_header.append(reasoner.name)

        self._csv_writer.write_row(csv_header)

    def run(self, entry):
        self.clear_temp()

        reasoners = self._usable_reasoners()
        reference = reasoners[0]
        reasoners = reasoners[1:]

        reasoner_out = os.path.join(self.temp_dir, 'reasoner.txt')
        reference_out = os.path.join(self.temp_dir, 'reference.txt')

        csv_row = [entry.name]

        # Reasoning
        self._logger.log('{}: '.format(reference.name), endl=False)
        self._logger.indent_level += 1

        ref_ontology = entry.ontology(self._syntax_for_reasoner(reference))
        ref_result = reference.perform_task(self.task, ref_ontology.path, output_file=reference_out,
                                            timeout=config.Test.TIMEOUT, mode=self.mode)

        self._logger.log('done', color=echo.Color.GREEN)

        for reasoner in reasoners:
            self._logger.log('{}: '.format(reasoner.name), endl=False)

            try:
                r_ontology = entry.ontology(self._syntax_for_reasoner(reasoner))
                r_result = reasoner.perform_task(self.task, r_ontology.path,
                                                 output_file=reasoner_out,
                                                 timeout=config.Test.TIMEOUT,
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
                if ref_result.output_matches(r_result):
                    result = 'same'
                    color = echo.Color.GREEN
                else:
                    result = 'different'
                    color = echo.Color.RED

            self._logger.log(result, color=color)
            csv_row.append(result)

        self._logger.indent_level -= 1
        self._csv_writer.write_row(csv_row)


class OntologyReasoningMeasurementTest(ReasoningTest, ABC):
    """Abstract test class for measuring stats of reasoning tasks over ontologies."""

    # Override

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        """Names for the columns of the CSV results."""
        pass

    @abstractmethod
    def extract_results(self, results: ReasoningResults) -> List:
        """Extract and log relevant results."""
        pass

    # Overrides

    def __init__(self,
                 task: str,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None,
                 iterations: int = 1):
        super().__init__(task=task, dataset=dataset, reasoners=reasoners, syntax=syntax)
        self.iterations = iterations

    def setup(self):
        csv_header = ['Ontology']

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        fail: List[str] = []

        for iteration in range(self.iterations):
            self._logger.log('Run {}:'.format(iteration + 1), color=echo.Color.YELLOW)
            self._logger.indent_level += 1

            csv_row = [entry.name]

            for reasoner in self._usable_reasoners():
                self._logger.log('- {}: '.format(reasoner.name), endl=False)
                ontology = entry.ontology(self._syntax_for_reasoner(reasoner))

                # Skip already failed or timed out.
                if reasoner.name in fail:
                    csv_row.extend(['skip'] * len(self.result_fields))
                    self._logger.log('skip')
                    continue

                try:
                    results = reasoner.perform_task(self.task, ontology.path,
                                                    timeout=config.Test.TIMEOUT, mode=self.mode)
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
                else:
                    csv_row.extend(self.extract_results(results))

            self._logger.indent_level -= 1
            self._logger.log('')
            self._csv_writer.write_row(csv_row)


class OntologyReasoningPerformanceTest(OntologyReasoningMeasurementTest):
    """Test class for measuring the performance of reasoning tasks over ontologies."""

    # Overrides

    @property
    def mode(self) -> str:
        return TestMode.PERFORMANCE

    @property
    def result_fields(self) -> List[str]:
        return ['parsing', 'reasoning', 'memory']

    def extract_results(self, results: ReasoningResults) -> List:
        self._logger.log('{:.0f} ms'.format(results.total_ms))

        self._logger.indent_level += 1
        self._logger.log('Parsing: {:.0f} ms'.format(results.parsing_ms))
        self._logger.log('Reasoning: {:.0f} ms'.format(results.reasoning_ms))
        self._logger.log('Memory: {}'.format(fileutils.human_readable_bytes(results.max_memory)))
        self._logger.indent_level -= 1

        return [results.parsing_ms, results.reasoning_ms, results.max_memory]


class OntologyReasoningEnergyTest(OntologyReasoningMeasurementTest, ABC):
    """Abstract test class for measuring energy used by reasoning tasks over ontologies."""

    # Overrides

    @property
    def mode(self) -> str:
        return TestMode.ENERGY

    @property
    def result_fields(self) -> List[str]:
        return ['energy']

    def extract_results(self, results: ReasoningResults) -> List:
        self._logger.log('{:.2f}'.format(results.energy_score))
        return [results.energy_score]
