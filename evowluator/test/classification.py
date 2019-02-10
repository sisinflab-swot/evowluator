import filecmp
import os
from subprocess import TimeoutExpired

from evowluator.config import Test as TestConfig
from evowluator.data.ontology import Ontology
from evowluator.pyutils import echo
from evowluator.reasoner.base import Reasoner, ReasoningTask
from evowluator.reasoner.results import ReasoningStats
from .base import ReasoningTest
from .test_mode import TestMode
from .performance import OntologyReasoningEnergyTest, OntologyReasoningPerformanceTest


class ClassificationCorrectnessTest(ReasoningTest):
    """Classification correctness test."""

    @property
    def task(self) -> str:
        return ReasoningTask.CLASSIFICATION

    @property
    def mode(self) -> str:
        return TestMode.CORRECTNESS

    def setup(self):
        csv_header = ['Ontology']

        for reasoner in self._reasoners[1:]:
            csv_header.append(reasoner.name)

        self._csv_writer.write_row(csv_header)

    def run(self, entry):
        self.clear_temp()

        reference = self._reasoners[0]
        reasoners = self._reasoners[1:]

        reasoner_out = os.path.join(self.temp_dir, 'reasoner.txt')
        reference_out = os.path.join(self.temp_dir, 'reference.txt')

        csv_row = [entry.name]

        # Classify
        self._logger.log('{}: '.format(reference.name), endl=False)
        self._logger.indent_level += 1

        reference.classify(entry.ontology(reference.preferred_syntax).path,
                           output_file=reference_out,
                           timeout=TestConfig.TIMEOUT)
        self._logger.log('done', color=echo.Color.GREEN)

        for reasoner in reasoners:
            self._logger.log('{}: '.format(reasoner.name), endl=False)

            try:
                reasoner.classify(entry.ontology(reasoner.preferred_syntax).path,
                                  output_file=reasoner_out,
                                  timeout=TestConfig.TIMEOUT)
            except TimeoutExpired:
                result = 'timeout'
                color = echo.Color.RED
            except Exception:
                result = 'error'
                color = echo.Color.RED
            else:
                if filecmp.cmp(reasoner_out, reference_out, shallow=False):
                    result = 'same'
                    color = echo.Color.GREEN
                else:
                    result = 'different'
                    color = echo.Color.RED

            self._logger.log(result, color=color)
            csv_row.append(result)

        self._logger.indent_level -= 1
        self._csv_writer.write_row(csv_row)


class ClassificationPerformanceTest(OntologyReasoningPerformanceTest):
    """Classification performance test."""

    @property
    def task(self) -> str:
        return ReasoningTask.CLASSIFICATION

    def run_reasoner(self, reasoner: Reasoner, ontology: Ontology) -> ReasoningStats:
        return reasoner.classify(ontology.path,
                                 timeout=TestConfig.TIMEOUT,
                                 mode=TestMode.PERFORMANCE)


class ClassificationEnergyTest(OntologyReasoningEnergyTest):
    """Classification energy test."""

    @property
    def task(self) -> str:
        return ReasoningTask.CLASSIFICATION

    def run_reasoner(self, reasoner: Reasoner, ontology: Ontology) -> ReasoningStats:
        return reasoner.classify(ontology.path,
                                 timeout=TestConfig.TIMEOUT,
                                 mode=TestMode.ENERGY)
