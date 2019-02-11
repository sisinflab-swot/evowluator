from subprocess import TimeoutExpired

from evowluator.config import Test as TestConfig
from evowluator.pyutils import echo
from evowluator.reasoner.base import ReasoningTask
from .base import ReasoningTest
from .test_mode import TestMode
from .performance import OntologyReasoningEnergyTest, OntologyReasoningPerformanceTest


class ConsistencyCorrectnessTest(ReasoningTest):
    """Consistency correctness test."""

    @property
    def task(self) -> str:
        return ReasoningTask.CONSISTENCY

    @property
    def mode(self) -> str:
        return TestMode.CORRECTNESS

    def setup(self):
        csv_header = ['Ontology']
        csv_header.extend([r.name for r in self._usable_reasoners()])
        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        csv_row = [entry.name]

        # Check consistency
        for reasoner in self._usable_reasoners():
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


class ConsistencyPerformanceTest(OntologyReasoningPerformanceTest):
    """Consistency performance test."""

    @property
    def task(self) -> str:
        return ReasoningTask.CONSISTENCY

    def run_reasoner(self, reasoner, ontology):
        return reasoner.consistency(ontology.path,
                                    timeout=TestConfig.TIMEOUT,
                                    mode=self.mode)


class ConsistencyEnergyTest(OntologyReasoningEnergyTest):
    """Consistency energy test."""

    @property
    def task(self) -> str:
        return ReasoningTask.CONSISTENCY

    def run_reasoner(self, reasoner, ontology):
        return reasoner.consistency(ontology.path,
                                    timeout=TestConfig.TIMEOUT,
                                    mode=TestMode.ENERGY).stats
