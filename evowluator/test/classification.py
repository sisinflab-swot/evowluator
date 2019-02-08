import filecmp
import os
from subprocess import TimeoutExpired

from evowluator.config import Test as TestConfig
from evowluator.pyutils import echo, fileutils
from evowluator.reasoner.base import ReasoningTask
from .base import StandardPerformanceTest, Test
from .enum import TestMode, TestName


class ClassificationCorrectnessTest(Test):
    """Classification correctness test."""

    @property
    def name(self):
        return TestName.Classification.CORRECTNESS

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CLASSIFICATION)

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
                           timeout=TestConfig.CLASSIFICATION_TIMEOUT)
        self._logger.log('done', color=echo.Color.GREEN)

        for reasoner in reasoners:
            self._logger.log('{}: '.format(reasoner.name), endl=False)

            try:
                reasoner.classify(entry.ontology(reasoner.preferred_syntax).path,
                                  output_file=reasoner_out,
                                  timeout=TestConfig.CLASSIFICATION_TIMEOUT)
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


class ClassificationTimeTest(StandardPerformanceTest):
    """Classification turnaround time test."""

    @property
    def name(self):
        return TestName.Classification.TIME

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CLASSIFICATION)

    @property
    def result_fields(self):
        return ['parsing', 'classification']

    def run_reasoner(self, reasoner, ontology):

        stats = reasoner.classify(ontology.path,
                                  timeout=TestConfig.CLASSIFICATION_TIMEOUT,
                                  mode=TestMode.TIME)

        self._logger.log('{:.0f} ms'.format(stats.total_ms))

        self._logger.indent_level += 1
        self._logger.log('Parsing {:.0f} ms'.format(stats.parsing_ms))
        self._logger.log('Classification {:.0f} ms'.format(stats.reasoning_ms))
        self._logger.indent_level -= 1

        return [stats.parsing_ms, stats.reasoning_ms]


class ClassificationMemoryTest(StandardPerformanceTest):
    """Classification memory test."""

    @property
    def name(self):
        return TestName.Classification.MEMORY

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CLASSIFICATION)

    @property
    def result_fields(self):
        return ['memory']

    def run_reasoner(self, reasoner, ontology):

        stats = reasoner.classify(ontology.path,
                                  timeout=TestConfig.CLASSIFICATION_TIMEOUT,
                                  mode=TestMode.MEMORY)

        self._logger.log(fileutils.human_readable_bytes(stats.max_memory))

        return [stats.max_memory]


class ClassificationEnergyTest(StandardPerformanceTest):
    """Classification energy test."""

    @property
    def name(self):
        return TestName.Classification.ENERGY

    @property
    def default_reasoners(self):
        return self._loader.reasoners_supporting_task(ReasoningTask.CLASSIFICATION)

    @property
    def result_fields(self):
        return ['energy']

    def run_reasoner(self, reasoner, ontology):

        stats = reasoner.classify(ontology.path,
                                  timeout=TestConfig.CLASSIFICATION_TIMEOUT,
                                  mode=TestMode.ENERGY)

        self._logger.log('{:.2f}'.format(stats.energy_score))

        return [stats.energy_score]
