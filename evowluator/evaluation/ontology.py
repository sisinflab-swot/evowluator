import os
from abc import ABC, abstractmethod
from subprocess import TimeoutExpired
from typing import List

from pyutils.io import echo, fileutils

from evowluator import config
from evowluator.data.dataset import Dataset
from evowluator.reasoner.results import ReasoningResults
from .base import ReasoningEnergyEvaluator, ReasoningEvaluator
from .mode import EvaluationMode


class OntologyReasoningCorrectnessEvaluator(ReasoningEvaluator):
    """Evaluates the correctness of reasoning tasks over ontologies."""

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.CORRECTNESS

    def setup(self):
        reasoners = self._usable_reasoners()
        csv_header = ['Ontology'] + [f'{r.name}: match' for r in reasoners[1:]]
        self._csv_writer.write_row(csv_header)

    def run(self, entry: Dataset.Entry) -> None:
        self.clear_temp()

        reasoners = self._usable_reasoners()
        reference = reasoners[0]
        reasoners = reasoners[1:]

        out = os.path.join(self.temp_dir, 'reasoner.txt')
        ref_out = os.path.join(self.temp_dir, 'reference.txt')

        csv_row = [entry.name]

        # Reasoning
        self._logger.log(f'{reference.name}: ', endl=False)
        self._logger.indent_level += 1

        try:
            ref_ontology = entry.ontology(self._syntax_for_reasoner(reference))
            ref_result = reference.perform_task(self.task, ref_ontology.path, output_file=ref_out,
                                                timeout=config.Evaluation.TIMEOUT, mode=self.mode)
        except Exception as e:
            if config.DEBUG:
                raise e

            self._logger.log('error', color=echo.Color.RED)
            csv_row.append(['unknown'] * len(reasoners))
        else:
            self._logger.log('done', color=echo.Color.GREEN)

            for reasoner in reasoners:
                self._logger.log(f'{reasoner.name}: ', endl=False)

                try:
                    r_ontology = entry.ontology(self._syntax_for_reasoner(reasoner))
                    r_result = reasoner.perform_task(self.task, r_ontology.path,
                                                     output_file=out,
                                                     timeout=config.Evaluation.TIMEOUT,
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
                        result = 'ok'
                        color = echo.Color.GREEN
                    else:
                        result = 'incorrect'
                        color = echo.Color.RED

                self._logger.log(result, color=color)
                csv_row.append(result)

        self._logger.indent_level -= 1
        self._csv_writer.write_row(csv_row)


class OntologyReasoningMeasurementEvaluator(ReasoningEvaluator, ABC):
    """Evaluates stats of reasoning tasks over ontologies."""

    # Override

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        """Names for the columns of the CSV results."""
        pass

    @abstractmethod
    def extract_results(self, results: ReasoningResults) -> List:
        """Extracts and logs relevant results."""
        pass

    # Overrides

    def setup(self):
        csv_header = ['Ontology']

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append(f'{reasoner.name}: {field}')

        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        fail: List[str] = []

        for iteration in range(config.Evaluation.ITERATIONS):
            self._logger.log(f'Run {iteration + 1}:', color=echo.Color.YELLOW)
            self._logger.indent_level += 1

            csv_row = [entry.name]

            for reasoner in self._usable_reasoners():
                self._logger.log(f'{reasoner.name}: ', endl=False)
                ontology = entry.ontology(self._syntax_for_reasoner(reasoner))

                # Skip already failed or timed out.
                if reasoner.name in fail:
                    csv_row.extend(['skip'] * len(self.result_fields))
                    self._logger.log('skip', color=echo.Color.YELLOW)
                    continue

                try:
                    results = reasoner.perform_task(self.task, ontology.path,
                                                    timeout=config.Evaluation.TIMEOUT,
                                                    mode=self.mode)
                    csv_row.extend(self.extract_results(results))
                except TimeoutExpired:
                    csv_row.extend(['timeout'] * len(self.result_fields))
                    self._logger.log('timeout', color=echo.Color.RED)
                    fail.append(reasoner.name)
                except Exception as e:
                    if config.DEBUG:
                        raise e
                    csv_row.extend(['error'] * len(self.result_fields))
                    self._logger.log('error', color=echo.Color.RED)
                    fail.append(reasoner.name)

            self._logger.indent_level -= 1
            self._logger.log('')
            self._csv_writer.write_row(csv_row)


class OntologyReasoningPerformanceEvaluator(OntologyReasoningMeasurementEvaluator):
    """Evaluates the performance of reasoning tasks over ontologies."""

    # Overrides

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.PERFORMANCE

    @property
    def result_fields(self) -> List[str]:
        return ['parsing', 'reasoning', 'memory']

    def extract_results(self, results: ReasoningResults) -> List:
        if not results.has_performance_stats:
            raise ValueError('Missing performance stats.')

        self._logger.log(f'{results.total_ms:.0f} ms')

        self._logger.indent_level += 1
        self._logger.log(f'Parsing: {results.parsing_ms:.0f} ms')
        self._logger.log(f'Reasoning: {results.reasoning_ms:.0f} ms')
        self._logger.log(f'Memory: {fileutils.human_readable_bytes(results.max_memory)}')
        self._logger.indent_level -= 1

        return [results.parsing_ms, results.reasoning_ms, results.max_memory]


class OntologyReasoningEnergyEvaluator(ReasoningEnergyEvaluator,
                                       OntologyReasoningMeasurementEvaluator):
    """Evaluates the energy usage of reasoning tasks over ontologies."""
    pass
