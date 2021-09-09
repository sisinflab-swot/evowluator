import os
from abc import ABC
from subprocess import TimeoutExpired
from typing import List

from pyutils.io import echo

from evowluator import config
from evowluator.data.dataset import Dataset
from .base import (
    ReasoningEnergyEvaluator,
    ReasoningEvaluator,
    ReasoningMeasurementEvaluator,
    ReasoningPerformanceEvaluator
)
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
            ref_result = self.task.run(reference, ref_ontology.path, output=ref_out,
                                       mode=self.mode, timeout=config.Evaluation.TIMEOUT)
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
                    r_result = self.task.run(reasoner, r_ontology.path, output=out,
                                             mode=self.mode, timeout=config.Evaluation.TIMEOUT)
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


class OntologyReasoningMeasurementEvaluator(ReasoningMeasurementEvaluator, ABC):
    """Evaluates stats of reasoning tasks over ontologies."""

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
                    results = self.task.run(reasoner, ontology.path, mode=self.mode,
                                            timeout=config.Evaluation.TIMEOUT)
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


class OntologyReasoningPerformanceEvaluator(ReasoningPerformanceEvaluator,
                                            OntologyReasoningMeasurementEvaluator):
    """Evaluates the performance of reasoning tasks over ontologies."""
    pass


class OntologyReasoningEnergyEvaluator(ReasoningEnergyEvaluator,
                                       OntologyReasoningMeasurementEvaluator):
    """Evaluates the energy usage of reasoning tasks over ontologies."""
    pass
