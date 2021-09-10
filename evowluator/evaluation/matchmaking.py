from __future__ import annotations

import os
from abc import ABC
from subprocess import TimeoutExpired
from typing import List

from pyutils.io import echo

from evowluator import config
from evowluator.data.dataset import Dataset
from evowluator.data.ontology import Syntax
from evowluator.reasoner.base import ReasoningTask
from .base import (
    ReasoningEnergyEvaluator,
    ReasoningEvaluator,
    ReasoningMeasurementEvaluator,
    ReasoningPerformanceEvaluator
)
from .mode import EvaluationMode


class MatchmakingCorrectnessEvaluator(ReasoningEvaluator):
    """Evaluates the correctness of non-standard reasoning tasks."""

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.CORRECTNESS

    def __init__(self,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None):
        super().__init__(ReasoningTask.MATCHMAKING,
                         dataset=dataset, reasoners=reasoners, syntax=syntax)

    def setup(self):
        reasoners = self._usable_reasoners()
        csv_header = ['Resource', 'Request'] + [f'{r.name}: match' for r in reasoners[1:]]
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
            self._logger.log(f'{reference.name}: ', endl=False)

            csv_row = [entry.name, request.name]

            try:
                inputs = [entry.ontology(ref_syntax).path, request.ontology(ref_syntax).path]
                ref_result = self.task.run(reference, inputs, output=ref_out, mode=self.mode,
                                           timeout=config.Evaluation.TIMEOUT)
            except Exception as e:
                if config.DEBUG:
                    raise e

                self._logger.log('error', color=echo.Color.RED)
                csv_row.append(['unknown'] * len(reasoners))
            else:
                self._logger.log('done', color=echo.Color.GREEN)
                self._logger.indent_level += 1

                for reasoner in reasoners:
                    self._logger.log(f'{reasoner.name}: ', endl=False)

                    syntax = self._syntax_for_reasoner(reasoner)
                    resource_onto, request_onto = entry.ontology(syntax), request.ontology(syntax)

                    try:
                        r_result = self.task.run(reasoner, [resource_onto.path, request_onto.path],
                                                 output=r_out, mode=self.mode,
                                                 timeout=config.Evaluation.TIMEOUT)
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
                            result = 'ok'
                            color = echo.Color.GREEN
                        else:
                            result = 'incorrect'
                            color = echo.Color.RED

                    self._logger.log(result, color=color)
                    csv_row.append(result)

                self._logger.indent_level -= 1

            self._logger.indent_level -= 1
            self._logger.log('')
            self._csv_writer.write_row(csv_row)
        self._logger.log('')


class MatchmakingMeasurementEvaluator(ReasoningMeasurementEvaluator, ABC):
    """Measures stats of matchmaking tasks."""

    def __init__(self,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None):
        super().__init__(ReasoningTask.MATCHMAKING,
                         dataset=dataset, reasoners=reasoners, syntax=syntax)

    def setup(self):
        csv_header = ['Resource', 'Request']

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append(f'{reasoner.name}: {field}')

        self._csv_writer.write_row(csv_header)

    def run(self, entry: Dataset.Entry) -> None:

        if entry.request_count() == 0:
            self._logger.log('No available requests.\n', color=echo.Color.YELLOW)
            return

        for iteration in range(config.Evaluation.ITERATIONS):
            self._logger.log(f'Run {iteration + 1}:', color=echo.Color.YELLOW)
            self._logger.indent_level += 1

            for request in entry.requests():
                self._logger.log('Request: ', color=echo.Color.YELLOW, endl=False)
                self._logger.log(request.name)
                self._logger.indent_level += 1

                csv_row = [entry.name, request.name]

                for reasoner in self._usable_reasoners():
                    self._logger.log(f'{reasoner.name}: ', endl=False)

                    syntax = self._syntax_for_reasoner(reasoner)
                    resource_onto = entry.ontology(syntax)
                    request_onto = request.ontology(syntax)

                    try:
                        results = self.task.run(reasoner, [resource_onto.path, request_onto.path],
                                                mode=self.mode, timeout=config.Evaluation.TIMEOUT)
                        csv_row.extend(self.extract_results(results))
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


class MatchmakingPerformanceEvaluator(MatchmakingMeasurementEvaluator,
                                      ReasoningPerformanceEvaluator):
    """Evaluates the performance of non-standard reasoning tasks."""
    pass


class MatchmakingEnergyEvaluator(MatchmakingMeasurementEvaluator,
                                 ReasoningEnergyEvaluator):
    """Evaluates the energy usage of non-standard reasoning tasks."""

    def __init__(self,
                 probe: str,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None):
        super().__init__(ReasoningTask.MATCHMAKING, probe,
                         dataset=dataset, reasoners=reasoners, syntax=syntax)
