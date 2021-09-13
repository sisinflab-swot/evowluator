from __future__ import annotations

from abc import ABC, abstractmethod
from subprocess import TimeoutExpired
from typing import Dict, List, Set

from pyutils.io import echo, fileutils
from pyutils.io.echo import Color
from .base import Evaluator
from .mode import EvaluationMode
from .. import config
from ..data.dataset import Dataset
from ..data.ontology import Syntax
from ..reasoner.base import Reasoner, ReasoningTask
from ..reasoner.results import Results


class ReasoningEvaluator(Evaluator, ABC):

    @property
    @abstractmethod
    def mode(self) -> EvaluationMode:
        pass

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        pass

    @abstractmethod
    def log_results(self, results: Results) -> None:
        pass

    @abstractmethod
    def extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        pass

    # Overrides

    @property
    def name(self) -> str:
        return f'{self.task} {self.mode}'

    def __init__(self,
                 task: ReasoningTask,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        super().__init__(dataset=dataset, reasoners=reasoners, syntax=syntax)
        self.task = task

        if not reasoners:
            self._reasoners = self._loader.reasoners_supporting_task(task)

    def setup(self):
        csv_header = ['Ontology']

        if self.task.requires_additional_inputs:
            csv_header.append('Input')

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)

    def run(self, entry: Dataset.Entry) -> None:
        if self.task.requires_additional_inputs and entry.inputs_count_for_task(self.task) == 0:
            self._logger.log('No additional input files.\n', color=echo.Color.YELLOW)
            return

        fail = set()
        iterations = config.Evaluation.ITERATIONS

        if self.mode == EvaluationMode.CORRECTNESS or iterations == 1:
            self._iterate(entry, fail)
            return

        for iteration in range(iterations):
            self._logger.log(f'Run {iteration + 1}:', color=echo.Color.YELLOW)
            self._logger.indent_level += 1
            self._iterate(entry, fail)
            self._logger.indent_level -= 1

    def _iterate(self, entry: Dataset.Entry, fail: Set[str] | None = None) -> None:
        self.clear_temp()

        if not self.task.requires_additional_inputs:
            self._csv_writer.write_row(self._run_reasoners([entry], fail))
            return

        for input_entry in entry.inputs_for_task(self.task):
            self._logger.log('Input: ', color=echo.Color.YELLOW, endl=False)
            self._logger.log(input_entry.name)
            self._logger.indent_level += 1
            self._csv_writer.write_row(self._run_reasoners([entry, input_entry], fail))
            self._logger.indent_level -= 1

    def _run_reasoners(self, entries: List[Dataset.Entry], fail: Set[str]) -> List:
        results = {}

        for reasoner in self._usable_reasoners():
            self._logger.log(f'{reasoner.name}: ', endl=False)

            # Skip already failed or timed out.
            if reasoner.name in fail:
                results[reasoner] = 'skip'
                self._logger.log('skip', color=echo.Color.YELLOW)
                continue

            syntax = self._syntax_for_reasoner(reasoner)
            inputs = [e.ontology(syntax).path for e in entries]

            if self.mode == EvaluationMode.CORRECTNESS:
                output = self._output_path_for_reasoner(reasoner)
            else:
                output = None

            try:
                res = self.task.run(reasoner, inputs, output=output,
                                    mode=self.mode, timeout=config.Evaluation.TIMEOUT)
                self.log_results(res)
                results[reasoner] = res
            except Exception as e:
                if config.DEBUG:
                    raise e
                fail_reason = 'timeout' if isinstance(e, TimeoutExpired) else 'error'
                self._logger.log(fail_reason, color=echo.Color.RED)
                results[reasoner] = fail_reason
                fail.add(reasoner.name)

        return [e.name for e in entries] + self.extract_results(results)


class ReasoningCorrectnessEvaluator(ReasoningEvaluator):

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.CORRECTNESS

    @property
    def result_fields(self) -> List[str]:
        return ['match']

    def log_results(self, results: Results) -> None:
        self._logger.log('done')

    def extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        # Ensure reference output is valid
        ref_reasoner, ref_results = next(iter(results.items()))
        del results[ref_reasoner]

        if not isinstance(ref_results, Results):
            return [ref_results] + ['unknown'] * len(results)

        # Populate correctness results
        csv_row = ['reference']
        ok, wrong = [], []

        for reasoner, res in results.items():
            if isinstance(res, Results):
                res = 'ok' if res.output_matches(ref_results) else 'incorrect'

            if res == 'ok':
                ok.append(reasoner.name)
            else:
                wrong.append(reasoner.name)

            csv_row.append(res)

        if ok:
            self._logger.log('Correct: ', color=Color.GREEN, endl=False)
            self._logger.log(', '.join(ok), endl=False)

        if wrong:
            if ok:
                self._logger.log(' | ', endl=False)
            self._logger.log('Incorrect: ', color=Color.RED, endl=False)
            self._logger.log(', '.join(wrong), endl=False)

        self._logger.log('')

        return csv_row


class ReasoningPerformanceEvaluator(ReasoningEvaluator):

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.PERFORMANCE

    @property
    def result_fields(self) -> List[str]:
        return ['parsing', 'reasoning', 'memory']

    def log_results(self, results: Results) -> None:
        if not results.has_performance_stats:
            raise ValueError('Missing performance stats.')

        self._logger.log('{:.0f} ms'.format(results.total_ms))

        self._logger.indent_level += 1
        self._logger.log('Parsing: {:.0f} ms'.format(results.parsing_ms))
        self._logger.log('Reasoning: {:.0f} ms'.format(results.reasoning_ms))
        self._logger.log('Memory: {}'.format(fileutils.human_readable_bytes(results.max_memory)))
        self._logger.indent_level -= 1

    def extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        csv_row = []

        for res in results.values():
            if isinstance(res, str):
                csv_row.extend([res] * len(self.result_fields))
            else:
                csv_row.extend([res.parsing_ms, res.reasoning_ms, res.max_memory])

        return csv_row


class ReasoningEnergyEvaluator(ReasoningEvaluator):

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.ENERGY

    @property
    def result_fields(self) -> List[str]:
        return ['energy']

    def log_results(self, results: Results) -> None:
        if not results.has_energy_stats:
            raise ValueError('Missing energy stats.')
        self._logger.log(f'{results.energy_score:.2f}')

    def extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        csv_row = []

        for res in results.values():
            if isinstance(res, str):
                csv_row.append(res)
            else:
                csv_row.append(res.energy_score)

        return csv_row

    def __init__(self,
                 task: ReasoningTask, probe: str,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None):
        if not probe:
            raise ValueError('No probe specified.')

        super().__init__(task, dataset=dataset, reasoners=reasoners, syntax=syntax)
        self.__configure_reasoners(probe)

    # Private

    def __configure_reasoners(self, probe_name: str) -> None:
        probe = self._loader.probe_with_name(probe_name)

        for reasoner in self._reasoners:
            reasoner.energy_probe = probe
