from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from random import shuffle
from subprocess import TimeoutExpired
from threading import Lock
from typing import Dict, List, Set, Union

from pyutils.io import echo, fileutils
from pyutils.io.echo import Color
from pyutils.stringutils import camel_case_split
from .base import Evaluator
from .mode import EvaluationMode
from .. import config
from ..config import Evaluation
from ..data.dataset import DatasetEntry, Syntax
from ..reasoner.base import Reasoner, ReasoningTask
from ..reasoner.results import Results


class Status:
    OK = 'y'
    INCORRECT = 'n'
    ERROR = 'error'
    TIMEOUT = 'timeout'
    UNKNOWN = 'unknown'

    NOT_OK = [INCORRECT, ERROR, TIMEOUT]


class ReasoningEvaluator(Evaluator, ABC):

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

    @cached_property
    def name(self) -> str:
        return f'{self.task} {Evaluation.MODE}'

    @property
    def should_measure_energy(self) -> bool:
        return Evaluation.ENERGY_PROBE is not None

    def __init__(self,
                 task: ReasoningTask,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        super().__init__(dataset=dataset, reasoners=reasoners, syntax=syntax)
        self._lock = Lock()
        self.task = task

        if not reasoners:
            self._reasoners = Reasoner.supporting_task(task)

    def setup(self):
        csv_header = ['Ontology']

        if self.task.requires_additional_inputs:
            csv_header.append('Input')

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)

    def run(self, entry: DatasetEntry) -> None:
        if self.task.requires_additional_inputs and entry.inputs_count_for_task(self.task) == 0:
            self._logger.log('No additional input files.\n', color=echo.Color.YELLOW)
            return

        iterations = config.Evaluation.ITERATIONS
        fail = set()

        if Evaluation.MODE == EvaluationMode.CORRECTNESS or iterations == 1:
            self._iterate(entry, fail)
            return

        for iteration in range(iterations):
            self._logger.log(f'Run {iteration + 1}:', color=echo.Color.YELLOW)
            self._logger.indent_level += 1
            self._iterate(entry, fail)
            self._logger.indent_level -= 1

    def _iterate(self, entry: DatasetEntry, fail: Set[str]) -> None:
        self.clear_temp()

        if Evaluation.MODE == EvaluationMode.CORRECTNESS:
            run_reasoners = self._run_reasoners_correctness
        else:
            run_reasoners = self._run_reasoners_performance

        if not self.task.requires_additional_inputs:
            self._csv_writer.write_row(run_reasoners([entry], fail))
            return

        for input_entry in entry.inputs_for_task(self.task):
            self._logger.log('Input: ', color=echo.Color.YELLOW, endl=False)
            self._logger.log(input_entry.name)
            self._logger.indent_level += 1
            self._csv_writer.write_row(run_reasoners([entry, input_entry], fail))
            self._logger.indent_level -= 1

    def _run_reasoners_performance(self, entries: List[DatasetEntry], fail: Set[str]) -> List:
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

            try:
                r = self.task.run(reasoner, inputs)
                self.log_results(r)
                results[reasoner] = r
            except Exception as e:
                if config.DEBUG:
                    raise e
                fail_reason = Status.TIMEOUT if isinstance(e, TimeoutExpired) else Status.ERROR
                self._logger.log(fail_reason, color=echo.Color.RED)
                results[reasoner] = fail_reason
                fail.add(reasoner.name)

        return [e.name for e in entries] + self.extract_results(results)

    # noinspection PyUnusedLocal
    def _run_reasoners_correctness(self, entries: List[DatasetEntry], fail: Set[str]) -> List:
        results = {}

        self._logger.log('Done: ', endl=False)

        with ThreadPoolExecutor() as pool:
            for reasoner in self._usable_reasoners():
                syntax = self._syntax_for_reasoner(reasoner)
                inputs = [e.ontology(syntax).path for e in entries]
                output = self._output_path_for_reasoner(reasoner)
                pool.submit(self._run_reasoner_correctness, reasoner, inputs, output, results)

        self._logger.log('')

        return [e.name for e in entries] + self.extract_results(results)

    def _run_reasoner_correctness(self, reasoner: Reasoner, inputs: List[str], output: str,
                                  results: Dict) -> None:
        try:
            res = self.task.run(reasoner, inputs, output)
        except Exception as e:
            if config.DEBUG:
                raise e
            res = Status.TIMEOUT if isinstance(e, TimeoutExpired) else Status.ERROR

        with self._lock:
            results[reasoner] = res
            self._logger.log(reasoner.name if len(results) == 1 else ', ' + reasoner.name,
                             endl=False)


class CorrectnessStrategy(ABC):

    __ALL: List[CorrectnessStrategy] = None

    @classmethod
    def all(cls) -> List[CorrectnessStrategy]:
        if cls.__ALL is None:
            cls.__ALL = [OracleStrategy(), RandomMajorityStrategy()]
        return cls.__ALL

    @classmethod
    def with_name(cls, name: str) -> CorrectnessStrategy:
        try:
            return next(s for s in cls.all() if s.name == name)
        except StopIteration:
            raise ValueError(f'No correctness strategy named "{name}"')

    @cached_property
    def name(self) -> str:
        return '_'.join(t.lower() for t in camel_case_split(type(self).__name__)[:-1])

    @abstractmethod
    def evaluate(self, results: Dict[Reasoner, Union[Results, str]]) -> Dict[Reasoner, str]:
        pass


class OracleStrategy(CorrectnessStrategy):

    def evaluate(self, results: Dict[Reasoner, Union[Results, str]]) -> Dict[Reasoner, str]:
        ref_reasoner, ref_results = next(iter(results.items()))
        del results[ref_reasoner]

        out = {}

        if not isinstance(ref_results, Results):
            out[ref_reasoner] = ref_results
            out.update({r: Status.UNKNOWN for r in results.keys()})
            return out

        out[ref_reasoner] = Status.OK

        for reasoner, res in results.items():
            if isinstance(res, Results):
                res = Status.OK if res.output_matches(ref_results) else Status.INCORRECT
            out[reasoner] = res

        return out


class RandomMajorityStrategy(CorrectnessStrategy):

    def evaluate(self, results: Dict[Reasoner, Union[Results, str]]) -> Dict[Reasoner, str]:
        out, groups = {}, {}

        for reasoner, res in results.items():
            if not isinstance(res, Results):
                out[reasoner] = res
                continue

            ohash = res.output_hash()
            if ohash in groups:
                groups[ohash].append(reasoner)
            else:
                groups[ohash] = [reasoner]

        groups = list(groups.values())
        shuffle(groups)

        correct = max((len(g), g) for g in groups)[1]
        out.update({r: Status.OK for r in correct})
        out.update({r: Status.INCORRECT for g in groups for r in g if g is not correct})

        return out


class ReasoningCorrectnessEvaluator(ReasoningEvaluator):

    @cached_property
    def result_fields(self) -> List[str]:
        return ['correct']

    def log_results(self, results: Results) -> None:
        self._logger.log('done')

    def extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        results = {r: results[r] for r in self._usable_reasoners()}
        results = self.strategy.evaluate(results)

        ok, wrong = [], []

        for r, v in results.items():
            if v == Status.OK:
                ok.append(r.name)
            elif v in Status.NOT_OK:
                wrong.append(r.name)

        if ok:
            self._logger.log('Correct: ', color=Color.GREEN, endl=False)
            self._logger.log(', '.join(ok), endl=False)

        if wrong:
            if ok:
                self._logger.log(' | ', endl=False)
            self._logger.log('Incorrect: ', color=Color.RED, endl=False)
            self._logger.log(', '.join(wrong), endl=False)

        self._logger.log('')

        return list(results.values())

    def __init__(self,
                 task: ReasoningTask, strategy: CorrectnessStrategy,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        super().__init__(task, dataset=dataset, reasoners=reasoners, syntax=syntax)
        self.strategy = strategy


class ReasoningPerformanceEvaluator(ReasoningEvaluator):

    @cached_property
    def result_fields(self) -> List[str]:
        fields = ['parsing', 'reasoning', 'memory']

        if self.should_measure_energy:
            fields.append('energy')

        return fields

    def log_results(self, results: Results) -> None:
        if not results.has_performance_stats:
            raise ValueError('Missing performance stats.')

        if self.should_measure_energy and not results.has_energy_stats:
            raise ValueError('Missing energy stats.')

        self._logger.log('{:.0f} ms'.format(results.total_ms))

        self._logger.indent_level += 1

        self._logger.log(f'Parsing: {results.parsing_ms:.0f} ms')
        self._logger.log(f'Reasoning: {results.reasoning_ms:.0f} ms')
        self._logger.log(f'Memory: {fileutils.human_readable_bytes(results.max_memory)}')

        if self.should_measure_energy:
            self._logger.log(f'Energy: {results.energy_score:.2f}')

        self._logger.indent_level -= 1

    def extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        csv_row = []

        for res in results.values():
            if isinstance(res, str):
                csv_row.extend([res] * len(self.result_fields))
            else:
                csv_row.extend((res.parsing_ms, res.reasoning_ms, res.max_memory))
                if self.should_measure_energy:
                    csv_row.append(res.energy_score)

        return csv_row
