from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from subprocess import TimeoutExpired
from threading import Lock
from typing import Dict, Iterable, List, Set

from pyutils import exc
from pyutils.io import echo, fileutils
from pyutils.io.echo import Color
from pyutils.proc.bench import Benchmark, EnergyProfiler
from pyutils.proc.task import Task
from .base import Evaluator
from .mode import EvaluationMode
from .. import config
from ..config import Evaluation, Paths
from ..data.dataset import DatasetEntry, Syntax
from ..reasoner.base import Reasoner, ReasoningTask, RemoteReasoner
from ..reasoner.results import Field, Results
from ..visualization.correctness import CorrectnessStrategy, Status


class ReasoningEvaluator(Evaluator, ABC):

    @abstractmethod
    def run_reasoners(self, entries: List[DatasetEntry]) -> List:
        pass

    @property
    def name(self) -> str:
        return f'{self.task} {Evaluation.MODE}'

    def __init__(self,
                 task: ReasoningTask,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        super().__init__(dataset=dataset, reasoners=reasoners, syntax=syntax)
        self.task = task

        if Evaluation.MODE == EvaluationMode.PERFORMANCE:
            self.output_fields = task.performance_fields
        else:
            self.output_fields = [Field.OUTPUT]

        if not reasoners:
            self._reasoners = Reasoner.supporting_task(task)

    def setup(self):
        csv_header = ['ontology']

        if self.task.requires_additional_inputs:
            csv_header.append('input')

        for reasoner in self._usable_reasoners():
            for field in self.output_fields:
                csv_header.append(f'{reasoner.name}: {field}')

        self._csv_writer.write_row(csv_header)

    def run(self, entry: DatasetEntry) -> None:
        if self.task.requires_additional_inputs and entry.inputs_count_for_task(self.task) == 0:
            self._logger.log('No additional input files.', color=echo.Color.YELLOW)
            return

        csv_rows = []
        iterations = config.Evaluation.ITERATIONS

        if iterations == 1:
            csv_rows.extend(self._iterate(entry))
        else:
            for iteration in range(iterations):
                self._logger.log(f'Run {iteration + 1}:', color=echo.Color.YELLOW)
                self._logger.indent_level += 1
                csv_rows.extend(self._iterate(entry))
                self._logger.indent_level -= 1

        for row in csv_rows:
            self._csv_writer.write_row(row)

    def run_reasoner(self, reasoner: Reasoner, inputs: str | List[str],
                     output: str | None = None) -> Results:
        if not isinstance(inputs, list):
            inputs = [inputs]

        for i in inputs:
            exc.raise_if_not_found(i, file_type=exc.FileType.FILE)

        if output:
            fileutils.remove(output)

        # Run reasoner

        reasoner.pre_run(self.task, inputs, output)
        task = Task(Paths.absolute(reasoner.path), args=reasoner.args(self.task, inputs, output))

        if Evaluation.MODE == EvaluationMode.PERFORMANCE:
            if not isinstance(reasoner, RemoteReasoner):
                task = Benchmark(task)

            energy_probe = Evaluation.ENERGY_PROBE

            if energy_probe:
                interval = Evaluation.ENERGY_POLLING_INTERVALS.get(energy_probe.name, 1000)
                task = EnergyProfiler(task, energy_probe, interval=interval)

        task.run(timeout=Evaluation.TIMEOUT if Evaluation.TIMEOUT else None).raise_if_failed()
        results = self.task.process_results(reasoner.parse_results(self.task, task, output), task)
        reasoner.post_run(self.task, inputs, output)

        return results

    def _iterate(self, entry: DatasetEntry) -> List[List]:
        self.clear_temp()

        csv_rows = []

        if self.task.requires_additional_inputs:
            for input_entry in entry.inputs_for_task(self.task):
                self._logger.log('Input: ', color=echo.Color.YELLOW, endl=False)
                self._logger.log(input_entry.name)
                self._logger.indent_level += 1
                csv_rows.append(self.run_reasoners([entry, input_entry]))
                self._logger.indent_level -= 1
        else:
            csv_rows.append(self.run_reasoners([entry]))

        return csv_rows


class ReasoningCorrectnessEvaluator(ReasoningEvaluator):

    def __init__(self,
                 task: ReasoningTask,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        super().__init__(task, dataset=dataset, reasoners=reasoners, syntax=syntax)
        self._strategy = CorrectnessStrategy.default()
        self._lock = Lock()

    def set_strategy(self, strategy: str | None) -> None:
        self._strategy = CorrectnessStrategy.with_name(strategy, [r.name for r in self._reasoners])

    def run_reasoners(self, entries: List[DatasetEntry]) -> List:
        results = {}

        self._logger.log('Done: ', color=Color.YELLOW, endl=False)

        with ThreadPoolExecutor() as pool:
            for reasoner in self._usable_reasoners():
                syntax = self._syntax_for_reasoner(reasoner)
                inputs = [e.ontology(syntax).path for e in entries]
                output = self._output_path_for_reasoner(reasoner)
                pool.submit(self._run_reasoner_correctness, reasoner, inputs, output, results)

        results = {r: results[r] for r in self._usable_reasoners()}

        self._logger.log('')
        self._log_results(results)

        return [e.name for e in entries] + list(results.values())

    def _run_reasoner_correctness(self, reasoner: Reasoner, inputs: List[str], output: str,
                                  results: Dict) -> None:
        try:
            res = self.run_reasoner(reasoner, inputs, output).output.hash()
        except Exception as e:
            if config.DEBUG:
                raise e
            res = Status.TIMEOUT if isinstance(e, TimeoutExpired) else Status.ERROR

        with self._lock:
            results[reasoner] = res
            self._logger.log(('' if len(results) == 1 else ', ') + reasoner.name, endl=False)

    def _log_results(self, results: Dict[Reasoner]) -> None:
        if not self._strategy:
            return

        ok, wrong = [], []

        for r, v in self._strategy.evaluate_dict(results).items():
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


class ReasoningPerformanceEvaluator(ReasoningEvaluator):

    def __init__(self,
                 task: ReasoningTask,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        super().__init__(task, dataset=dataset, reasoners=reasoners, syntax=syntax)
        self._skip: Dict[str, Set[str]] = {r.name: set() for r in self._reasoners}

    def skip_ontologies(self, reasoner: str, ontologies: Iterable[str]) -> None:
        try:
            self._skip[reasoner].update(ontologies)
        except KeyError:
            pass

    def run_reasoners(self, entries: List[DatasetEntry]) -> List:
        results = {}

        for reasoner in self._usable_reasoners():
            self._logger.log(f'{reasoner.name}: ', color=Color.YELLOW, endl=False)

            # Skip already failed or timed out.
            root_ontology = entries[0].name

            if root_ontology in self._skip[reasoner.name]:
                results[reasoner] = 'skip'
                self._logger.log('skip', color=echo.Color.YELLOW)
                continue

            syntax = self._syntax_for_reasoner(reasoner)
            inputs = [e.ontology(syntax).path for e in entries]

            try:
                r = self.run_reasoner(reasoner, inputs)
                self._log_results(r)
                results[reasoner] = r
            except Exception as e:
                if config.DEBUG:
                    raise e
                fail_reason = Status.TIMEOUT if isinstance(e, TimeoutExpired) else Status.ERROR
                self._logger.log(fail_reason, color=echo.Color.RED)
                results[reasoner] = fail_reason
                self._skip[reasoner.name].add(root_ontology)

        return [e.name for e in entries] + self._extract_results(results)

    def _log_results(self, results: Results) -> None:
        if len(self.output_fields) == 1:
            self._logger.log(results.get_readable(self.output_fields[0]))
        else:
            if Field.PARSING in self.output_fields and Field.REASONING in self.output_fields:
                self._logger.log(f'{results.total_time:.2f} ms')
            else:
                self._logger.log('')

            self._logger.indent_level += 1

            for field in self.output_fields:
                self._logger.log(f'{field.value.capitalize()}: {results.get_readable(field)}')

            self._logger.indent_level -= 1

    def _extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        csv_row = []

        for res in results.values():
            if isinstance(res, str):
                csv_row.extend([res] * len(self.output_fields))
            else:
                csv_row.extend(res.get(f) for f in self.output_fields)

        return csv_row
