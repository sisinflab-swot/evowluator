from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from os import path
from subprocess import TimeoutExpired
from sys import stdout
from threading import Lock
from typing import Dict, Iterable, List, Set

from pyutils import exc
from pyutils.io import echo, file
from pyutils.io.pretty_printer import PrettyPrinter
from pyutils.proc.bench import Benchmark
from pyutils.proc.energy import EnergyProfiler
from pyutils.proc.task import Task
from pyutils.types.unit import MemoryUnit
from .mode import EvaluationMode
from .. import config
from ..config import ConfigKey
from ..config import Evaluation, Paths
from ..data import json
from ..data.csv import CSVWriter
from ..data.dataset import Dataset, SortBy
from ..data.dataset import DatasetEntry, Syntax
from ..data.info import DatasetInfo
from ..reasoner.base import Reasoner, ReasoningTask, RemoteReasoner
from ..reasoner.results import Results
from ..visualization.correctness import CorrectnessStrategy, Status


class Evaluator(ABC):
    """Abstract evaluator class."""

    @property
    def _name(self) -> str:
        return f'{self._task} {Evaluation.MODE}'

    @cached_property
    def _work_dir(self) -> str:
        return Paths.new_results_dir(self._name)

    @cached_property
    def _temp_dir(self) -> str:
        new_dir = path.join(self._work_dir, 'temp')
        file.create_dir(new_dir)
        return new_dir

    @property
    def _requires_inputs(self) -> bool:
        return self._task.requires_additional_inputs

    def __init__(self,
                 task: ReasoningTask,
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        self._task = task
        self._data = Dataset.with_name(dataset) if dataset else Dataset.first()

        if syntax and syntax not in self._data.syntaxes:
            msg = f'"{syntax}" syntax not available for "{self._data.name}" dataset.'
            raise ValueError(msg)

        self._syntax = syntax
        self._log: PrettyPrinter | None = None
        self._csv: CSVWriter | None = None

        if reasoners:
            try:
                self._reasoners = [Reasoner.with_name(n) for n in reasoners]
            except KeyError as e:
                exc.re_raise_new_message(e, 'No such reasoner: ' + str(e))
        else:
            self._reasoners = Reasoner.supporting_task(task)

        if Evaluation.MODE == EvaluationMode.PERFORMANCE:
            self._fields = task.performance_fields + [p.name for p in Evaluation.ENERGY_PROBES]
        else:
            self._fields = ['output']

    def start(self, sort_by: SortBy = SortBy.NAME, resume_after: str | None = None) -> None:
        """Starts the evaluation."""
        self._log = PrettyPrinter(stdout, path.join(self._work_dir, config.Paths.LOG_FILE_NAME))
        self._csv = CSVWriter(path.join(self._work_dir, config.Paths.RESULTS_FILE_NAME))

        try:
            self._setup_reasoners()
            self._save_config()

            with self._log, self._csv:
                self._log.clear()
                self._log_config()
                self._setup()
                self._start(sort_by=sort_by, resume_after=resume_after)
                self._log.spacer(2, flush=True)
        finally:
            file.chmod(self._work_dir, 0o644, recursive=True, dir_mode=0o755)
            self._teardown_reasoners()
            echo.success('Evaluation results: ', endl=False)
            echo.info(self._work_dir)

    @abstractmethod
    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        pass

    def _clear_temp(self) -> None:
        file.remove_dir_contents(self._temp_dir)

    def _setup(self) -> None:
        csv_header = ['ontology']

        if self._requires_inputs:
            csv_header.append('input')

        for reasoner in self._usable_reasoners():
            for field in self._fields:
                csv_header.append(f'{reasoner.name}: {field}')

        self._csv.write_row(csv_header)

    def _start(self, sort_by: SortBy = SortBy.NAME, resume_after: str | None = None) -> None:
        used_syntaxes = self._syntaxes()

        dataset_count, dataset_size = self._data.cumulative_stats(syntaxes=used_syntaxes,
                                                                  sort_by=sort_by,
                                                                  resume_after=resume_after)
        dataset_size = MemoryUnit.B(dataset_size).readable()
        tot_size = 0

        for idx, entry in enumerate(self._data.get_entries(sort_by=sort_by,
                                                           resume_after=resume_after)):
            sizes = list(sorted((o.syntax, o.size) for o in entry.ontologies()))
            size_str = ' | '.join(f'{syntax}: {MemoryUnit.B(size).readable()}'
                                  for syntax, size in sizes)
            tot_size += sum(size for syntax, size in sizes if syntax in used_syntaxes)

            self._log.spacer(2)
            self._log.green(entry.name)
            self._log.yellow('Sizes: ', endl=False)
            self._log(size_str, endl=False)
            self._log.yellow(' Progress: ', endl=False)
            self._log(f'{idx + 1}/{dataset_count} '
                      f'({MemoryUnit.B(tot_size).to_value(dataset_size.unit):.1f}/{dataset_size})')

            with self._log.indent:
                try:
                    self._run(entry)
                except Exception as e:
                    if config.DEBUG:
                        raise e
                    else:
                        self._log.error(str(e))

    def _output_path_for_reasoner(self, reasoner: Reasoner) -> str:
        return os.path.join(self._temp_dir, reasoner.name.lower().replace(' ', '_'))

    def _syntaxes_for_reasoner(self, reasoner: Reasoner) -> List[Syntax]:
        available = self._data.syntaxes
        return [s for s in reasoner.supported_syntaxes if s in available]

    def _syntax_for_reasoner(self, reasoner: Reasoner) -> Syntax:
        supported = reasoner.supported_syntaxes

        if self._syntax in supported:
            return self._syntax

        available = self._syntaxes_for_reasoner(reasoner)

        if reasoner.preferred_syntax in available:
            return reasoner.preferred_syntax

        if available:
            return available[0]

        raise ValueError(f'No available syntax for reasoner "{reasoner.name}"')

    def _syntaxes(self) -> List[Syntax]:
        return list({self._syntax_for_reasoner(r) for r in self._usable_reasoners()})

    def _usable_reasoners(self) -> List[Reasoner]:
        return [r for r in self._reasoners if self._syntaxes_for_reasoner(r)]

    def _run(self, entry: DatasetEntry) -> None:
        if self._requires_inputs and entry.inputs_count_for_task(self._task) == 0:
            self._log.yellow('No additional input files.')
            return

        csv_rows = []
        iterations = config.Evaluation.ITERATIONS

        if iterations == 1:
            csv_rows.extend(self._iterate(entry))
        else:
            for iteration in range(iterations):
                self._log.yellow(f'Run {iteration + 1}:')
                with self._log.indent:
                    csv_rows.extend(self._iterate(entry))

        for row in csv_rows:
            self._csv.write_row(row)

    def _iterate(self, entry: DatasetEntry) -> List[List]:
        self._clear_temp()

        csv_rows = []

        if self._requires_inputs:
            for input_entry in entry.inputs_for_task(self._task):
                self._log.yellow('Input: ', endl=False)
                self._log(input_entry.name)
                with self._log.indent:
                    csv_rows.append(self._run_reasoners([entry, input_entry]))
        else:
            csv_rows.append(self._run_reasoners([entry]))

        return csv_rows

    def _run_reasoner(self, reasoner: Reasoner, inputs: str | List[str],
                      output: str | None = None) -> Results:
        if not isinstance(inputs, list):
            inputs = [inputs]

        for i in inputs:
            exc.raise_if_not_found(i, file_type=exc.FileType.FILE)

        if output:
            file.remove(output)

        # Run reasoner

        reasoner.pre_run(self._task, inputs, output)
        task = Task(Paths.absolute(reasoner.path), args=reasoner.args(self._task, inputs, output))

        if Evaluation.MODE == EvaluationMode.PERFORMANCE:
            if not isinstance(reasoner, RemoteReasoner):
                task = Benchmark(task)
            if Evaluation.ENERGY_PROBES:
                task = EnergyProfiler(task, Evaluation.ENERGY_PROBES)

        task.run(timeout=Evaluation.TIMEOUT if Evaluation.TIMEOUT else None).raise_if_failed()
        results = self._task.process_results(reasoner.parse_results(self._task, task, output), task)
        reasoner.post_run(self._task, inputs, output)

        return results

    def _setup_reasoners(self) -> None:
        for r in self._usable_reasoners():
            r.setup()

    def _teardown_reasoners(self) -> None:
        for r in self._usable_reasoners():
            r.teardown()

    def _log_config(self) -> None:
        self._log.green('Selected reasoners and serializations:')
        with self._log.indent:
            for reasoner in self._reasoners:
                self._log_syntaxes(reasoner)

        self._log.spacer(2)
        self._log.green('Selected dataset:')
        with self._log.indent:
            self._log(f'{self._data.name}: {self._data.count()} ontologies')

        self._log.spacer(2)
        self._log.green(f'Starting {self._name} evaluation...')
        self._log.spacer(2)

    def _log_syntaxes(self, reasoner: Reasoner) -> None:
        syntaxes = sorted(self._syntaxes_for_reasoner(reasoner))

        if not syntaxes:
            self._log(f'{reasoner.name}: no syntaxes')
            return

        syntax = self._syntax_for_reasoner(reasoner)
        syntaxes = [f'[{s}]' if s == syntax else s for s in syntaxes]
        self._log(f'{reasoner.name}: {" ".join(syntaxes)}')

    def _config(self) -> Dict:
        return {
            ConfigKey.TASK: self._task.name,
            ConfigKey.MODE: Evaluation.MODE,
            ConfigKey.FIELDS: self._fields,
            ConfigKey.TIMEOUT: Evaluation.TIMEOUT,
            ConfigKey.ITERATIONS: Evaluation.ITERATIONS,
            ConfigKey.DATASET: DatasetInfo.with_dataset(self._data).to_dict(self._syntaxes()),
            ConfigKey.REASONERS: [{
                ConfigKey.NAME: r.name,
                ConfigKey.SYNTAX: self._syntax_for_reasoner(r)
            } for r in self._usable_reasoners()]
        }

    def _save_config(self) -> None:
        json.save(self._config(), path.join(self._work_dir, config.Paths.CONFIG_FILE_NAME))


class CorrectnessEvaluator(Evaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._strategy = CorrectnessStrategy.default()
        self._max_workers: int | None = None
        self._lock = Lock()

    def set_strategy(self, strategy: str | None) -> None:
        self._strategy = CorrectnessStrategy.with_name(strategy, [r.name for r in self._reasoners])

    def set_max_workers(self, workers: int) -> None:
        self._max_workers = workers

    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        results = {}

        self._log.yellow('Done: ', endl=False)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for reasoner in self._usable_reasoners():
                syntax = self._syntax_for_reasoner(reasoner)
                inputs = [e.ontology(syntax).path for e in entries]
                output = self._output_path_for_reasoner(reasoner)
                pool.submit(self._run_reasoner_correctness, reasoner, inputs, output, results)

        results = {r: results[r] for r in self._usable_reasoners()}
        self._log.spacer()
        self._log_results(results)

        return [e.name for e in entries] + list(results.values())

    def _run_reasoner_correctness(self, reasoner: Reasoner, inputs: List[str], output: str,
                                  results: Dict) -> None:
        try:
            res = self._run_reasoner(reasoner, inputs, output).output.hash()
        except Exception as e:
            if config.DEBUG:
                raise e
            res = Status.TIMEOUT if isinstance(e, TimeoutExpired) else Status.ERROR

        with self._lock:
            results[reasoner] = res
            self._log(('' if len(results) == 1 else ', ') + reasoner.name, endl=False)

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
            self._log.green('Correct: ', endl=False)
            self._log(', '.join(ok), endl=False)

        if wrong:
            if ok:
                self._log(' | ', endl=False)
            self._log.red('Incorrect: ', endl=False)
            self._log(', '.join(wrong), endl=False)

        self._log.spacer()


class PerformanceEvaluator(Evaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._skip: Dict[str, Set[str]] = {r.name: set() for r in self._reasoners}

    def skip_ontologies(self, reasoner: str, ontologies: Iterable[str]) -> None:
        try:
            self._skip[reasoner].update(ontologies)
        except KeyError:
            pass

    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        results = {}

        for reasoner in self._usable_reasoners():
            self._log.yellow(f'{reasoner.name}: ', endl=False)

            # Skip already failed or timed out.
            root_ontology = entries[0].name

            if root_ontology in self._skip[reasoner.name]:
                results[reasoner] = 'skip'
                self._log.yellow('skip')
                continue

            syntax = self._syntax_for_reasoner(reasoner)
            inputs = [e.ontology(syntax).path for e in entries]

            try:
                r = self._run_reasoner(reasoner, inputs)
                self._log_results(r)
                results[reasoner] = r
            except Exception as e:
                if config.DEBUG:
                    raise e

                if isinstance(e, TimeoutExpired):
                    fail_reason = Status.TIMEOUT
                    self._log.red(fail_reason)
                else:
                    fail_reason = Status.ERROR
                    self._log.red(f'{fail_reason}: {str(e)}')

                results[reasoner] = fail_reason
                self._skip[reasoner.name].add(root_ontology)

        return [e.name for e in entries] + self._extract_results(results)

    def _log_results(self, results: Results) -> None:
        if len(self._fields) == 1:
            self._log(results.get_readable(self._fields[0]))
        else:
            if len(results.times) > 1:
                self._log(results.get_readable('total_time'))

            self._log.spacer()
            with self._log.indent:
                for field in self._fields:
                    self._log(f'{field.capitalize()}: {results.get_readable(field)}')

    def _extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        csv_row = []

        for res in results.values():
            if isinstance(res, str):
                csv_row.extend([res] * len(self._fields))
            else:
                csv_row.extend(res.get(f) for f in self._fields)

        return csv_row

    def _config(self) -> Dict:
        cfg = super()._config()

        if Evaluation.ENERGY_PROBES:
            cfg[ConfigKey.ENERGY_PROBES] = [
                {ConfigKey.NAME: p.name, ConfigKey.POLLING_INTERVAL: p.interval}
                for p in Evaluation.ENERGY_PROBES
            ]

        return cfg
