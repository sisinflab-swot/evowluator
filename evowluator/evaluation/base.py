from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
from ..config import ConfigKey, Debug, Evaluation, Paths, OnError
from ..data import json
from ..data.csv import CSVWriter
from ..data.dataset import DatasetEntry, Ontology, Syntax
from ..data.info import DatasetInfo
from ..reasoner.base import Reasoner, RemoteReasoner
from ..reasoner.results import Results
from ..util.math import evaluate_expression
from ..visualization.correctness import CorrectnessStrategy, Status


class Evaluator(ABC):
    """Abstract evaluator class."""

    def __init__(self) -> None:
        self._log: PrettyPrinter | None = None
        self._csv: CSVWriter | None = None

    def start(self) -> None:
        """Starts the evaluation."""
        work_dir = Evaluation.work_dir()
        self._log = PrettyPrinter(stdout, path.join(work_dir, Paths.LOG_FILE_NAME))
        self._csv = CSVWriter(path.join(work_dir, Paths.RESULTS_FILE_NAME))
        cfg = self._generate_config()

        try:
            self._setup_reasoners()
            self._save_config(cfg)

            with self._log, self._csv:
                self._log.clear()
                self._log_config()
                self._setup()
                self._start()
                self._log.spacer(2, flush=True)
        finally:
            self._teardown_reasoners()
            self._save_config(cfg, end=True)
            echo.success('Evaluation results: ', endl=False)
            echo.info(work_dir)

    @abstractmethod
    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        pass

    @staticmethod
    def _timeout_seconds(inputs: List[Ontology]) -> float | None:
        try:
            timeout = float(Evaluation.timeout())
        except ValueError:
            # Not a float, could be an expression
            size_mb = MemoryUnit.B(sum(i.size for i in inputs)).to_value(MemoryUnit.MB)
            expression = Evaluation.timeout().replace('s', f'{size_mb:.2f}')
            timeout = evaluate_expression(expression)
        return timeout if timeout else None

    def _setup(self) -> None:
        task = Evaluation.task()
        csv_header = ['ontology']

        if task.requires_additional_inputs:
            csv_header.append('input')

        for reasoner in self._usable_reasoners():
            for field in Evaluation.fields():
                csv_header.append(f'{reasoner.name}: {field}')

        self._csv.write_row(csv_header)

    def _start(self) -> None:
        data = Evaluation.dataset()
        used_syntaxes = self._syntaxes()
        dataset_count, dataset_size = data.cumulative_stats(syntaxes=used_syntaxes)
        dataset_size = MemoryUnit.B(dataset_size).readable()
        tot_size = 0

        for idx, entry in enumerate(data.get_entries()):
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
                    self._handle(e)

    def _handle(self, e: Exception | Dict[Reasoner, Exception]) -> None:
        if Debug.ON_ERROR == OnError.IGNORE:
            return

        if Debug.ON_ERROR == OnError.ABORT:
            raise next(iter(e.values())) if isinstance(e, dict) else e

        if isinstance(e, dict):
            for k, v in e.items():
                self._log.red(f'{k.name}: {Debug.format(v)}')
        else:
            self._log.red(Debug.format(e))

        if Debug.ON_ERROR == OnError.PAUSE:
            input('Press Enter to continue...')

    def _output_path_for_reasoner(self, reasoner: Reasoner) -> str:
        return os.path.join(Evaluation.temp_dir(), reasoner.name.lower().replace(' ', '_'))

    def _syntaxes_for_reasoner(self, reasoner: Reasoner) -> List[Syntax]:
        available = Evaluation.dataset().syntaxes
        return [s for s in reasoner.supported_syntaxes if s in available]

    def _syntax_for_reasoner(self, reasoner: Reasoner) -> Syntax:
        syntax = Evaluation.syntax()
        supported = reasoner.supported_syntaxes

        if syntax in supported:
            return syntax

        available = self._syntaxes_for_reasoner(reasoner)

        if reasoner.preferred_syntax in available:
            return reasoner.preferred_syntax

        if available:
            return available[0]

        raise ValueError(f'No available syntax for reasoner "{reasoner.name}"')

    def _syntaxes(self) -> List[Syntax]:
        return list({self._syntax_for_reasoner(r) for r in self._usable_reasoners()})

    def _usable_reasoners(self) -> List[Reasoner]:
        return [r for r in Evaluation.reasoners() if self._syntaxes_for_reasoner(r)]

    def _run(self, entry: DatasetEntry) -> None:
        task = Evaluation.task()
        if task.requires_additional_inputs and entry.inputs_count_for_task(task) == 0:
            self._log.yellow('No additional input files.')
            return

        csv_rows = []
        iterations = Evaluation.iterations()

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
        task = Evaluation.task()
        file.remove_dir_contents(Evaluation.temp_dir())

        csv_rows = []

        if task.requires_additional_inputs:
            for input_entry in entry.inputs_for_task(task):
                self._log.yellow('Input: ', endl=False)
                self._log(input_entry.name)
                with self._log.indent:
                    csv_rows.append(self._run_reasoners([entry, input_entry]))
        else:
            csv_rows.append(self._run_reasoners([entry]))

        return csv_rows

    def _run_reasoner(self, reasoner: Reasoner, inputs: Ontology | List[Ontology]) -> Results:
        if not isinstance(inputs, list):
            inputs = [inputs]

        input_paths = [i.path for i in inputs]

        for i in input_paths:
            exc.raise_if_not_found(i, file_type=exc.FileType.FILE)

        output = self._output_path_for_reasoner(reasoner)
        file.remove(output)

        # Run reasoner

        reasoning_task = Evaluation.task()
        reasoner.pre_run(reasoning_task, input_paths, output)
        task = Task(Paths.absolute(reasoner.path),
                    args=reasoner.args(reasoning_task, input_paths, output))

        if Evaluation.mode() == EvaluationMode.PERFORMANCE:
            if not isinstance(reasoner, RemoteReasoner):
                task = Benchmark(task)
            if Evaluation.energy_probes():
                task = EnergyProfiler(task, Evaluation.energy_probes())

        task.run(timeout=self._timeout_seconds(inputs)).raise_if_failed()
        results = reasoner.parse_results(reasoning_task, task, output)
        results = reasoning_task.process_results(results, task)
        reasoner.post_run(reasoning_task, input_paths, output)

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
            for reasoner in Evaluation.reasoners():
                self._log_syntaxes(reasoner)

        self._log.spacer(2)
        self._log.green('Selected dataset:')
        with self._log.indent:
            dataset = Evaluation.dataset()
            self._log(f'{dataset.name}: {dataset.count()} ontologies')

        self._log.spacer(2)
        self._log.green(f'Starting {Evaluation.name()} evaluation...')
        self._log.spacer(2)

    def _log_syntaxes(self, reasoner: Reasoner) -> None:
        syntaxes = sorted(self._syntaxes_for_reasoner(reasoner))

        if not syntaxes:
            self._log(f'{reasoner.name}: no syntaxes')
            return

        syntax = self._syntax_for_reasoner(reasoner)
        syntaxes = [f'[{s}]' if s == syntax else s for s in syntaxes]
        self._log(f'{reasoner.name}: {" ".join(syntaxes)}')

    def _generate_config(self) -> Dict:
        task = Evaluation.task()
        data = Evaluation.dataset()
        return {
            ConfigKey.TASK: task.name,
            ConfigKey.MODE: Evaluation.mode(),
            ConfigKey.FIELDS: Evaluation.fields(),
            ConfigKey.TIMEOUT: Evaluation.timeout(),
            ConfigKey.ITERATIONS: Evaluation.iterations(),
            ConfigKey.SORT_BY: data.sort_by,
            ConfigKey.START: _cur_datetime_string(),
            ConfigKey.REASONERS: [{
                ConfigKey.NAME: r.name,
                ConfigKey.SYNTAX: self._syntax_for_reasoner(r)
            } for r in self._usable_reasoners()],
            ConfigKey.DATASET: DatasetInfo.with_dataset(data).to_dict(self._syntaxes()),
        }

    def _save_config(self, cfg: Dict, end=False) -> None:
        if end:
            cfg = _add_after(cfg, ConfigKey.START, ConfigKey.END, _cur_datetime_string())
        json.save(cfg, path.join(Evaluation.work_dir(), Paths.CONFIG_FILE_NAME))


class CorrectnessEvaluator(Evaluator):

    def __init__(self) -> None:
        super().__init__()
        self._strategy = CorrectnessStrategy.default()
        self._max_workers: int | None = None
        self._lock = Lock()

    def set_strategy(self, strategy: str | None) -> None:
        reasoners = [r.name for r in Evaluation.reasoners()]
        self._strategy = CorrectnessStrategy.with_name(strategy, reasoners)

    def set_max_workers(self, workers: int) -> None:
        self._max_workers = workers

    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        results = {}
        errors = {}

        self._log.yellow('Done: ', endl=False)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for reasoner in self._usable_reasoners():
                syntax = self._syntax_for_reasoner(reasoner)
                inputs = [e.ontology(syntax) for e in entries]
                pool.submit(self._run_reasoner_correctness, reasoner, inputs, results, errors)

        results = {r: results[r] for r in self._usable_reasoners()}
        self._log.spacer()
        self._log_results(results, errors)

        return [e.name for e in entries] + list(results.values())

    def _run_reasoner_correctness(self, reasoner: Reasoner, inputs: List[Ontology],
                                  results: Dict, errors: Dict) -> None:
        try:
            res = self._run_reasoner(reasoner, inputs).output.hash()
        except Exception as e:
            res = Status.TIMEOUT if isinstance(e, TimeoutExpired) else e

        with self._lock:
            if isinstance(res, Exception):
                errors[reasoner] = res
                res = Status.ERROR
            results[reasoner] = res
            self._log(('' if len(results) == 1 else ', ') + reasoner.name, endl=False)

    def _log_results(self, results: Dict[Reasoner], errors: Dict[Reasoner, Exception]) -> None:
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

        self._log.spacer(flush=True)

        if errors:
            self._handle(errors)


class PerformanceEvaluator(Evaluator):

    def _generate_config(self) -> Dict:
        cfg = super()._generate_config()

        if Evaluation.energy_probes():
            probes = [
                {ConfigKey.NAME: p.name, ConfigKey.POLLING_INTERVAL: p.interval}
                for p in Evaluation.energy_probes()
            ]
            cfg = _add_after(cfg, ConfigKey.FIELDS, ConfigKey.ENERGY_PROBES, probes)

        return cfg

    def __init__(self) -> None:
        super().__init__()
        self._skip: Dict[str, Set[str]] = {r.name: set() for r in Evaluation.reasoners()}

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
            inputs = [e.ontology(syntax) for e in entries]

            try:
                r = self._run_reasoner(reasoner, inputs)
                self._log_results(r)
                results[reasoner] = r
            except Exception as e:
                fail_reason = Status.TIMEOUT if isinstance(e, TimeoutExpired) else Status.ERROR
                self._log.red(fail_reason)

                results[reasoner] = fail_reason
                self._skip[reasoner.name].add(root_ontology)

                self._handle(e)

        return [e.name for e in entries] + self._extract_results(results)

    def _log_results(self, results: Results) -> None:
        fields = Evaluation.fields()
        if len(fields) == 1:
            self._log(results.get_readable(fields[0]))
        else:
            if len(results.times) > 1:
                self._log(results.get_readable('total_time'))

            self._log.spacer()
            with self._log.indent:
                for field in fields:
                    self._log(f'{field.capitalize()}: {results.get_readable(field)}')

    def _extract_results(self, results: Dict[Reasoner, Results | str]) -> List:
        fields = Evaluation.fields()
        csv_row = []

        for res in results.values():
            if isinstance(res, str):
                csv_row.extend([res] * len(fields))
            else:
                csv_row.extend(res.get(f) for f in fields)

        return csv_row


def _cur_datetime_string() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _add_after(dictionary: Dict, after, key, value) -> Dict:
    new_dictionary = {}
    for k, v in dictionary.items():
        new_dictionary[k] = v
        if k == after:
            new_dictionary[key] = value
    if key not in new_dictionary:
        new_dictionary[key] = value
    return new_dictionary
