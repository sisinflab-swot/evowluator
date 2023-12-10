from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from os.path import basename
from subprocess import TimeoutExpired
from sys import stdout
from threading import Lock
from typing import Dict, List, Set

from pyutils import exc
from pyutils.io import echo, file
from pyutils.io.pretty_printer import PrettyPrinter
from pyutils.proc.bench import Benchmark
from pyutils.proc.energy import EnergyProfiler
from pyutils.proc.task import Task
from pyutils.types.unit import MemoryUnit
from .mode import EvaluationMode
from ..config.debug import Debug, OnError
from ..config.evaluation import Evaluation, Paths
from ..data.csv import CSVReader, CSVWriter
from ..data.dataset import DatasetEntry, Ontology
from ..reasoner.base import Reasoner, RemoteReasoner
from ..reasoner.results import Results
from ..util.process import incorrect_ontologies
from ..visualization.correctness import Status


class Evaluator(ABC):
    """Abstract evaluator class."""

    def __init__(self) -> None:
        self._log: PrettyPrinter | None = None
        self._csv: CSVWriter | None = None
        self._resumed = False

    def resume(self, cfg: Dict) -> None:
        """Resumes an evaluation."""
        resumed = self._resume()
        if Evaluation.dataset().count() == 0:
            echo.error(f'Evaluation \"{basename(Evaluation.work_dir())}\" is already complete.')
            return
        self.start(cfg, resumed=resumed)

    def start(self, cfg: Dict, resumed: bool = False) -> None:
        """Starts the evaluation."""
        work_dir = Evaluation.work_dir()
        self._log = PrettyPrinter(stdout, Evaluation.log_file())
        self._csv = CSVWriter(Evaluation.results_file())

        try:
            self._setup_reasoners()
            Evaluation.save_config(cfg)

            with self._log, self._csv:
                if resumed:
                    self._log.green(f'Resuming \"{basename(work_dir)}\"')
                else:
                    self._write_header()
                self._log_config()
                self._start()
                self._log.spacer(2, flush=True)
        finally:
            self._teardown_reasoners()
            Evaluation.save_config(cfg, end=True)
            echo.success('Evaluation results: ', endl=False)
            echo.info(work_dir)

    @abstractmethod
    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        pass

    def _resume(self) -> bool:
        try:
            with CSVReader(Evaluation.results_file()) as reader:
                for row in reader:
                    pass
                if not row:
                    return False
                last_onto = row[0]
            dataset = Evaluation.dataset()
            if not dataset.get_entry(last_onto):
                return False
            dataset.start_after = last_onto
            return True
        except FileNotFoundError:
            return False

    def _write_header(self) -> None:
        task = Evaluation.task()
        csv_header = ['ontology']

        if task.requires_additional_inputs:
            csv_header.append('input')

        for reasoner in Evaluation.usable_reasoners():
            for field in Evaluation.fields():
                csv_header.append(f'{reasoner.name}: {field}')

        self._csv.write_row(csv_header)

    def _start(self) -> None:
        data = Evaluation.dataset()
        used_syntaxes = Evaluation.used_syntaxes()
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

        output = Evaluation.output_path_for_reasoner(reasoner)
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

        task.run(timeout=Evaluation.timeout_seconds(inputs)).raise_if_failed()
        results = reasoner.parse_results(reasoning_task, task, output)
        results = reasoning_task.process_results(results, task)
        reasoner.post_run(reasoning_task, input_paths, output)

        return results

    def _setup_reasoners(self) -> None:
        for r in Evaluation.usable_reasoners():
            r.setup()

    def _teardown_reasoners(self) -> None:
        for r in Evaluation.usable_reasoners():
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
        syntaxes = sorted(Evaluation.syntaxes_for_reasoner(reasoner))

        if not syntaxes:
            self._log(f'{reasoner.name}: no syntaxes')
            return

        syntax = Evaluation.syntax_for_reasoner(reasoner)
        syntaxes = [f'[{s}]' if s == syntax else s for s in syntaxes]
        self._log(f'{reasoner.name}: {" ".join(syntaxes)}')


class CorrectnessEvaluator(Evaluator):

    def __init__(self) -> None:
        super().__init__()
        self._lock = Lock()

    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        results = {}
        errors = {}

        self._log.yellow('Done: ', endl=False)

        with ThreadPoolExecutor(max_workers=Evaluation.max_workers()) as pool:
            for reasoner in Evaluation.usable_reasoners():
                syntax = Evaluation.syntax_for_reasoner(reasoner)
                inputs = [e.ontology(syntax) for e in entries]
                pool.submit(self._run_reasoner_correctness, reasoner, inputs, results, errors)

        results = {r: results[r] for r in Evaluation.usable_reasoners()}
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
        strategy = Evaluation.correctness_strategy()

        if not strategy:
            return

        ok, wrong = [], []

        for r, v in strategy.evaluate_dict(results).items():
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

    def __init__(self) -> None:
        super().__init__()
        self._skip: Dict[str, Set[str]] = {r.name: set() for r in Evaluation.reasoners()}
        self._skip_ontologies()

    def _skip_ontologies(self) -> None:
        results = Evaluation.correctness_results()
        if not results:
            return
        for r, o in incorrect_ontologies(results, Evaluation.correctness_strategy()).items():
            try:
                self._skip[r].update(o)
            except KeyError:
                pass

    def _run_reasoners(self, entries: List[DatasetEntry]) -> List:
        results = {}

        for reasoner in Evaluation.usable_reasoners():
            self._log.yellow(f'{reasoner.name}: ', endl=False)

            # Skip already failed or timed out.
            root_ontology = entries[0].name

            if root_ontology in self._skip[reasoner.name]:
                results[reasoner] = 'skip'
                self._log.yellow('skip')
                continue

            syntax = Evaluation.syntax_for_reasoner(reasoner)
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
