import re
import tempfile
import time
from abc import ABC, abstractmethod
from os import path
from typing import List, Optional

from pyutils import exc
from pyutils.decorators import cached_property
from pyutils.io import echo, fileutils
from pyutils.io.logger import Logger

from evowluator import config
from evowluator.config import ConfigKey, Paths
from evowluator.data import json
from evowluator.data.csv import CSVWriter
from evowluator.data.dataset import Dataset
from evowluator.data.ontology import Syntax
from evowluator.reasoner.base import Reasoner, ReasoningTask
from evowluator.reasoner.results import Results
from evowluator.user.loader import Loader
from .mode import EvaluationMode


class Evaluator(ABC):
    """Abstract evaluator class."""

    # Override

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this evaluator."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Called before the evaluator starts iterating on ontologies."""
        pass

    @abstractmethod
    def run(self, entry: Dataset.Entry) -> None:
        """Runs the evaluator over a single ontology."""
        pass

    # Public

    @cached_property
    def work_dir(self) -> str:
        """Work directory."""
        name = re.sub(r"[^\w\s]", '', self.name)
        name = re.sub(r"\s+", '_', name)
        prefix = time.strftime(f'{name}_%Y%m%d_%H%M%S_')
        fileutils.create_dir(Paths.RESULTS_DIR)
        return tempfile.mkdtemp(dir=Paths.RESULTS_DIR, prefix=prefix)

    @cached_property
    def temp_dir(self) -> str:
        """Directory for temporary files."""
        new_dir = path.join(self.work_dir, 'temp')
        fileutils.create_dir(new_dir)
        return new_dir

    @property
    def log_path(self) -> str:
        """Log file path."""
        return self._logger.file_path

    @property
    def csv_path(self) -> str:
        """CSV results file path."""
        return self._csv_writer.file_path

    def __init__(self,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[Syntax] = None) -> None:
        self._dataset = Dataset.with_name(dataset) if dataset else Dataset.first()

        if syntax and syntax not in self._dataset.syntaxes:
            msg = f'"{syntax}" syntax not available for "{self._dataset.name}" dataset.'
            raise ValueError(msg)

        self._loader = Loader()
        self._syntax = syntax
        self._logger: Optional[Logger] = None
        self._csv_writer: Optional[CSVWriter] = None

        if reasoners:
            try:
                self._reasoners = [self._loader.reasoner_with_name(n) for n in reasoners]
            except KeyError as e:
                exc.re_raise_new_message(e, 'No such reasoner: ' + str(e))
        else:
            self._reasoners = self._loader.reasoners

    def clear_temp(self) -> None:
        """Clears temporary files."""
        fileutils.remove_dir_contents(self.temp_dir)

    def start(self, resume_ontology: Optional[str] = None) -> None:
        """Starts the evaluation."""
        self._logger = Logger(path.join(self.work_dir, config.Paths.LOG_FILE_NAME))
        self._csv_writer = CSVWriter(path.join(self.work_dir, config.Paths.RESULTS_FILE_NAME))

        try:
            self.__setup_reasoners()
            self.__save_config()

            with self._logger, self._csv_writer:
                self._logger.clear()
                self.__log_config()
                self.setup()
                self._start(resume_ontology)
        finally:
            fileutils.chmod(self.work_dir, 0o666, recursive=True, dir_mode=0o777)
            self.__teardown_reasoners()
            echo.success('Evaluation results: ', endl=False)
            echo.info(self.work_dir)

    # Protected

    def _syntaxes_for_reasoner(self, reasoner: Reasoner) -> List[Syntax]:
        available = self._dataset.syntaxes
        return [s for s in reasoner.supported_syntaxes if s in available]

    def _syntax_for_reasoner(self, reasoner: Reasoner) -> Optional[Syntax]:
        supported = reasoner.supported_syntaxes

        if self._syntax in supported:
            return self._syntax

        available = self._syntaxes_for_reasoner(reasoner)

        if reasoner.preferred_syntax in available:
            return reasoner.preferred_syntax

        return available[0] if available else None

    def _usable_reasoners(self) -> List[Reasoner]:
        return [r for r in self._reasoners if self._syntaxes_for_reasoner(r)]

    def _start(self, resume_ontology: Optional[str] = None) -> None:
        for entry in self._dataset.get_entries(resume_after=resume_ontology):
            sizes = sorted(f'{o.syntax}: {o.readable_size}' for o in entry.ontologies())
            size_str = ' | '.join(sizes)

            self._logger.log(f'{entry.name}', color=echo.Color.YELLOW, endl=False)
            self._logger.log(f' ({size_str})')
            self._logger.indent_level += 1

            try:
                self.run(entry)
            except Exception as e:
                if config.DEBUG:
                    raise e
                else:
                    echo.error(str(e))
            finally:
                self._logger.indent_level -= 1

        self._logger.log('')

    # Private

    def __setup_reasoners(self) -> None:
        for r in self._usable_reasoners():
            r.setup()

    def __teardown_reasoners(self) -> None:
        for r in self._usable_reasoners():
            r.teardown()

    def __log_config(self) -> None:
        self._logger.log('Selected reasoners and serializations:', color=echo.Color.GREEN)
        self._logger.indent_level += 1
        for reasoner in self._reasoners:
            self.__log_syntaxes(reasoner)
        self._logger.indent_level -= 1

        self._logger.log('\nSelected dataset: ', color=echo.Color.GREEN)
        self._logger.indent_level += 1
        self._logger.log(f'{self._dataset.name}: {self._dataset.size} ontologies')
        self._logger.indent_level -= 1

        self._logger.log(f'\nStarting {self.name} evaluation...\n', color=echo.Color.GREEN)

    def __log_syntaxes(self, reasoner: Reasoner) -> None:
        syntaxes = sorted(self._syntaxes_for_reasoner(reasoner), key=lambda s: s.value)

        if not syntaxes:
            self._logger.log(f'{reasoner.name}: no syntaxes')
            return

        syntax = self._syntax_for_reasoner(reasoner)
        syntaxes = [f'[{s}]' if s == syntax else s.value for s in syntaxes]
        self._logger.log(f'{reasoner.name}: {" ".join(syntaxes)}')

    def __save_config(self) -> None:

        cfg = {
            ConfigKey.NAME: self.name,
            ConfigKey.DATASET: self._dataset.name,
            ConfigKey.REASONERS: [{
                ConfigKey.NAME: r.name,
                ConfigKey.SYNTAX: self._syntax_for_reasoner(r).value
            } for r in self._usable_reasoners()]
        }

        json.save(cfg, path.join(self.work_dir, config.Paths.CONFIG_FILE_NAME))


class ReasoningEvaluator(Evaluator):
    """Abstract reasoning evaluator class."""

    @property
    @abstractmethod
    def mode(self) -> EvaluationMode:
        """Evaluation mode."""
        pass

    # Overrides

    @property
    def name(self) -> str:
        return f'{self.task} {self.mode}'

    def __init__(self,
                 task: ReasoningTask,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[Syntax] = None) -> None:
        super().__init__(dataset=dataset, reasoners=reasoners, syntax=syntax)
        self.task = task

        if not reasoners:
            self._reasoners = self._loader.reasoners_supporting_task(task)


class ReasoningMeasurementEvaluator(ReasoningEvaluator, ABC):
    """Evaluates stats of reasoning tasks."""

    # Override

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        """Names for the columns of the CSV results."""
        pass

    @abstractmethod
    def extract_results(self, results: Results) -> List:
        """Extracts and logs relevant results."""
        pass

    # Overrides

    def setup(self):
        csv_header = ['Ontology']

        for reasoner in self._usable_reasoners():
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)


class ReasoningPerformanceEvaluator(ReasoningMeasurementEvaluator, ABC):
    """Evaluates the performance of reasoning tasks."""

    # Overrides

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.PERFORMANCE

    @property
    def result_fields(self) -> List[str]:
        return ['parsing', 'reasoning', 'memory']

    def extract_results(self, results: Results) -> List:
        if not results.has_performance_stats:
            raise ValueError('Missing performance stats.')

        self._logger.log('{:.0f} ms'.format(results.total_ms))

        self._logger.indent_level += 1
        self._logger.log('Parsing: {:.0f} ms'.format(results.parsing_ms))
        self._logger.log('Reasoning: {:.0f} ms'.format(results.reasoning_ms))
        self._logger.log('Memory: {}'.format(fileutils.human_readable_bytes(results.max_memory)))
        self._logger.indent_level -= 1

        return [results.parsing_ms, results.reasoning_ms, results.max_memory]


class ReasoningEnergyEvaluator(ReasoningMeasurementEvaluator, ABC):
    """Abstract reasoning energy evaluator class."""

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.ENERGY

    @property
    def result_fields(self) -> List[str]:
        return ['energy']

    def __init__(self,
                 task: ReasoningTask, probe: str,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[Syntax] = None):
        if not probe:
            raise ValueError('No probe specified.')

        super().__init__(task=task, dataset=dataset, reasoners=reasoners, syntax=syntax)
        self.__configure_reasoners(probe)

    def extract_results(self, results: Results) -> List:
        if not results.has_energy_stats:
            raise ValueError('Missing energy stats.')

        self._logger.log(f'{results.energy_score:.2f}')
        return [results.energy_score]

    # Private

    def __configure_reasoners(self, probe_name: str) -> None:
        probe = self._loader.probe_with_name(probe_name)

        for reasoner in self._reasoners:
            reasoner.energy_probe = probe


class NotImplementedEvaluator:
    """Not implemented evaluator."""

    def start(self, resume_ontology: Optional[str] = None):
        raise NotImplementedError('Not implemented.')
