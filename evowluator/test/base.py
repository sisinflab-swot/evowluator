import re
import tempfile
import time
from abc import ABC, abstractmethod
from os import path
from typing import List, Optional

from evowluator import config
from evowluator.config import ConfigKey, Paths
from evowluator.data import json
from evowluator.data.csv import CSVWriter
from evowluator.data.dataset import Dataset
from evowluator.pyutils import echo, exc, fileutils
from evowluator.pyutils.decorators import cached_property
from evowluator.pyutils.logger import Logger
from evowluator.reasoner.base import Reasoner
from evowluator.reasoner.loader import Loader


class Test(ABC):
    """Abstract test class."""

    # Override

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this test."""
        pass

    @property
    @abstractmethod
    def default_reasoners(self) -> List[Reasoner]:
        """Default reasoners for this test."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Called before the test starts iterating on ontologies."""
        pass

    @abstractmethod
    def run(self, entry: Dataset.Entry) -> None:
        """Runs test over a single ontology."""
        pass

    # Public

    @cached_property
    def work_dir(self) -> str:
        """Work directory for this test."""
        name = re.sub(r"[^\w\s]", '', self.name)
        name = re.sub(r"\s+", '_', name)
        prefix = time.strftime('{}_%Y%m%d_%H%M%S_'.format(name))
        fileutils.create_dir(Paths.RESULTS_DIR)
        return tempfile.mkdtemp(dir=Paths.RESULTS_DIR, prefix=prefix)

    @cached_property
    def temp_dir(self) -> str:
        """Temp directory for this test."""
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
                 syntax: Optional[str] = None) -> None:
        self._dataset = Dataset.with_name(dataset) if dataset else Dataset.first()

        if syntax and syntax not in self._dataset.syntaxes:
            msg = '"{}" syntax not available in "{}" dataset.'.format(syntax, self._dataset.name)
            raise ValueError(msg)

        self._loader = Loader()
        self._syntax = syntax
        self._logger: Logger = None
        self._csv_writer: CSVWriter = None

        if reasoners:
            try:
                self._reasoners = [self._loader.reasoners_by_name[n] for n in reasoners]
            except KeyError as e:
                exc.re_raise_new_message(e, 'No such reasoner: ' + str(e))
        else:
            self._reasoners = self.default_reasoners

    def clear_temp(self) -> None:
        """Clears temporary files."""
        fileutils.remove_dir_contents(self.temp_dir)

    def start(self, resume_ontology: Optional[str] = None) -> None:
        """Starts the test."""
        self._logger = Logger(path.join(self.work_dir, config.Paths.LOG_FILE_NAME))
        self._csv_writer = CSVWriter(path.join(self.work_dir, config.Paths.RESULTS_FILE_NAME))
        self.__save_config()

        with self._logger, self._csv_writer:
            self._logger.clear()
            self.__log_config()
            self.setup()
            self._start_test(resume_ontology)

        fileutils.chmod(self.work_dir, 0o666, recursive=True, dir_mode=0o777)

    # Protected

    def _syntaxes_for_reasoner(self, reasoner: Reasoner) -> List[str]:
        available = self._dataset.syntaxes
        return [s for s in reasoner.supported_syntaxes if s in available]

    def _syntax_for_reasoner(self, reasoner: Reasoner) -> Optional[str]:
        supported = reasoner.supported_syntaxes

        if self._syntax in supported:
            return self._syntax

        available = self._syntaxes_for_reasoner(reasoner)

        if reasoner.preferred_syntax in available:
            return reasoner.preferred_syntax

        return available[0] if available else None

    def _usable_reasoners(self) -> List[Reasoner]:
        return [r for r in self._reasoners if self._syntaxes_for_reasoner(r)]

    def _start_test(self, resume_ontology: Optional[str] = None) -> None:
        for entry in self._dataset.get_entries(resume_after=resume_ontology):
            sizes = ('{}: {}'.format(o.syntax, o.readable_size) for o in entry.ontologies())
            size_str = ' | '.join(sizes)

            self._logger.log('{}'.format(entry.name), color=echo.Color.YELLOW, endl=False)
            self._logger.log(' ({})'.format(size_str))
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

    def __log_config(self) -> None:
        self._logger.log('Selected reasoners and serializations:', color=echo.Color.GREEN)
        self._logger.indent_level += 1

        for reasoner in self._reasoners:
            self.__log_syntaxes(reasoner)

        self._logger.indent_level -= 1

        msg = '\nStarting {} test on "{}" dataset ({} ontologies)...\n'.format(self.name,
                                                                               self._dataset.name,
                                                                               self._dataset.size)
        self._logger.log(msg, color=echo.Color.GREEN)

    def __log_syntaxes(self, reasoner: Reasoner) -> None:
        syntaxes = self._syntaxes_for_reasoner(reasoner)
        syntaxes.sort()

        if not syntaxes:
            self._logger.log('{}: no syntaxes'.format(reasoner.name))
            return

        syntax = self._syntax_for_reasoner(reasoner)
        syntaxes = ['[{}]'.format(s) if s == syntax else s for s in syntaxes]
        self._logger.log('{}: {}'.format(reasoner.name, ' '.join(syntaxes)))

    def __save_config(self) -> None:

        cfg = {
            ConfigKey.NAME: self.name,
            ConfigKey.DATASET: self._dataset.name,
            ConfigKey.REASONERS: [{
                ConfigKey.NAME: r.name,
                ConfigKey.SYNTAX: self._syntax_for_reasoner(r)
            } for r in self._usable_reasoners()]
        }

        json.save(cfg, path.join(self.work_dir, config.Paths.CONFIG_FILE_NAME))


class ReasoningTest(Test):
    """Abstract reasoning test class."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """Test mode."""

    # Overrides

    @property
    def name(self) -> str:
        return '{} {}'.format(self.task, self.mode)

    @property
    def default_reasoners(self) -> List[Reasoner]:
        return self._loader.reasoners_supporting_task(self.task)

    def __init__(self,
                 task: str,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None) -> None:
        self.task = task
        super().__init__(dataset=dataset, reasoners=reasoners, syntax=syntax)


class NotImplementedTest:
    """Not implemented test."""

    def start(self, resume_ontology: Optional[str] = None):
        raise NotImplementedError('Not implemented.')
