import re
import tempfile
import time
from abc import ABCMeta, abstractmethod
from os import path
from subprocess import TimeoutExpired
from typing import List, Optional

from evowluator import config
from evowluator.config import ConfigKey, Paths
from evowluator.data import json
from evowluator.data.csv import CSVWriter
from evowluator.data.dataset import Dataset
from evowluator.data.ontology import Ontology
from evowluator.pyutils import echo, exc, fileutils
from evowluator.pyutils.decorators import cached_property
from evowluator.pyutils.logger import Logger
from evowluator.reasoner.loader import Loader
from evowluator.reasoner.base import Reasoner
from .enum import TestName


class Test:
    """Abstract test class."""
    __metaclass__ = ABCMeta

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
                 syntax: Optional[str] = None):
        self._loader = Loader()
        self._dataset = Dataset.with_name(dataset) if dataset else Dataset.first()
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
            self.__start_test(resume_ontology)

        fileutils.chmod(self.work_dir, 0o666, recursive=True, dir_mode=0o777)

    # Private

    def __start_test(self, resume_ontology: Optional[str] = None) -> None:
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
        syntax = reasoner.syntax_for_requested(self._syntax)

        syntaxes = reasoner.supported_syntaxes
        syntaxes.sort()

        syntaxes = ['[{}]'.format(s) if s == syntax else s for s in reasoner.supported_syntaxes]
        self._logger.log('{}: {}'.format(reasoner.name, ' '.join(syntaxes)))

    def __save_config(self) -> None:

        cfg = {
            ConfigKey.NAME: self.name,
            ConfigKey.DATASET: self._dataset.name,
            ConfigKey.REASONERS: [{
                ConfigKey.NAME: r.name,
                ConfigKey.SYNTAX: r.syntax_for_requested(self._syntax)
            } for r in self._reasoners]
        }

        json.save(cfg, path.join(self.work_dir, config.Paths.CONFIG_FILE_NAME))


# noinspection PyTypeChecker
class StandardPerformanceTest(Test):
    """Abstract test class for measuring the performance of standard reasoning tasks."""
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def result_fields(self) -> List[str]:
        """Names for the columns of the CSV results."""
        pass

    @abstractmethod
    def run_reasoner(self, reasoner: Reasoner, ontology: Ontology) -> List[str]:
        """Called every run, for each reasoner and each ontology.

        :return : Values for the CSV result fields.
        """
        pass

    def __init__(self,
                 dataset: Optional[str] = None,
                 reasoners: Optional[List[str]] = None,
                 syntax: Optional[str] = None,
                 iterations: int = 1):
        """
        :param dataset : If specified, run the test on the specified dataset.
        :param reasoners : If specified, limit the test to the specified reasoners.
        :param syntax : If true, the test is run on the specified OWL syntax whenever possible.
        :param iterations : Number of iterations per ontology.
        """
        super().__init__(dataset, reasoners, syntax)
        self.iterations = iterations

    def setup(self):
        csv_header = ['Ontology']

        for reasoner in self._reasoners:
            for field in self.result_fields:
                csv_header.append('{}: {}'.format(reasoner.name, field))

        self._csv_writer.write_row(csv_header)

    def run(self, entry):

        fail: List[str] = []

        for iteration in range(self.iterations):
            self._logger.log('Run {}:'.format(iteration + 1), color=echo.Color.YELLOW)
            self._logger.indent_level += 1

            csv_row = [entry.name]

            for reasoner in self._reasoners:
                self._logger.log('- {}: '.format(reasoner.name), endl=False)
                ontology = entry.ontology(reasoner.syntax_for_requested(self._syntax))

                # Skip already failed or timed out.
                if reasoner.name in fail:
                    csv_row.extend(['skip'] * len(self.result_fields))
                    self._logger.log('skip')
                    continue

                try:
                    csv_row.extend(self.run_reasoner(reasoner, ontology))
                except TimeoutExpired:
                    csv_row.extend(['timeout'] * len(self.result_fields))
                    self._logger.log('timeout')
                    fail.append(reasoner.name)
                except Exception as e:
                    if config.DEBUG:
                        raise e

                    csv_row.extend(['error'] * len(self.result_fields))
                    self._logger.log('error')
                    fail.append(reasoner.name)

            self._logger.indent_level -= 1
            self._logger.log('')
            self._csv_writer.write_row(csv_row)


class NotImplementedTest(Test):
    """Not implemented test."""

    @property
    def name(self):
        return TestName.Misc.NOT_IMPLEMENTED

    @property
    def default_reasoners(self):
        return []

    def setup(self):
        raise NotImplementedError('Not implemented.')

    def run(self, entry):
        del entry  # Unused
        pass
