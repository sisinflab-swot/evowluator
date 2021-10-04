from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import cached_property
from os import path
from typing import List

from pyutils import exc
from pyutils.io import echo, fileutils
from pyutils.io.echo import Color
from pyutils.io.fileutils import readable_bytes, readable_scale_and_unit
from pyutils.io.logger import Logger
from .. import config
from ..config import ConfigKey, Paths
from ..data import json
from ..data.csv import CSVWriter
from ..data.dataset import Dataset, DatasetEntry, SortBy, Syntax
from ..data.info import DatasetInfo
from ..reasoner.base import Reasoner


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
    def run(self, entry: DatasetEntry) -> None:
        """Runs the evaluator over a single ontology."""
        pass

    # Public

    @cached_property
    def work_dir(self) -> str:
        """Work directory."""
        return Paths.new_results_dir(self.name)

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
                 dataset: str | None = None,
                 reasoners: List[str] | None = None,
                 syntax: Syntax | None = None) -> None:
        self._dataset = Dataset.with_name(dataset) if dataset else Dataset.first()

        if syntax and syntax not in self._dataset.syntaxes:
            msg = f'"{syntax}" syntax not available for "{self._dataset.name}" dataset.'
            raise ValueError(msg)

        self._syntax = syntax
        self._logger: Logger | None = None
        self._csv_writer: CSVWriter | None = None

        if reasoners:
            try:
                self._reasoners = [Reasoner.with_name(n) for n in reasoners]
            except KeyError as e:
                exc.re_raise_new_message(e, 'No such reasoner: ' + str(e))
        else:
            self._reasoners = Reasoner.local()

    def clear_temp(self) -> None:
        """Clears temporary files."""
        fileutils.remove_dir_contents(self.temp_dir)

    def start(self, sort_by: SortBy = SortBy.NAME, resume_ontology: str | None = None) -> None:
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
                self._start(sort_by=sort_by, resume_ontology=resume_ontology)
        finally:
            fileutils.chmod(self.work_dir, 0o666, recursive=True, dir_mode=0o777)
            self.__teardown_reasoners()
            echo.success('Evaluation results: ', endl=False)
            echo.info(self.work_dir)

    # Protected

    def _output_path_for_reasoner(self, reasoner: Reasoner) -> str:
        return os.path.join(self.temp_dir, reasoner.name.lower().replace(' ', '_'))

    def _syntaxes_for_reasoner(self, reasoner: Reasoner) -> List[Syntax]:
        available = self._dataset.syntaxes
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

    def _start(self, sort_by: SortBy = SortBy.NAME, resume_ontology: str | None = None) -> None:
        used_syntaxes = self._syntaxes()
        dataset_count = self._dataset.count
        dataset_size = self._dataset.cumulative_size(syntaxes=used_syntaxes)

        scale, _ = readable_scale_and_unit(dataset_size)
        dataset_size = readable_bytes(dataset_size)
        cumulative_size = 0

        for idx, entry in enumerate(self._dataset.get_entries(sort_by=sort_by,
                                                              resume_after=resume_ontology)):
            sizes = list(sorted((o.syntax, o.size) for o in entry.ontologies()))
            size_str = ' | '.join(f'{syntax}: {readable_bytes(size)}' for syntax, size in sizes)
            cumulative_size += sum(size for syntax, size in sizes if syntax in used_syntaxes)

            self._logger.log(f'\n{entry.name}', color=Color.GREEN)
            self._logger.log('Sizes: ', color=Color.YELLOW, endl=False)
            self._logger.log(size_str, endl=False)
            self._logger.log('  Progress: ', color=Color.YELLOW, endl=False)
            self._logger.log((f'{idx + 1}/{dataset_count} | '
                              f'{cumulative_size / scale:.1f}/{dataset_size}'))
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
        self._logger.log('Selected reasoners and serializations:', color=Color.GREEN)
        self._logger.indent_level += 1
        for reasoner in self._reasoners:
            self.__log_syntaxes(reasoner)
        self._logger.indent_level -= 1

        self._logger.log('\nSelected dataset: ', color=Color.GREEN)
        self._logger.indent_level += 1
        self._logger.log(f'{self._dataset.name}: {self._dataset.count} ontologies')
        self._logger.indent_level -= 1

        self._logger.log(f'\nStarting {self.name} evaluation...\n', color=Color.GREEN)

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
            ConfigKey.DATASET: DatasetInfo.with_dataset(self._dataset).to_dict(self._syntaxes()),
            ConfigKey.REASONERS: [{
                ConfigKey.NAME: r.name,
                ConfigKey.SYNTAX: self._syntax_for_reasoner(r).value
            } for r in self._usable_reasoners()]
        }

        json.save(cfg, path.join(self.work_dir, config.Paths.CONFIG_FILE_NAME))


class NotImplementedEvaluator:
    """Not implemented evaluator."""

    def start(self, resume_ontology: str | None = None):
        raise NotImplementedError('Not implemented.')
