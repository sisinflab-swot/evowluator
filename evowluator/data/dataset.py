from __future__ import annotations

import os
from typing import Iterable, Iterator, List

from pyutils.io import fileutils
from .syntax import Syntax
from ..config import Paths
from ..reasoner.base import ReasoningTask


class Ontology:
    """Models ontology files."""

    @property
    def path(self) -> str:
        """Path of the ontology."""
        return os.path.join(self.entry.base_path, self.syntax.value, self.entry.name)

    @property
    def name(self) -> str:
        """File name of the ontology."""
        return self.entry.name

    @property
    def size(self) -> int:
        """File size of the ontology."""
        return os.path.getsize(self.path)

    @property
    def readable_size(self) -> str:
        """Human readable string for the ontology size."""
        return fileutils.human_readable_bytes(self.size)

    def __init__(self, entry: DatasetEntry, syntax: Syntax):
        self.entry = entry
        self.syntax = syntax


class DatasetEntry:
    """Represents an ontology in any of its provided serializations."""

    @property
    def max_size(self) -> int:
        return max(o.size for o in self.ontologies())

    def __init__(self, base_path: str, name: str) -> None:
        self.base_path = base_path
        self.name = name

    def ontology(self, syntax: Syntax) -> Ontology:
        return Ontology(self, syntax)

    def ontologies(self, syntaxes: Iterable[Syntax] | None = None) -> Iterator[Ontology]:
        if not syntaxes:
            syntaxes = _available_syntaxes(self.base_path)

        for s in syntaxes:
            yield self.ontology(s)

    def inputs_for_task(self, task: ReasoningTask) -> Iterator[DatasetEntry]:
        input_dir = os.path.join(self.base_path, task.name, os.path.splitext(self.name)[0])

        try:
            syntax = _available_syntaxes(input_dir)[0]
            for n in sorted(f for f in os.listdir(os.path.join(input_dir, syntax.value))
                            if not f.startswith('.')):
                yield DatasetEntry(input_dir, n)
        except (IndexError, FileNotFoundError):
            return

    def inputs_count_for_task(self, task: ReasoningTask) -> int:
        return sum(1 for _ in self.inputs_for_task(task))


class Dataset:
    """Models a dataset containing multiple ontologies."""

    @classmethod
    def with_name(cls, name: str) -> Dataset:
        return Dataset(os.path.join(Paths.DATA_DIR, name))

    @classmethod
    def with_names(cls, names: List[str] | None = None) -> List[Dataset]:
        return [Dataset.with_name(d) for d in names] if names else cls.all()

    @classmethod
    def all(cls) -> List[Dataset]:
        data_dir = Paths.DATA_DIR

        datasets = (os.path.join(data_dir, d) for d in os.listdir(data_dir))
        datasets = sorted(d for d in datasets if os.path.isdir(d))

        return [Dataset(d) for d in datasets]

    @classmethod
    def first(cls) -> Dataset:
        all_datasets = cls.all()

        if not all_datasets:
            raise FileNotFoundError('No datasets provided.')

        return all_datasets[0]

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def count(self) -> int:
        return sum(1 for _ in self.get_entries())

    @property
    def syntaxes(self) -> List[Syntax]:
        return _available_syntaxes(self.path)

    def __init__(self, path: str) -> None:
        self.path = path

        if not os.path.isdir(path):
            raise FileNotFoundError('No such dataset: ' + self.name)

        if not self.syntaxes:
            raise ValueError('Invalid dataset: ' + self.name)

    def get_dir(self, syntax: Syntax) -> str:
        return os.path.join(self.path, syntax.value)

    def get_entry(self, name: str) -> DatasetEntry:
        return DatasetEntry(self.path, name)

    def get_ontology(self, name: str, syntax: Syntax) -> Ontology:
        return self.get_entry(name).ontology(syntax)

    def get_entries(self, sort_by_size: bool = False,
                    resume_after: str | None = None) -> Iterator[DatasetEntry]:
        entries = (self.get_entry(n)
                   for n in os.listdir(self.get_dir(self.syntaxes[0]))
                   if not n.startswith('.'))

        if sort_by_size:
            entries = sorted(entries, key=lambda e: e.max_size)
        else:
            entries = sorted(entries, key=lambda e: e.name)

        for entry in entries:
            if resume_after:
                if entry.name == resume_after:
                    resume_after = None
                continue

            yield entry


# Private


def _available_syntaxes(dataset_dir: str) -> List[Syntax]:
    syntaxes = []

    for name in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, name)):
            try:
                syntaxes.append(Syntax(name))
            except ValueError:
                pass

    return syntaxes
