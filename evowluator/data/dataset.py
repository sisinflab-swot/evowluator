from __future__ import annotations

import os
from functools import cache, cached_property
from operator import attrgetter
from typing import Iterable, Iterator, List

from pyutils.types.strenum import StrEnum
from .syntax import Syntax
from ..config import Paths
from ..reasoner.base import ReasoningTask


class SortBy(StrEnum):
    """Sort-by strategies."""
    NAME_ASC = 'name'
    NAME_DESC = 'name-desc'
    SIZE_ASC = 'size'
    SIZE_DESC = 'size-desc'

    NAME = NAME_ASC
    SIZE = SIZE_ASC

    def sorted(self, what: Iterable, name_attr: str = 'name', size_attr: str = 'size'):
        attr = size_attr if self in (SortBy.SIZE_ASC, SortBy.SIZE_DESC) else name_attr
        reverse = self.endswith('-desc')
        return sorted(what, key=attrgetter(attr), reverse=reverse)


class Ontology:
    """Models ontology files."""

    @property
    def path(self) -> str:
        """Path of the ontology."""
        return os.path.join(self.entry.base_path, self.syntax, self.entry.name)

    @property
    def name(self) -> str:
        """File name of the ontology."""
        return self.entry.name

    @cached_property
    def size(self) -> int:
        """File size of the ontology."""
        return os.path.getsize(self.path)

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

    def cumulative_size(self, syntaxes: Iterable[Syntax] | None = None) -> int:
        return sum(o.size for o in self.ontologies(syntaxes=syntaxes))

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
            for n in sorted(f for f in os.listdir(os.path.join(input_dir, syntax))
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
    def syntaxes(self) -> List[Syntax]:
        return _available_syntaxes(self.path)

    def __init__(self, path: str) -> None:
        self.path = path

        if not os.path.isdir(path):
            raise FileNotFoundError('No such dataset: ' + self.name)

        if not self.syntaxes:
            raise ValueError('Invalid dataset: ' + self.name)

    def cumulative_stats(self, syntaxes: Iterable[Syntax] | None = None,
                         sort_by: SortBy = SortBy.NAME,
                         resume_after: str | None = None) -> (int, int):
        count, size = 0, 0

        for e in self.get_entries(sort_by=sort_by, resume_after=resume_after):
            count += 1
            size += e.cumulative_size(syntaxes=syntaxes)

        return count, size

    def count(self, sort_by: SortBy = SortBy.NAME, resume_after: str | None = None) -> int:
        return self.cumulative_stats(sort_by=sort_by, resume_after=resume_after)[0]

    def cumulative_size(self, syntaxes: Iterable[Syntax] | None = None,
                        sort_by: SortBy = SortBy.NAME, resume_after: str | None = None) -> int:
        return self.cumulative_stats(syntaxes=syntaxes, sort_by=sort_by,
                                     resume_after=resume_after)[1]

    def get_dir(self, syntax: Syntax) -> str:
        return os.path.join(self.path, syntax)

    def get_entry(self, name: str) -> DatasetEntry:
        return DatasetEntry(self.path, name)

    def get_ontology(self, name: str, syntax: Syntax) -> Ontology:
        return self.get_entry(name).ontology(syntax)

    def get_entries(self, sort_by: SortBy = SortBy.NAME,
                    resume_after: str | None = None) -> Iterator[DatasetEntry]:
        entries = (self.get_entry(n)
                   for n in os.listdir(self.get_dir(self.syntaxes[0]))
                   if not n.startswith('.'))
        entries = sort_by.sorted(entries, size_attr='max_size')

        for entry in entries:
            if resume_after:
                if entry.name == resume_after:
                    resume_after = None
                continue

            yield entry


# Private


@cache
def _available_syntaxes(dataset_dir: str) -> List[Syntax]:
    syntaxes = []

    for name in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, name)):
            try:
                syntaxes.append(Syntax(name))
            except ValueError:
                pass

    return syntaxes
