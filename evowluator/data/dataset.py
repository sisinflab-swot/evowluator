from __future__ import annotations

import os
from functools import cache, cached_property
from typing import Iterable, Iterator, List

from .sort_by import SortBy
from .syntax import Syntax
from ..config.paths import Paths
from ..reasoner.base import ReasoningTask


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

    def __init__(self, name: str) -> None:
        self.path = os.path.join(Paths.DATA_DIR, name)
        self.sort_by = SortBy.NAME
        self.start_after: str | None = None

        if not os.path.isdir(self.path):
            raise FileNotFoundError('No such dataset: ' + self.name)

        if not self.syntaxes:
            raise ValueError('Invalid dataset: ' + self.name)

    def cumulative_stats(self, syntaxes: Iterable[Syntax] | None = None) -> (int, int):
        count, size = 0, 0

        for e in self.get_entries():
            count += 1
            size += e.cumulative_size(syntaxes=syntaxes)

        return count, size

    def count(self) -> int:
        return self.cumulative_stats()[0]

    def cumulative_size(self, syntaxes: Iterable[Syntax] | None = None) -> int:
        return self.cumulative_stats(syntaxes=syntaxes)[1]

    def get_dir(self, syntax: Syntax) -> str:
        return os.path.join(self.path, syntax)

    def get_entry(self, name: str) -> DatasetEntry:
        return DatasetEntry(self.path, name)

    def get_ontology(self, name: str, syntax: Syntax) -> Ontology:
        return self.get_entry(name).ontology(syntax)

    def get_entries(self) -> Iterator[DatasetEntry]:
        entries = (self.get_entry(n)
                   for n in os.listdir(self.get_dir(self.syntaxes[0]))
                   if not n.startswith('.'))
        entries = self.sort_by.sorted(entries, size_attr='max_size')
        should_yield = False if self.start_after else True

        for entry in entries:
            if not should_yield:
                if entry.name == self.start_after:
                    should_yield = True
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
