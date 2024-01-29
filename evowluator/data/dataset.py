from __future__ import annotations

import os
from functools import cache, cached_property
from typing import Dict, Iterable, Iterator, List
from pyutils.types.unit import MemoryMeasurement, MemoryUnit

from .sort_by import SortBy
from .syntax import Syntax
from ..config.paths import Paths
from ..reasoner.base import Reasoner, ReasoningTask


class Stats:
    """Dataset stats."""
    def __init__(self, count: int, size_bytes: int):
        self.count = count
        self.size_bytes = size_bytes
        self.size_readable: MemoryMeasurement = MemoryUnit.B(size_bytes).readable()


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
    def exists(self) -> bool:
        """Whether the ontology file exists."""
        return os.path.isfile(self.path)

    @cached_property
    def size(self) -> int:
        """File size of the ontology."""
        try:
            return os.path.getsize(self.path)
        except FileNotFoundError:
            return 0

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

    def size(self, syntaxes: Syntax | Iterable[Syntax] | None = None) -> int:
        return sum(o.size for o in self.ontologies(syntaxes=syntaxes))

    def ontology(self, syntax: Syntax) -> Ontology:
        return Ontology(self, syntax)

    def ontologies(self, syntaxes: Syntax | Iterable[Syntax] | None = None) -> Iterator[Ontology]:
        if not syntaxes:
            syntaxes = _available_syntaxes(self.base_path)
        elif isinstance(syntaxes, Syntax):
            syntaxes = (syntaxes,)

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
    def first(cls) -> Dataset | None:
        all_datasets = cls.all()
        return all_datasets[0] if all_datasets else None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def syntaxes(self) -> List[Syntax]:
        return _available_syntaxes(self.path)

    @cached_property
    def reference_syntax(self) -> Syntax:
        def count(my_iter) -> int:
            return sum(1 for _ in my_iter)

        counts_per_syntax = (
            (syntax, count(n for n in os.listdir(self.get_dir(syntax)) if not n.startswith('.')))
            for syntax in self.syntaxes
        )

        return max(counts_per_syntax, key=lambda e: e[1])[0]

    @property
    def reasoning_tasks(self) -> List[ReasoningTask]:
        return [
            r for r in ReasoningTask.all()
            if not r.requires_additional_inputs or os.path.isdir(self.get_dir(r))
        ]

    def __init__(self, name: str) -> None:
        self.path = os.path.join(Paths.DATA_DIR, name)
        self.sort_by = SortBy.NAME
        self.start_after: str | None = None
        self.preferred_syntax: Syntax | None = None

        if not os.path.isdir(self.path):
            raise FileNotFoundError('No such dataset: ' + self.name)

        if not self.syntaxes:
            raise ValueError('Invalid dataset: ' + self.name)

    def to_json_dict(self, ontologies: bool = False) -> Dict:
        stats = self.stats()
        syntaxes = sorted(self.syntaxes)
        ret = {
            'count': stats.count,
            'size_bytes': stats.size_bytes,
            'size_readable': str(stats.size_readable),
            'tasks': sorted(r.name for r in self.reasoning_tasks),
            'syntaxes': {
                syntax: {
                    'count': syntax_stats.count,
                    'size_bytes': syntax_stats.size_bytes,
                    'size_readable': str(syntax_stats.size_readable)
                }
                for syntax, syntax_stats in ((s, self.stats(s)) for s in syntaxes)
            }
        }
        if ontologies:
            ret['ontologies'] = {
                entry.name: {
                    'size_bytes': size,
                    'size_readable': str(MemoryUnit.B(size).readable()),
                    'syntaxes': {
                        syntax: {
                            'size_bytes': syntax_size,
                            'size_readable': str(MemoryUnit.B(syntax_size).readable())
                        }
                        for syntax, syntax_size in ((s, entry.size(s)) for s in syntaxes)
                        if syntax_size
                    }
                }
                for entry, size in ((e, e.size()) for e in self.get_entries())
            }
        return ret

    def syntaxes_for_reasoner(self, reasoner: Reasoner) -> List[Syntax]:
        return [s for s in reasoner.supported_syntaxes if s in self.syntaxes]

    def syntax_for_reasoner(self, reasoner: Reasoner) -> Syntax:
        if self.preferred_syntax:
            syntax = self.preferred_syntax

            if syntax not in self.syntaxes:
                msg = f'"{syntax}" syntax not available for "{self.name}" dataset.'
                raise ValueError(msg)

            supported = reasoner.supported_syntaxes

            if syntax in supported:
                return syntax

        available = self.syntaxes_for_reasoner(reasoner)

        if reasoner.preferred_syntax in available:
            return reasoner.preferred_syntax

        if available:
            return available[0]

        raise ValueError(f'No available syntax for reasoner "{reasoner.name}"')

    def stats(self, syntaxes: Syntax | Iterable[Syntax] | None = None) -> Stats:
        count, size = 0, 0

        for e in self.get_entries():
            entry_size = e.size(syntaxes=syntaxes)
            if entry_size:
                count += 1
                size += entry_size

        return Stats(count, size)

    def count(self) -> int:
        return self.stats().count

    def get_dir(self, target: Syntax | ReasoningTask) -> str:
        return os.path.join(self.path, target.name.lower())

    def get_entry(self, name: str) -> DatasetEntry:
        return DatasetEntry(self.path, name)

    def get_entries(self) -> Iterator[DatasetEntry]:
        entries = (self.get_entry(n)
                   for n in os.listdir(self.get_dir(self.reference_syntax))
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
