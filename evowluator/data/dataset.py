from __future__ import annotations

import os
from typing import Iterable, Iterator, List

from .ontology import Ontology, Syntax
from ..config import Paths
from ..reasoner.base import ReasoningTask


class Dataset:
    """Models a dataset containing multiple ontologies."""

    class Entry:
        """Represents an ontology in any of its provided serializations."""

        @property
        def max_size(self) -> int:
            return max(o.size for o in self.ontologies())

        def __init__(self, dataset_dir: str, name: str) -> None:
            self.dataset_dir = dataset_dir
            self.name = name

        def ontology(self, syntax: Syntax) -> Ontology:
            return Ontology(os.path.join(self.dataset_dir, syntax.value, self.name), syntax)

        def ontologies(self, syntaxes: Iterable[Syntax] | None = None) -> Iterator[Ontology]:
            if not syntaxes:
                syntaxes = _available_syntaxes(self.dataset_dir)

            for s in syntaxes:
                yield self.ontology(s)

        def inputs_for_task(self, task: ReasoningTask,
                            syntax: Syntax | None = None) -> Iterator[Dataset.Entry]:
            input_dir = os.path.join(self.dataset_dir, task.name, os.path.splitext(self.name)[0])

            try:
                if not syntax:
                    syntax = _available_syntaxes(input_dir)[0]

                for n in sorted(f for f in os.listdir(os.path.join(input_dir, syntax.value))
                                if not f.startswith('.')):
                    yield Dataset.Entry(input_dir, n)
            except (IndexError, FileNotFoundError):
                return

        def inputs_count_for_task(self, task: ReasoningTask) -> int:
            return sum(1 for _ in self.inputs_for_task(task))

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
    def size(self) -> int:
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

    def get_entry(self, name: str) -> Entry:
        return Dataset.Entry(self.path, name)

    def get_ontology(self, name: str, syntax: Syntax) -> Ontology:
        return self.get_entry(name).ontology(syntax)

    def get_max_ontology_size(self) -> int:
        return max(e.max_size for e in self.get_entries())

    def get_ontologies(self, syntax: Syntax, names: Iterable[str] | None = None,
                       sort_by_size: bool = False) -> Iterator[Ontology]:
        entries = self.get_entries()

        if names is not None:
            entries = (e for e in entries if e.name in names)

        ontologies = (e.ontology(syntax) for e in entries)
        return sorted(ontologies, key=lambda o: o.size) if sort_by_size else ontologies

    def get_entries(self, resume_after: str | None = None) -> Iterator[Entry]:
        onto_dir = self.get_dir(self.syntaxes[0])
        onto_names = sorted(f for f in os.listdir(onto_dir) if not f.startswith('.'))

        for onto_name in onto_names:
            if resume_after:
                if onto_name == resume_after:
                    resume_after = None
                continue

            yield Dataset.Entry(self.path, onto_name)


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
