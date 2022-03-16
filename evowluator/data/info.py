from __future__ import annotations

from typing import Dict, Iterable, List, Iterator

from ..config import ConfigKey
from ..data.dataset import Dataset, DatasetEntry, SortBy, Syntax


class OntologyInfo:
    """Contains information about an ontology."""

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size


class EntryInfo:
    """Contains information about a dataset entry."""

    @classmethod
    def with_entry(cls, entry: DatasetEntry) -> EntryInfo:
        return cls(entry.name, {o.syntax: o.size for o in entry.ontologies()})

    @classmethod
    def from_dict(cls, e_dict: Dict) -> EntryInfo:
        return cls(e_dict[ConfigKey.NAME],
                   {Syntax(s): size for s, size in e_dict[ConfigKey.SIZE].items()})

    @property
    def max_size(self) -> int:
        return max(self.sizes.values())

    def __init__(self, name: str, sizes: Dict[Syntax, int]) -> None:
        self.name = name
        self.sizes = sizes

    def ontology(self, syntax: Syntax) -> OntologyInfo:
        return OntologyInfo(self.name, self.sizes[syntax])

    def to_dict(self, syntaxes: List[Syntax] | None = None) -> Dict:
        sizes = {s: self.sizes[s] for s in syntaxes} if syntaxes else self.sizes
        return {ConfigKey.NAME: self.name, ConfigKey.SIZE: sizes}


class DatasetInfo:
    """Contains information about a dataset."""

    @classmethod
    def with_dataset(cls, dataset: Dataset) -> DatasetInfo:
        return cls(dataset.name, (EntryInfo.with_entry(e) for e in dataset.get_entries()))

    @classmethod
    def from_dict(cls, d_dict: Dict) -> DatasetInfo:
        return cls(d_dict[ConfigKey.NAME],
                   (EntryInfo.from_dict(d) for d in d_dict[ConfigKey.ONTOLOGIES]))

    def __init__(self, name: str, entries: Iterable[EntryInfo]) -> None:
        self.name = name
        self.entries = list(entries)

    def to_dict(self, syntaxes: List[Syntax] | None = None) -> Dict:
        return {
            ConfigKey.NAME: self.name,
            ConfigKey.ONTOLOGIES: [e.to_dict(syntaxes=syntaxes) for e in self.entries]
        }

    def max_ontology_size(self) -> int:
        return max(e.max_size for e in self.entries)

    def get_ontologies(self, syntax: Syntax, names: Iterable[str] | None = None,
                       sort_by: SortBy = SortBy.NAME) -> Iterator[OntologyInfo]:
        entries = self.entries

        if names is not None:
            entries = (e for e in entries if e.name in names)

        return sort_by.sorted(e.ontology(syntax) for e in entries)
