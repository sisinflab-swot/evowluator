from __future__ import annotations

from typing import Dict, Iterable, List, Iterator

from ..config.key import ConfigKey
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
        return cls(dataset.name, dataset.preferred_syntax, dataset.sort_by,
                   (EntryInfo.with_entry(e) for e in dataset.get_entries()))

    @classmethod
    def from_dict(cls, d_dict: Dict) -> DatasetInfo:
        return cls(d_dict[ConfigKey.NAME],
                   d_dict.get(ConfigKey.SYNTAX),
                   SortBy(d_dict.get(ConfigKey.SORT_BY, SortBy.NAME)),
                   (EntryInfo.from_dict(d) for d in d_dict[ConfigKey.ONTOLOGIES]))

    def __init__(self, name: str, syntax: str | None, sort_by: SortBy,
                 entries: Iterable[EntryInfo]) -> None:
        self.name = name
        self.syntax = syntax
        self.sort_by = sort_by
        self.entries = list(entries)

    def to_dict(self, syntaxes: List[Syntax] | None = None) -> Dict:
        dictionary = {
            ConfigKey.NAME: self.name,
            ConfigKey.SORT_BY: self.sort_by
        }
        if self.syntax:
            dictionary[ConfigKey.SYNTAX] = self.syntax
        dictionary[ConfigKey.ONTOLOGIES] = [e.to_dict(syntaxes=syntaxes) for e in self.entries]
        return dictionary

    def max_ontology_size(self) -> int:
        return max(e.max_size for e in self.entries)

    def get_ontologies(self, syntax: Syntax | None = None, names: Iterable[str] | None = None,
                       sort_by: SortBy | None = None) -> Iterator[OntologyInfo]:
        if syntax is None:
            syntax = self.syntax

        if sort_by is None:
            sort_by = self.sort_by

        entries = self.entries

        if names is not None:
            entries = (e for e in entries if e.name in names)

        return sort_by.sorted(e.ontology(syntax) for e in entries)
