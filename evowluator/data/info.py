from __future__ import annotations

from typing import Collection, Dict, Iterable, List, Iterator

from . import metadata
from .dataset import Dataset, DatasetEntry, SortBy, Syntax
from .size_unit import SizeUnit
from ..config.key import ConfigKey


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
        ret = cls(e_dict[ConfigKey.NAME],
                  {Syntax(s): size for s, size in e_dict[ConfigKey.SIZE].items()})
        if constructs := e_dict.get(ConfigKey.CONSTRUCTS):
            ret.constructs = constructs
        return ret

    @property
    def max_size(self) -> int:
        return max(self.sizes.values())

    def __init__(self, name: str, sizes: Dict[Syntax, int],
                 constructs: Dict[str, int] | None = None) -> None:
        self.name = name
        self.sizes = sizes
        self.constructs = constructs

    def ontology(self, size_unit: SizeUnit, syntax: Syntax) -> OntologyInfo:
        if size_unit == SizeUnit.SIZE:
            return OntologyInfo(self.name, self.sizes[syntax])
        return OntologyInfo(self.name, self.constructs[size_unit.value])

    def to_dict(self, syntaxes: List[Syntax] | None = None) -> Dict:
        sizes = {s: self.sizes[s] for s in syntaxes} if syntaxes else self.sizes
        ret = {ConfigKey.NAME: self.name, ConfigKey.SIZE: sizes}
        if self.constructs:
            ret[ConfigKey.CONSTRUCTS] = self.constructs
        return ret


class DatasetInfo:
    """Contains information about a dataset."""

    @classmethod
    def with_dataset(cls, dataset: Dataset) -> DatasetInfo:
        return cls(dataset.name, dataset.preferred_syntax, dataset.sort_by,
                   (EntryInfo.with_entry(e) for e in dataset.get_entries()))

    @classmethod
    def from_dict(cls, d_dict: Dict) -> DatasetInfo:
        info = cls(d_dict[ConfigKey.NAME],
                   d_dict.get(ConfigKey.SYNTAX),
                   SortBy(d_dict.get(ConfigKey.SORT_BY, SortBy.NAME)),
                   (EntryInfo.from_dict(d) for d in d_dict[ConfigKey.ONTOLOGIES]))
        info._constructs_info = bool(d_dict.get(ConfigKey.CONSTRUCTS, False))
        return info

    def __init__(self, name: str, syntax: str | None, sort_by: SortBy,
                 entries: Iterable[EntryInfo]) -> None:
        self.name = name
        self.syntax = syntax
        self.sort_by = sort_by
        self.entries = list(entries)
        self._constructs_info = False

    def to_dict(self, syntaxes: List[Syntax] | None = None) -> Dict:
        dictionary = {
            ConfigKey.NAME: self.name,
            ConfigKey.SORT_BY: self.sort_by,
            ConfigKey.CONSTRUCTS: self._constructs_info,
        }
        if self.syntax:
            dictionary[ConfigKey.SYNTAX] = self.syntax
        dictionary[ConfigKey.ONTOLOGIES] = [e.to_dict(syntaxes=syntaxes) for e in self.entries]
        return dictionary

    def max_ontology_size(self, size_unit: SizeUnit = SizeUnit.SIZE) -> int:
        if size_unit == SizeUnit.SIZE:
            return max(e.max_size for e in self.entries)
        return max(e.constructs[size_unit] for e in self.entries)

    def get_ontologies(self, syntax: Syntax | None = None,
                       names: Collection[str] | None = None,
                       size_unit: SizeUnit = SizeUnit.SIZE) -> Iterator[OntologyInfo]:
        if syntax is None:
            syntax = self.syntax

        if size_unit != SizeUnit.SIZE:
            self.retrieve_constructs_info()

        names = [] if names is None else names
        ontologies = (e.ontology(size_unit, syntax) for e in self.entries if e.name in names)
        return sorted(ontologies, key=lambda o: o.size)

    def retrieve_constructs_info(self) -> bool:
        if self._constructs_info:
            return False
        meta = metadata.retrieve(Dataset(self.name))
        for entry in self.entries:
            entry.constructs = meta[entry.name]
        self._constructs_info = True
        return True
