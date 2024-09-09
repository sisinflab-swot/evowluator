from __future__ import annotations

import os
from typing import Collection, Dict, List, Iterator

from . import metadata
from .dataset import Dataset, DatasetEntry, SortBy, Syntax
from .size_unit import SizeUnit
from ..config.key import ConfigKey
from ..config.paths import Paths


class OntologyInfo:
    """Contains information about an ontology."""

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size


class EntryInfo:
    """Contains information about a dataset entry."""

    @classmethod
    def with_entry(cls, dataset: DatasetInfo, entry: DatasetEntry) -> EntryInfo:
        return cls(dataset, entry.name, {o.syntax: o.size for o in entry.ontologies()})

    @classmethod
    def from_dict(cls, dataset: DatasetInfo, e_dict: Dict) -> EntryInfo:
        ret = cls(dataset, e_dict[ConfigKey.NAME],
                  {Syntax(s): size for s, size in e_dict[ConfigKey.SIZE].items()})
        if constructs := e_dict.get(ConfigKey.CONSTRUCTS):
            ret.constructs = constructs
        return ret

    @property
    def has_constructs_info(self) -> bool:
        return self._constructs is not None

    @property
    def max_size(self) -> int:
        return max(self.sizes.values())

    @property
    def constructs(self) -> Dict[str, int]:
        if not self._constructs:
            self._dataset.update_constructs_info()
        return self._constructs

    @constructs.setter
    def constructs(self, value: Dict[str, int]) -> None:
        self._constructs = value

    def __init__(self, dataset: DatasetInfo, name: str, sizes: Dict[Syntax, int],
                 constructs: Dict[str, int] | None = None) -> None:
        self._dataset = dataset
        self._constructs = constructs
        self.name = name
        self.sizes = sizes

    def ontology(self, size_unit: SizeUnit, syntax: Syntax) -> OntologyInfo:
        if size_unit == SizeUnit.SIZE:
            return OntologyInfo(self.name, self.sizes[syntax])
        return OntologyInfo(self.name, self.constructs[size_unit.value])

    def to_dict(self, syntaxes: List[Syntax] | None = None) -> Dict:
        sizes = {s: self.sizes[s] for s in syntaxes} if syntaxes else self.sizes
        ret = {ConfigKey.NAME: self.name, ConfigKey.SIZE: sizes}
        if self._constructs:
            ret[ConfigKey.CONSTRUCTS] = self._constructs
        return ret


class DatasetInfo:
    """Contains information about a dataset."""

    @property
    def has_constructs_info(self) -> bool:
        return next((e.has_constructs_info for e in self.entries), False)

    @property
    def dataset_is_present(self) -> bool:
        return os.path.isdir(Paths.dataset(self.name))

    @classmethod
    def with_dataset(cls, dataset: Dataset) -> DatasetInfo:
        info = cls(dataset.name, dataset.preferred_syntax, dataset.sort_by)
        info.entries = [EntryInfo.with_entry(info, e) for e in dataset.get_entries()]
        return info

    @classmethod
    def from_dict(cls, d_dict: Dict) -> DatasetInfo:
        info = cls(d_dict[ConfigKey.NAME],
                   d_dict.get(ConfigKey.SYNTAX),
                   SortBy(d_dict.get(ConfigKey.SORT_BY, SortBy.NAME)))
        info.entries = [EntryInfo.from_dict(info, d) for d in d_dict[ConfigKey.ONTOLOGIES]]
        return info

    def __init__(self, name: str, syntax: str | None, sort_by: SortBy) -> None:
        self.name = name
        self.syntax = syntax
        self.sort_by = sort_by
        self.entries: List[EntryInfo] = []

    def to_dict(self, syntaxes: List[Syntax] | None = None) -> Dict:
        dictionary = {
            ConfigKey.NAME: self.name,
            ConfigKey.SORT_BY: self.sort_by,
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
        names = [] if names is None else names
        ontologies = (e.ontology(size_unit, syntax) for e in self.entries if e.name in names)
        return sorted(ontologies, key=lambda o: o.size)

    def update_constructs_info(self) -> None:
        meta = metadata.retrieve(Dataset(self.name))
        for entry in self.entries:
            entry.constructs = meta[entry.name]
