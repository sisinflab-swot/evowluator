from __future__ import annotations

import os
from importlib import import_module
from typing import List

from pyutils.proc.bench import EnergyProbe

from evowluator.config import Paths
from evowluator.reasoner.base import Reasoner, ReasoningTask


class Loader:
    """User modules loader."""

    def __init__(self):
        _import_modules(Paths.REASONERS_DIR)
        _import_modules(Paths.PROBES_DIR)

        self.reasoners = [subclass() for subclass in _all_subclasses(Reasoner)
                          if not subclass.is_template()]
        self.reasoners.sort(key=lambda r: r.name)

        self.probes = [subclass() for subclass in _all_subclasses(EnergyProbe)]
        self.probes.sort(key=lambda p: p.__class__.__name__)

    def reasoner_with_name(self, name: str) -> Reasoner | None:
        """Returns the reasoner having the specified name."""
        lower_name = name.lower()
        return next((r for r in self.reasoners if r.name.lower() == lower_name), None)

    def reasoners_supporting_task(self, task: ReasoningTask) -> List[Reasoner]:
        """Returns the reasoners that support the specified reasoning task."""
        return [r for r in self.reasoners if task in r.supported_tasks]

    def probe_with_name(self, name: str) -> EnergyProbe | None:
        """Returns the energy probe having the specified name."""
        lower_name = name.lower()
        possible = [lower_name, lower_name + 'probe']
        return next((p for p in self.probes if p.__class__.__name__.lower() in possible), None)


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])


def _import_modules(directory: str):
    modules = [file.rsplit(sep='.', maxsplit=1)[0]
               for file in os.listdir(directory)
               if file.endswith('.py') and not file.startswith('_')]

    package = directory[len(Paths.ROOT_DIR) + 1:].replace(os.path.sep, '.')

    for module in modules:
        import_module(f'.{module}', package)
