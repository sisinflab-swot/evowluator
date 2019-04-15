import os
from importlib import import_module
from typing import Dict, List

from evowluator.config import Paths
from evowluator.pyutils.decorators import cached_property
from .base import Reasoner


class Loader:
    """Reasoner loader."""

    @property
    def reasoners(self) -> List[Reasoner]:
        """Reasoner instances."""
        return self._reasoners

    @cached_property
    def reasoners_by_name(self) -> Dict[str, Reasoner]:
        """Reasoner instances indexed by name."""
        return dict(zip([r.name for r in self.reasoners], self.reasoners))

    def __init__(self):
        modules = [file.rsplit(sep='.', maxsplit=1)[0]
                   for file in os.listdir(Paths.REASONERS_DIR)
                   if file != '__init__.py']

        package = __name__.rsplit(sep='.', maxsplit=1)[0] + '.reasoners'

        for module in modules:
            import_module('.{}'.format(module), package)

        self._reasoners = [subclass() for subclass in _all_subclasses(Reasoner)
                           if not subclass.is_template()]
        self._reasoners.sort(key=lambda r: r.name)

    def reasoners_supporting_task(self, task: str) -> List[Reasoner]:
        """Returns the reasoners that support the specified reasoning task."""
        return [r for r in self.reasoners if task in r.supported_tasks]


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])
