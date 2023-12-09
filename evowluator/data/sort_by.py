from operator import attrgetter
from typing import Iterable

from pyutils.types.strenum import StrEnum


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
