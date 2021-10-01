from __future__ import annotations

from enum import Enum
from typing import List, Type, TypeVar

T = TypeVar('T', bound='StrEnum')


class StrEnum(Enum):

    @classmethod
    def all(cls: Type[T]) -> List[T]:
        return [v for v in cls]

    @property
    def value(self) -> str:
        return self._value_

    def __str__(self):
        return self.value

    def __lt__(self, other) -> bool:
        return self.value < other.value
