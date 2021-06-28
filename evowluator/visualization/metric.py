from typing import Optional


class Metric:

    @property
    def capitalized_name(self) -> str:
        return self.name[0].upper() + self.name[1:]

    def __init__(self, name: str, unit: Optional[str] = None, fmt: Optional[str] = None):
        self.name = name
        self.unit = unit
        self.fmt = fmt

    def to_string(self, capitalize: bool = False) -> str:
        name = self.capitalized_name if capitalize else self.name
        return f'{name} ({self.unit})' if self.unit else name
