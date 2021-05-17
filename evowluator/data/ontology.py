from __future__ import annotations

import os

from pyutils.io import fileutils

from evowluator.util import owltool
from evowluator.util.strenum import StrEnum


class Syntax(StrEnum):
    """OWL ontology syntaxes."""

    DL = 'dl'
    """DL syntax."""

    FUNCTIONAL = 'functional'
    """Functional syntax."""

    KRSS = 'krss'
    """KRSS syntax."""

    KRSS2 = 'krss2'
    """KRSS2 syntax."""

    MANCHESTER = 'manchester'
    """Manchester syntax."""

    OBO = 'obo'
    """OBO syntax."""

    OWLXML = 'owlxml'
    """OWL/XML syntax."""

    RDFXML = 'rdfxml'
    """RDF/XML syntax."""

    TURTLE = 'turtle'
    """Turtle syntax."""


class Ontology:
    """Models ontology files."""

    class ConversionResult(StrEnum):
        """Ontology conversion result."""
        SUCCESS = 'done'
        ALREADY_CONVERTED = 'already converted'
        ERROR = 'error'

    @property
    def name(self) -> str:
        """The file name of the ontology."""
        return os.path.basename(self.path)

    @property
    def size(self) -> int:
        """Size of the ontology in bytes."""
        return os.path.getsize(self.path)

    @property
    def readable_size(self) -> str:
        """Human readable string for the ontology size."""
        return fileutils.human_readable_size(self.path)

    def __init__(self, path: str, syntax: Syntax):
        self.path = path
        self.syntax = syntax

    def convert(self, target: Ontology) -> ConversionResult:
        """Converts the ontology into the specified target ontology."""
        if os.path.isfile(target.path):
            return Ontology.ConversionResult.ALREADY_CONVERTED

        if owltool.convert(self.path, target.path, target.syntax.value):
            return Ontology.ConversionResult.SUCCESS
        else:
            return Ontology.ConversionResult.ERROR
