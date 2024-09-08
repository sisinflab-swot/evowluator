from __future__ import annotations

from typing import List

from pyutils.types.strenum import StrEnum


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

    @classmethod
    def priority(cls) -> List[Syntax]:
        """Returns the syntaxes in order of priority."""
        top = [Syntax.FUNCTIONAL, Syntax.OWLXML, Syntax.TURTLE, Syntax.RDFXML]
        return top + [s for s in cls.all() if s not in top]
