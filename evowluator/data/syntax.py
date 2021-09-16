from ..util.strenum import StrEnum


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
