import os

from evowluator.pyutils import exc, fileutils


class Ontology:
    """Models ontology files."""

    class Syntax:
        """OWL ontology syntax namespace."""
        FUNCTIONAL = 'functional'
        MANCHESTER = 'manchester'
        OWLXML = 'owlxml'
        RDFXML = 'rdfxml'

        ALL = [FUNCTIONAL, MANCHESTER, OWLXML, RDFXML]

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

    def __init__(self, path: str, syntax: str):
        exc.raise_if_not_found(path, file_type=exc.FileType.FILE)
        self.path = path
        self.syntax = syntax
