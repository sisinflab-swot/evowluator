import os
from enum import Enum

from evowluator.pyutils import fileutils, proc


class Ontology:
    """Models ontology files."""

    class Syntax:
        """OWL ontology syntax namespace."""
        FUNCTIONAL = 'functional'
        MANCHESTER = 'manchester'
        OWLXML = 'owlxml'
        RDFXML = 'rdfxml'

        ALL = [FUNCTIONAL, MANCHESTER, OWLXML, RDFXML]

    class ConversionResult(Enum):
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

    def __init__(self, path: str, syntax: str):
        self.path = path
        self.syntax = syntax

    def convert(self, target: 'Ontology') -> ConversionResult:
        """Converts the ontology into the specified target ontology."""
        if os.path.isfile(target.path):
            return Ontology.ConversionResult.ALREADY_CONVERTED

        robot_exe = os.path.realpath(proc.find_executable('robot'))
        robot_jar = os.path.join(os.path.dirname(robot_exe), 'robot.jar')

        args = [
            'convert',
            '-i', self.path,
            '-o', target.path,
            '-f', {
                Ontology.Syntax.FUNCTIONAL: 'ofn',
                Ontology.Syntax.MANCHESTER: 'omn',
                Ontology.Syntax.OWLXML: 'owx',
                Ontology.Syntax.RDFXML: 'owl'
            }[target.syntax]
        ]

        task = proc.Jar.spawn(robot_jar, jar_args=args)

        if task.exit_code == 0:
            return Ontology.ConversionResult.SUCCESS
        else:
            return Ontology.ConversionResult.ERROR
