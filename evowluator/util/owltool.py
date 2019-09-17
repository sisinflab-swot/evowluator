from typing import List

from pyutils import exc
from pyutils.proc.task import Jar, OutputAction

from evowluator.config import OWLTool


def _spawn_owltool(args: List[str]) -> bool:
    exc.raise_if_not_found(OWLTool.PATH, file_type=exc.FileType.FILE)
    return Jar.spawn(OWLTool.PATH, jar_args=args, vm_opts=OWLTool.VM_OPTS,
                     output_action=OutputAction.DISCARD).exit_code == 0


def convert(source_path: str, target_path: str, target_syntax: str) -> bool:
    """Converts the source ontology into the specified target ontology."""
    return _spawn_owltool(['convert', '-i', source_path, '-o', target_path, '-f', target_syntax])


def print_tbox(onto_path: str, output_path: str) -> bool:
    """Prints the TBox starting from the top concept."""
    return _spawn_owltool(['print-tbox', '-i', onto_path, '-o', output_path])
