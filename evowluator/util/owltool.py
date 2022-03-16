import os
from typing import List

from pyutils import exc
from pyutils.proc.task import OutputAction, Task
from ..config import OWLTool


def _spawn_owltool(args: List[str]) -> None:
    exc.raise_if_not_found(OWLTool.PATH, file_type=exc.FileType.FILE)
    Task.jar(OWLTool.PATH, jar_args=args, jvm_opts=OWLTool.VM_OPTS,
             output_action=OutputAction.DISCARD).run().raise_if_failed()


def convert(source_path: str, target_path: str, target_syntax: str) -> None:
    """Converts the source ontology into the specified target ontology."""
    try:
        _spawn_owltool(['convert', '-i', source_path, '-o', target_path, '-f', target_syntax])
    except Exception as e:
        exc.re_raise_new_message(e, f'Failed to convert ontology "{os.path.basename(source_path)}"')


def print_taxonomy(onto_path: str, output_path: str) -> None:
    """Prints the taxonomy starting from the top concept."""
    if onto_path == output_path:
        onto_path = os.path.splitext(onto_path)[0]
        os.rename(output_path, onto_path)
    _spawn_owltool(['taxonomy', '-i', onto_path, '-o', output_path])
