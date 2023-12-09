import os
from typing import List

from pyutils import exc
from pyutils.proc.task import OutputAction, Task
from ..config import Paths


def _spawn_owltool(args: List[str]) -> None:
    exc.raise_if_not_found(Paths.OWLTOOL, file_type=exc.FileType.FILE)
    vm_opts = ['-Xmx32g', '-DentityExpansionLimit=1000000000']
    Task.jar(Paths.OWLTOOL, jar_args=args, jvm_opts=vm_opts,
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
