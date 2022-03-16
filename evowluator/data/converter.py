from __future__ import annotations

import os
import sys

from pyutils.io import file
from pyutils.io.pretty_printer import PrettyPrinter
from .dataset import Dataset, DatasetEntry, Ontology, Syntax
from ..reasoner.base import ReasoningTask
from ..util import owltool


def convert_ontology(source: Ontology, target: Ontology) -> bool:
    """Converts the ontology into the specified target ontology."""
    if os.path.isfile(target.path):
        return False
    owltool.convert(source.path, target.path, target.syntax)
    return True


def convert_dataset(dataset: Dataset, syntax: Syntax, source_syntax: Syntax | None = None) -> None:
    """Converts a dataset into the specified syntax."""
    def _convert_entry(lentry: DatasetEntry) -> None:
        try:
            target_ontology = lentry.ontology(syntax)
            log.yellow(f'{target_ontology.name}: ', endl=False)
            file.create_dir(os.path.dirname(target_ontology.path))

            if convert_ontology(entry.ontology(source_syntax), target_ontology):
                log('converted')
            else:
                log('already converted')
        except Exception:
            log.red('error')

    log = PrettyPrinter(sys.stdout)
    log.green((f'Starting conversion of "{dataset.name}" dataset '
               f'({dataset.count()} ontologies) in {syntax} syntax...'))
    log.spacer(2)

    if not source_syntax:
        source_syntax = next(s for s in dataset.syntaxes if s != syntax)

    for entry in dataset.get_entries():
        _convert_entry(entry)
        with log.indent_context():
            for i_entry in (e for t in ReasoningTask.all() for e in entry.inputs_for_task(t)):
                _convert_entry(i_entry)

    log.green('Done!')
