from __future__ import annotations

import os

from pyutils.io import echo, fileutils
from .dataset import Dataset, Ontology, Syntax
from ..reasoner.base import ReasoningTask
from ..util import owltool
from ..util.strenum import StrEnum


class ConversionResult(StrEnum):
    """Ontology conversion result."""
    SUCCESS = 'done'
    ALREADY_CONVERTED = 'already converted'
    ERROR = 'error'


def convert_ontology(source: Ontology, target: Ontology) -> ConversionResult:
    """Converts the ontology into the specified target ontology."""
    if os.path.isfile(target.path):
        return ConversionResult.ALREADY_CONVERTED

    if owltool.convert(source.path, target.path, target.syntax.value):
        return ConversionResult.SUCCESS
    else:
        return ConversionResult.ERROR


def convert_dataset(dataset: Dataset, syntax: Syntax, source_syntax: Syntax | None = None) -> None:
    """Converts a dataset into the specified syntax."""
    echo.pretty((f'Starting conversion of "{dataset.name}" dataset '
                 f'({dataset.count()} ontologies) in {syntax} syntax...\n'),
                color=echo.Color.GREEN)

    if not source_syntax:
        source_syntax = next(s for s in dataset.syntaxes if s != syntax)

    for entry in dataset.get_entries():
        target_ontology = entry.ontology(syntax)
        echo.pretty(f'{target_ontology.name}: ', color=echo.Color.YELLOW, endl=False)
        fileutils.create_dir(os.path.dirname(target_ontology.path))
        result = convert_ontology(entry.ontology(source_syntax), target_ontology)
        _print_conversion_result(result)

        for i_entry in (e for t in ReasoningTask.all() for e in entry.inputs_for_task(t)):
            target = i_entry.ontology(syntax)
            echo.pretty(f'    {target.name}: ', color=echo.Color.YELLOW, endl=False)
            fileutils.create_dir(os.path.dirname(target.path))
            result = convert_ontology(i_entry.ontology(source_syntax), target)
            _print_conversion_result(result)

    echo.pretty('Done!', color=echo.Color.GREEN)


def _print_conversion_result(result: ConversionResult) -> None:
    if result in [ConversionResult.SUCCESS, ConversionResult.ALREADY_CONVERTED]:
        echo.info(result.value)
    else:
        echo.error(result.value)
