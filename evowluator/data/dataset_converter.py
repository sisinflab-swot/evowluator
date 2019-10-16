import os

from pyutils.io import echo, fileutils

from evowluator.data.ontology import Ontology, Syntax
from .dataset import Dataset


def convert(dataset: Dataset, syntax: Syntax) -> None:
    echo.pretty(('Starting conversion of "{}" dataset '
                 '({} ontologies) in {} syntax...\n'.format(dataset.name, dataset.size, syntax)),
                color=echo.Color.GREEN)

    source_syntax = next(s for s in dataset.syntaxes if s != syntax)

    for entry in dataset.get_entries():
        target_ontology = entry.ontology(syntax)

        echo.pretty('{}: '.format(target_ontology.name),
                    color=echo.Color.YELLOW, endl=False)

        fileutils.create_dir(os.path.dirname(target_ontology.path))
        result = entry.ontology(source_syntax).convert(target_ontology)

        _print_conversion_result(result)

        for request in entry.requests(source_syntax):
            target_request = request.ontology(syntax)

            echo.pretty('    {}: '.format(target_request.name),
                        color=echo.Color.YELLOW, endl=False)

            fileutils.create_dir(os.path.dirname(target_request.path))
            result = request.ontology(source_syntax).convert(target_request)

            _print_conversion_result(result)

        echo.info('')

    echo.pretty('Done!', color=echo.Color.GREEN)


def _print_conversion_result(result: Ontology.ConversionResult) -> None:
    if result in [Ontology.ConversionResult.SUCCESS, Ontology.ConversionResult.ALREADY_CONVERTED]:
        echo.info(result.value)
    else:
        echo.error(result.value)
