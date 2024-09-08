from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator

from pyutils.io import file
from pyutils.io.pretty_printer import PrettyPrinter
from .dataset import Dataset, DatasetEntry, Ontology, Syntax
from ..reasoner.base import ReasoningTask
from ..util import owltool


def convert_ontology(source: Ontology, target: Ontology) -> None:
    """Converts the ontology into the specified target ontology."""
    owltool.convert(source.path, target.path, target.syntax)


def _all_entries(dataset: Dataset) -> Iterator[DatasetEntry]:
    for entry in dataset.get_entries():
        yield entry
        for i_entry in (e for t in ReasoningTask.all() for e in entry.inputs_for_task(t)):
            yield i_entry


def convert_dataset(dataset: Dataset, syntax: Syntax) -> None:
    """Converts a dataset into the specified syntax."""
    def _convert_entry(lentry: DatasetEntry) -> bool:
        target_ontology = lentry.ontology(syntax)
        file.create_dir(os.path.dirname(target_ontology.path))
        convert_ontology(lentry.ontology(dataset.reference_syntax), target_ontology)

    log = PrettyPrinter(sys.stdout)
    log.green((f'Starting conversion of "{dataset.name}" dataset '
               f'({dataset.count()} ontologies) in {syntax} syntax...'))
    log.spacer(2)

    incomplete: Dict[DatasetEntry, None] = {}  # Use dict to preserve ordering.

    for entry in _all_entries(dataset):
        if os.path.isfile(entry.ontology(syntax).path):
            log.yellow(f'{entry.name}: ', endl=False)
            log('already converted')
        else:
            incomplete[entry] = None

    with ThreadPoolExecutor() as pool:
        try:
            submitted = {pool.submit(_convert_entry, entry): entry for entry in incomplete}
            for future in as_completed(submitted):
                entry = submitted[future]
                if exception := future.exception():
                    raise exception
                incomplete.pop(entry)
                log.yellow(f'{entry.name}: ', endl=False)
                log('converted')
        except:
            pool.shutdown(wait=False, cancel_futures=True)
            for entry in incomplete:
                file.remove(entry.ontology(syntax).path)
            raise

    log.green('Done!')
