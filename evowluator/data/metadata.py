from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from pyutils.io import file
from pyutils.io.pretty_printer import PrettyPrinter

from . import json
from .dataset import Dataset, DatasetEntry
from ..config.paths import Paths
from ..util import owltool


def _metadata_path(dataset_name: str, partial: bool = False) -> str:
    ret = os.path.join(Paths.dataset(dataset_name), Paths.METADATA_FILE_NAME)
    if partial:
        ret += '.partial'
    return ret


def _load_metadata(dataset: Dataset, partial: bool = False) -> Dict:
    try:
        metadata_path = _metadata_path(dataset.path, partial) 
        ret = json.load(metadata_path)
        if partial:
            file.remove(metadata_path)
        return ret
    except Exception:
        return {}


def _save_metadata(dataset: Dataset, metadata: Dict, partial: bool = False) -> None:
    json.save(metadata, _metadata_path(dataset.path, partial))


def _compute_metadata(dataset: Dataset) -> Dict:
    log = PrettyPrinter(sys.stdout)
    log.green(f'Computing metadata for "{dataset.name}" dataset...')
    log.spacer(2)

    syntax = dataset.reference_syntax
    metadata = _load_metadata(dataset, partial=True)

    def _get_metadata(entry: DatasetEntry) -> None:
        metadata[entry.name] = owltool.get_metadata(entry.ontology(syntax).path)

    try:
        missing = []

        for entry in dataset.get_entries():
            if entry.name in metadata:
                log.yellow(f'{entry.name}: ', endl=False)
                log('cached')
            else:
                missing.append(entry)

        with ThreadPoolExecutor() as pool:
            try:
                submitted = {pool.submit(_get_metadata, entry): entry for entry in missing}
                for future in as_completed(submitted):
                    entry = submitted[future]
                    if exception := future.exception():
                        raise exception
                    log.yellow(f'{entry.name}: ', endl=False)
                    log('done')
            except:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    except:
        _save_metadata(dataset, metadata, partial=True)
        raise
    else:
        _save_metadata(dataset, metadata)

    log.green('Done!')
    return metadata


def retrieve(dataset: Dataset) -> Dict:
    return _load_metadata(dataset) or _compute_metadata(dataset)


def newer_than(dataset_path: str, path: str) -> bool:
    try:
        return os.path.getmtime(_metadata_path(dataset_path)) > os.path.getmtime(path)
    except Exception:
        return False
