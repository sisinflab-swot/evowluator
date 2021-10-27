from __future__ import annotations

import os
from typing import Dict, Iterable

import pandas as pd

from pyutils.io import echo
from ..config import ConfigKey, Paths
from ..data import json
from ..visualization.base import infer_index
from ..visualization.correctness import Status


def merge(input_dirs: Iterable[str], dataset: str | None = None) -> None:
    out_results: pd.DataFrame | None = None
    out_config: Dict | None = None

    for input_dir in input_dirs:
        input_dir = Paths.absolute(input_dir, Paths.RESULTS_DIR)
        out_results = merge_results(out_results, input_dir)
        out_config = merge_configs(out_config, input_dir, dataset)

    output_dir = Paths.new_results_dir(out_config[ConfigKey.NAME])
    write_csv(out_results, os.path.join(output_dir, Paths.RESULTS_FILE_NAME))
    json.save(out_config, os.path.join(output_dir, Paths.CONFIG_FILE_NAME))

    echo.success(f'Merge results: ', endl=False)
    echo.info(output_dir)


def merge_configs(config: Dict, input_dir: str, dataset: str | None) -> Dict:
    cur = json.load(os.path.join(input_dir, Paths.CONFIG_FILE_NAME))

    if config is None:
        return cur

    if config[ConfigKey.NAME] != cur[ConfigKey.NAME]:
        raise ValueError(f'Cannot merge "{config[ConfigKey.NAME]}" and "{cur[ConfigKey.NAME]}".')

    # Merge reasoners

    reasoners: Dict = {e[ConfigKey.NAME]: e[ConfigKey.SYNTAX] for e in config[ConfigKey.REASONERS]}
    cur_reasoners = ((e[ConfigKey.NAME], e[ConfigKey.SYNTAX]) for e in cur[ConfigKey.REASONERS])

    for r, s in cur_reasoners:
        if r not in reasoners:
            reasoners[r] = s
        elif reasoners[r] != s:
            raise ValueError(f'Cannot merge, syntaxes differ for "{r}".')

    config[ConfigKey.REASONERS] = [{ConfigKey.NAME: r, ConfigKey.SYNTAX: s}
                                   for r, s in reasoners.items()]

    # Merge datasets

    if not dataset:
        dataset = config[ConfigKey.DATASET][ConfigKey.NAME]
        cur_dataset = cur[ConfigKey.DATASET][ConfigKey.NAME]

        if dataset != cur_dataset:
            dataset = f'{dataset}+{cur_dataset}'

    config[ConfigKey.DATASET][ConfigKey.NAME] = dataset

    ontos: Dict = {e[ConfigKey.NAME]: e[ConfigKey.SIZE]
                   for e in config[ConfigKey.DATASET][ConfigKey.ONTOLOGIES]}
    cur_ontos = ((e[ConfigKey.NAME], e[ConfigKey.SIZE])
                 for e in cur[ConfigKey.DATASET][ConfigKey.ONTOLOGIES])

    for n, s in cur_ontos:
        if n in ontos:
            ontos[n].update(s)
        else:
            ontos[n] = s

    config[ConfigKey.DATASET][ConfigKey.ONTOLOGIES] = [{ConfigKey.NAME: n, ConfigKey.SIZE: o}
                                                       for n, o in ontos.items()]

    return config


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).convert_dtypes()
    index = infer_index(df.columns)
    df['seq'] = df.groupby(index).cumcount()
    index.append('seq')
    df.set_index(index, inplace=True)
    return df


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.reset_index(inplace=True)
    df.drop('seq', axis=1, inplace=True)
    df.fillna(Status.UNKNOWN, inplace=True)
    df.to_csv(path, float_format='%.2f', index=False)


def merge_results(results: pd.DataFrame | None, input_dir: str) -> pd.DataFrame:
    cur = read_csv(os.path.join(input_dir, Paths.RESULTS_FILE_NAME))

    if results is None:
        results = cur
    else:
        columns = list(dict.fromkeys(list(results.columns) + list(cur.columns)))
        results = results.reindex(columns=columns, copy=False)
        cur = cur.reindex(columns=columns, copy=False)

        results.update(cur)
        cur.drop(results.index, inplace=True, errors='ignore')
        results = pd.concat([results, cur], copy=False)

    return results
