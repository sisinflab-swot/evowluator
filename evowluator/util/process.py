from __future__ import annotations

import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from pyutils import exc
from pyutils.io import echo
from ..config import ConfigKey, Paths
from ..data import csv, json
from ..visualization.correctness import CorrectnessStrategy, Status


def process(input_dirs: Iterable[str], correctness_dir: str | None,
            correctness_strategy: str | None, dataset: str | None = None) -> None:
    out_dir = merge(input_dirs, dataset=dataset)

    if correctness_dir:
        filter_correct(out_dir, correctness_dir, correctness_strategy)

    echo.success(f'Results: ', endl=False)
    echo.info(out_dir)


def correctness_results(input_dir: str, strategy: str | None) -> pd.DataFrame:
    df = csv.read(os.path.join(input_dir, Paths.RESULTS_FILE_NAME))
    df.rename(lambda x: x.split(':')[0], axis=1, inplace=True)
    return CorrectnessStrategy.with_name(strategy, list(df.columns)).evaluate_dataframe(df)


def filter_correct(input_dir: str, correctness_dir: str, strategy: str) -> None:
    csv_path = os.path.join(input_dir, Paths.RESULTS_FILE_NAME)
    df = csv.read(csv_path)
    cr = correctness_results(correctness_dir, strategy)
    reasoners = list(cr.columns)

    for col in df.columns:
        reasoner = col.split(':')[0]
        cr[col] = cr[reasoner]

    cr.drop(reasoners, axis=1, inplace=True)
    cr.replace('y', np.nan, inplace=True)
    cr.replace('n', 'incorrect', inplace=True)
    df.update(cr)

    csv.write(df, csv_path)


def incorrect_ontologies(input_dir: str, strategy: str | None) -> Dict[str, List[str]]:
    df = correctness_results(input_dir, strategy)
    incorrect = {}

    for reasoner in df.columns:
        res = df[reasoner]
        res.drop(res[res == Status.OK].index, inplace=True)
        incorrect[reasoner] = list(res.to_dict().keys())

    return incorrect


def merge(input_dirs: Iterable[str], dataset: str | None = None) -> str:
    out_results: pd.DataFrame | None = None
    out_config: Dict | None = None

    for input_dir in input_dirs:
        input_dir = Paths.absolute(input_dir, Paths.RESULTS_DIR)
        out_results = merge_results(out_results, input_dir)
        out_config = merge_configs(out_config, input_dir, dataset)

    output_dir = Paths.new_results_dir(f'{out_config[ConfigKey.TASK]} {out_config[ConfigKey.MODE]}')
    write_csv(out_results, os.path.join(output_dir, Paths.RESULTS_FILE_NAME))
    json.save(out_config, os.path.join(output_dir, Paths.CONFIG_FILE_NAME))

    return output_dir


def merge_configs(config: Dict, input_dir: str, dataset: str | None) -> Dict:
    cur = json.load(os.path.join(input_dir, Paths.CONFIG_FILE_NAME))

    if config is None:
        return cur

    for key in (ConfigKey.TASK, ConfigKey.MODE, ConfigKey.FIELDS, ConfigKey.ITERATIONS):
        try:
            if config[key] != cur[key]:
                raise ValueError(f'Cannot merge "{input_dir}", different values for key "{key}".')
        except KeyError as e:
            exc.re_raise_new_message(e, f'Cannot merge "{input_dir}, missing key "{key}".')

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
    df = csv.read(path).convert_dtypes()
    df['seq'] = df.groupby(df.index.names).cumcount()
    df.set_index('seq', append=True, inplace=True)
    return df


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.reset_index(inplace=True)
    df.drop('seq', axis=1, inplace=True)
    df.fillna(Status.UNKNOWN, inplace=True)
    csv.write(df, path, index=False)


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
