import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from math import ceil
from os import path
from typing import Callable, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from evowluator.config import ConfigKey, Paths
from evowluator.data import json
from evowluator.pyutils import fileutils
from evowluator.test.enum import TestName
from . import plotutils


class Evaluator:
    __metaclass__ = ABCMeta

    # Abstract

    @abstractmethod
    def _write_results(self) -> None:
        pass

    # Public

    @classmethod
    def from_dir(cls, test_dir: str) -> 'Evaluator':
        from .performance import EnergyEvaluator, MemoryEvaluator, TimeEvaluator

        cfg = json.load(os.path.join(test_dir, Paths.CONFIG_FILE_NAME))
        test_name = cfg[ConfigKey.NAME]

        cols = ['Resource', 'Request'] if test_name in TestName.Matchmaking.ALL else ['Ontology']

        if test_name in TestName.TIME:
            return TimeEvaluator(test_dir, cfg, index_columns=cols)
        elif test_name in TestName.MEMORY:
            return MemoryEvaluator(test_dir, cfg, index_columns=cols)
        elif test_name in TestName.ENERGY:
            return EnergyEvaluator(test_dir, cfg, index_columns=cols)
        else:
            raise NotImplementedError('Evaluator not implemented for test "{}"'.format(test_name))

    @property
    def plotters(self) -> List[Callable[[plt.Axes], None]]:
        return []

    @property
    def results_path(self) -> str:
        return os.path.join(self.test_dir, Paths.RESULTS_FILE_NAME)

    @property
    def evaluation_dir(self) -> str:
        return path.join(self.test_dir, 'evaluation')

    @property
    def config_path(self) -> str:
        return os.path.join(self.test_dir, Paths.CONFIG_FILE_NAME)

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        self.test_dir = test_dir
        self.index_columns = index_columns if index_columns else ['Ontology']
        self.dataset_name = cfg[ConfigKey.DATASET]

        self._syntaxes_by_reasoner: 'OrderedDict[str, str]' = OrderedDict(
            (r[ConfigKey.NAME], r[ConfigKey.SYNTAX]) for r in cfg[ConfigKey.REASONERS]
        )

        results = pd.read_csv(self.results_path, index_col=self.index_columns)
        results = results.groupby(results.index).mean()

        if len(self.index_columns) > 1:
            results.index = pd.MultiIndex.from_tuples(results.index, names=self.index_columns)

        self._results: pd.DataFrame = results

    def reasoners(self) -> Iterable[str]:
        return self._syntaxes_by_reasoner.keys()

    def ontologies(self) -> Iterable[str]:
        return self._results.index.values

    def results_for_reasoner(self, reasoner: str) -> pd.DataFrame:
        needle = reasoner + ':'
        return self._results[[f for f in self._results if f.startswith(needle)]]

    def results_for_ontology(self, ontology: str) -> pd.DataFrame:
        return self._results.loc[ontology]

    def results_grouped_by_reasoner(self, columns: List[str] = None):
        results = self._results[columns] if columns else self._results
        return results.groupby(lambda x: x.split(':', maxsplit=1)[0], axis=1)

    def write_results(self) -> None:
        fileutils.create_dir(self.evaluation_dir)

        # Average results
        avg_res_path = path.join(self.evaluation_dir, 'avg_results.csv')
        self._results.to_csv(avg_res_path, float_format='%.2f')

        self._write_results()

    def plot_results(self) -> None:
        plotters = self.plotters
        n_plotters = len(plotters)

        nrows = 2 if n_plotters > 1 else 1
        ncols = ceil(n_plotters / nrows)

        fig, ax = plotutils.setup_plot(nrows=nrows, ncols=ncols)
        axes = ax.flatten()

        for i, plotter in enumerate(plotters):
            plotter(axes[i])

        plt.show()
