import os
from collections import OrderedDict
from math import ceil
from os import path
from typing import Callable, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyutils.io import fileutils

from evowluator.config import ConfigKey, Paths
from evowluator.data import json
from evowluator.data.dataset import Dataset
from evowluator.reasoner.base import ReasoningTask
from evowluator.test.test_mode import TestMode
from . import plotutils
from .metric import Metric


class Evaluator:

    # Override

    def write_results(self) -> None:
        fileutils.create_dir(self.evaluation_dir)
        avg_res_path = path.join(self.evaluation_dir, 'avg_results.csv')
        self._results.to_csv(avg_res_path, float_format='%.2f')

    @property
    def plotters(self) -> List[Callable[[plt.Axes], None]]:
        return []

    # Public

    @classmethod
    def from_dir(cls, test_dir: str) -> 'Evaluator':
        from .correctness import CorrectnessEvaluator
        from .performance import EnergyEvaluator, PerformanceEvaluator

        cfg = json.load(os.path.join(test_dir, Paths.CONFIG_FILE_NAME))
        test_name = cfg[ConfigKey.NAME]

        cols = ['Resource', 'Request'] if ReasoningTask.MATCHMAKING in test_name else ['Ontology']

        if TestMode.CORRECTNESS in test_name:
            return CorrectnessEvaluator(test_dir, cfg, index_columns=cols)
        elif TestMode.PERFORMANCE in test_name:
            return PerformanceEvaluator(test_dir, cfg, index_columns=cols)
        elif TestMode.ENERGY in test_name:
            return EnergyEvaluator(test_dir, cfg, index_columns=cols)
        else:
            raise NotImplementedError('Evaluator not implemented for test "{}"'.format(test_name))

    @property
    def results_path(self) -> str:
        return os.path.join(self.test_dir, Paths.RESULTS_FILE_NAME)

    @property
    def evaluation_dir(self) -> str:
        return path.join(self.test_dir, 'evaluation')

    @property
    def config_path(self) -> str:
        return os.path.join(self.test_dir, Paths.CONFIG_FILE_NAME)

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None,
                 non_numeric_columns: Union[bool, List[str]] = False) -> None:
        self.test_dir = test_dir
        self.index_columns = index_columns if index_columns else ['Ontology']
        self.dataset_name = cfg[ConfigKey.DATASET]

        self._syntaxes_by_reasoner: 'OrderedDict[str, str]' = OrderedDict(
            (r[ConfigKey.NAME], r[ConfigKey.SYNTAX]) for r in cfg[ConfigKey.REASONERS]
        )

        self._results: pd.DataFrame = self.load_results(non_numeric_columns)
        self.reasoners: List[str] = list(self._syntaxes_by_reasoner.keys())

    def ontologies(self) -> Iterable[str]:
        return self._results.index.values

    def results_for_reasoner(self, reasoner: str,
                             col_filter: Optional[Callable[[str], bool]] = None) -> pd.DataFrame:
        needle = reasoner + ':'
        results = self._results[[f for f in self._results if f.startswith(needle)]]
        results = results.rename(lambda s: s.rsplit(':', maxsplit=1)[1].strip(), axis='columns')

        if col_filter:
            results = results[[c for c in results.columns if col_filter(c)]]

        return results

    def results_for_ontology(self, ontology: str) -> pd.DataFrame:
        return self._results.loc[ontology]

    def results_grouped_by_reasoner(self, columns: List[str] = None, drop_missing: bool = True):
        results = self._results[columns] if columns else self._results

        if drop_missing:
            results = results.dropna()

        return results.groupby(lambda x: x.split(':', maxsplit=1)[0], axis=1)

    def plot_results(self, plots: Optional[List[int]] = None) -> None:
        plotters = self.plotters

        if plots:
            plotters = [plotters[i] for i in plots]

        n_plotters = len(plotters)

        if n_plotters == 0:
            return

        nrows = 2 if n_plotters > 1 else 1
        ncols = ceil(n_plotters / nrows)

        fig, ax = plotutils.setup_plot(nrows=nrows, ncols=ncols)
        axes = ax.flatten()

        for i, plotter in enumerate(plotters):
            plotter(axes[i])

        fig.tight_layout()
        plt.show()

    def load_results(self, non_numeric_columns: Union[bool, List[str]] = False) -> pd.DataFrame:
        results = pd.read_csv(self.results_path, index_col=self.index_columns)

        numeric_columns = results.columns.values.tolist()

        if non_numeric_columns:
            if isinstance(non_numeric_columns, list):
                numeric_columns = [c for c in numeric_columns if c not in non_numeric_columns]
            else:
                numeric_columns = []

        if numeric_columns:
            results[numeric_columns] = results[numeric_columns].apply(pd.to_numeric,
                                                                      errors='coerce')
            results[numeric_columns] = results[numeric_columns].replace(0, np.nan)
            results.dropna(inplace=True)

        if not results.index.is_unique:
            results = results.groupby(results.index).mean()

        if len(self.index_columns) > 1:
            results.index = pd.MultiIndex.from_tuples(results.index, names=self.index_columns)

        return results

    def draw_scatter(self, ax: plt.Axes, metric: Metric,
                     col_filter: Optional[Callable[[str], bool]] = None) -> None:
        dataset = Dataset(os.path.join(Paths.DATA_DIR, self.dataset_name))

        xscale, xunit = fileutils.human_readable_scale_and_unit(dataset.get_max_ontology_size())
        xmetric = Metric('ontology size', xunit, '.2f')

        data = []

        for reasoner in self.reasoners:
            ontologies = dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                sort_by_size=True)
            results = self.results_for_reasoner(reasoner, col_filter=col_filter)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            ontologies = [o for o in ontologies if o.name in results.index]

            x = [o.size / xscale for o in ontologies]
            y = [results.loc[o.name].sum() for o in ontologies]

            data.append((x, y))

        plotutils.draw_scatter_plot(ax, dict(zip(self.reasoners, data)), xmetric, metric)

    def draw_min_max_avg(self, ax: plt.Axes, data: pd.DataFrame, metric: Metric,
                         col_filter: Optional[Callable[[str], bool]] = None) -> None:
        if col_filter:
            cols = [c for c in data.columns if col_filter(c)]
            data = data[cols]

        reasoners = data.index.values

        data = [data.loc[r].values for r in reasoners]
        data = dict(zip(reasoners, data))
        plotutils.draw_min_avg_max_histograms(ax, data, metric)
