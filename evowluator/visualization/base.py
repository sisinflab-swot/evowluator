from __future__ import annotations

import os
from os import path
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from pyutils.io import fileutils

from evowluator.config import ConfigKey, Paths
from evowluator.data import json
from evowluator.data.dataset import Dataset
from evowluator.data.ontology import Syntax
from evowluator.reasoner.base import ReasoningTask
from evowluator.evaluation.mode import EvaluationMode

from .metric import Metric
from .plot import Figure, MinMaxAvgHistogramPlot, ScatterPlot


class Visualizer:

    # Override

    def configure_plotters(self) -> None:
        pass

    def write_results(self) -> None:
        fileutils.create_dir(self.output_dir)
        avg_res_path = path.join(self.output_dir, 'avg_results.csv')
        self._results.to_csv(avg_res_path, float_format='%.2f')

    # Public

    @classmethod
    def from_dir(cls, results_dir: str, reasoners: Optional[List[str]] = None) -> Visualizer:
        from .correctness import CorrectnessVisualizer
        from .performance import EnergyVisualizer, PerformanceVisualizer

        cfg = json.load(os.path.join(results_dir, Paths.CONFIG_FILE_NAME))
        eval_name = cfg[ConfigKey.NAME]

        if ReasoningTask.MATCHMAKING.value in eval_name:
            cols = ['Resource', 'Request']
        else:
            cols = ['Ontology']

        if reasoners:
            reasoner_cfg = {r[ConfigKey.NAME]: r for r in cfg[ConfigKey.REASONERS]}
            cfg[ConfigKey.REASONERS] = [reasoner_cfg[r] for r in reasoners]

        if EvaluationMode.CORRECTNESS.value in eval_name:
            return CorrectnessVisualizer(results_dir, cfg, index_columns=cols)
        elif EvaluationMode.PERFORMANCE.value in eval_name:
            return PerformanceVisualizer(results_dir, cfg, index_columns=cols)
        elif EvaluationMode.ENERGY.value in eval_name:
            return EnergyVisualizer(results_dir, cfg, index_columns=cols)
        else:
            raise NotImplementedError('Visualizer not implemented for "{}"'.format(eval_name))

    @property
    def results_path(self) -> str:
        return os.path.join(self._results_dir, Paths.RESULTS_FILE_NAME)

    @property
    def output_dir(self) -> str:
        return path.join(self._results_dir, 'visualization')

    @property
    def config_path(self) -> str:
        return os.path.join(self._results_dir, Paths.CONFIG_FILE_NAME)

    def __init__(self, results_dir: str, cfg, index_columns: List[str] = None,
                 non_numeric_columns: Union[bool, List[str]] = False) -> None:
        self._results_dir = results_dir
        self._index_columns = index_columns if index_columns else ['Ontology']
        self._dataset_name = cfg[ConfigKey.DATASET]

        self._syntaxes_by_reasoner: Dict[str, Syntax] = {
            r[ConfigKey.NAME]: Syntax(r[ConfigKey.SYNTAX])
            for r in cfg[ConfigKey.REASONERS]
        }

        self._reasoners: List[str] = list(self._syntaxes_by_reasoner.keys())
        self._results: pd.DataFrame = self.load_results(non_numeric_columns)

        self.colors: Dict[str, str] = {}
        self.markers: Dict[str, str] = {}
        self.figure = Figure()

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

    def plot_results(self, gui: bool = True, plots: Optional[List[int]] = None) -> None:
        self.configure_plotters()
        self.figure.draw(plots=plots)
        self.figure.save(path.join(self.output_dir, 'figure.pdf'))

        if gui:
            self.figure.show()

    def load_results(self, non_numeric_columns: Union[bool, List[str]] = False) -> pd.DataFrame:
        res = pd.read_csv(self.results_path, index_col=self._index_columns)

        columns = [c for c in res.columns if c.split(':', maxsplit=1)[0] in self._reasoners]
        res = res[columns]

        if non_numeric_columns:
            if isinstance(non_numeric_columns, list):
                columns = [c for c in columns if c not in non_numeric_columns]
            else:
                columns = []
        else:
            columns = columns

        if columns:
            res[columns] = res[columns].apply(pd.to_numeric, errors='coerce')
            res[columns] = res[columns].replace(0, np.nan)
            res.dropna(inplace=True)

        if not res.index.is_unique:
            res = res.groupby(res.index).mean()

        if len(self._index_columns) > 1:
            res.index = pd.MultiIndex.from_tuples(res.index, names=self._index_columns)

        return res

    def set_colors(self, colors: List[str]) -> None:
        self.colors = {
            self._reasoners[idx]: color
            for idx, color in enumerate(colors[:len(self._reasoners)]) if color != 'auto'
        }

    def set_markers(self, markers: List[str]) -> None:
        self.markers = {
            self._reasoners[idx]: marker
            for idx, marker in enumerate(markers[:len(self._reasoners)]) if marker != 'auto'
        }

    def add_plotter(self, plot_type: type, **kwargs) -> None:
        kwargs['colors'] = self.colors
        kwargs['markers'] = self.markers
        self.figure.add_plotter(plot_type, **kwargs)

    def add_scatter_plotter(self, metric: Metric,
                            col_filter: Optional[Callable[[str], bool]] = None) -> None:
        dataset = Dataset(os.path.join(Paths.DATA_DIR, self._dataset_name))

        xscale, xunit = fileutils.human_readable_scale_and_unit(dataset.get_max_ontology_size())
        xmetric = Metric('ontology size', xunit, '.2f')

        data = []

        for reasoner in self._reasoners:
            ontologies = dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                sort_by_size=True)
            results = self.results_for_reasoner(reasoner, col_filter=col_filter)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            ontologies = [o for o in ontologies if o.name in results.index]

            x = [o.size / xscale for o in ontologies]
            y = [results.loc[o.name].sum() for o in ontologies]

            data.append((x, y))

        data = dict(zip(self._reasoners, data))
        self.add_plotter(ScatterPlot, data=data, xmetric=xmetric, ymetric=metric)

    def add_min_max_avg_plotter(self, data: pd.DataFrame, metric: Metric,
                                col_filter: Optional[Callable[[str], bool]] = None) -> None:
        if col_filter:
            cols = [c for c in data.columns if col_filter(c)]
            data = data[cols]

        reasoners = data.index.values

        data = [data.loc[r].values for r in reasoners]
        data = dict(zip(reasoners, data))
        self.add_plotter(MinMaxAvgHistogramPlot, data=data, metric=metric)
