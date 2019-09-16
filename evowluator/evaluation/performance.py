from abc import ABC, abstractmethod
from os import path
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import plotutils
from .evaluator import Evaluator
from .metric import Metric


class SingleValueEvaluator(Evaluator, ABC):

    # Abstract

    @property
    @abstractmethod
    def metric(self) -> Metric:
        pass

    # Overrides

    @property
    def plotters(self):
        return super().plotters + [self._histogram_plotter, self._scatter_plotter]

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(test_dir, cfg, index_columns)
        self._global_stats: Optional[pd.DataFrame] = None

    def write_results(self):
        super().write_results()
        self._write_global_stats(path.join(self.evaluation_dir, 'global_stats.csv'))

    # Private

    def _write_global_stats(self, file_path: str) -> None:
        results = np.asarray([self.results_for_reasoner(r).values for r in self.reasoners()])
        metric_str = self.metric.to_string()

        self._global_stats = pd.DataFrame({
            'Reasoner': list(self.reasoners()),
            'Min ' + metric_str: results.min(axis=1).flatten(),
            'Avg ' + metric_str: results.mean(axis=1).flatten(),
            'Max ' + metric_str: results.max(axis=1).flatten()
        }).set_index('Reasoner')

        self._global_stats.to_csv(file_path, float_format='%.2f')

    def _histogram_plotter(self, ax: plt.Axes) -> None:
        self.draw_min_max_avg(ax, self._global_stats, self.metric)

    def _scatter_plotter(self, ax: plt.Axes) -> None:
        self.draw_scatter(ax, self.metric)


class EnergyEvaluator(SingleValueEvaluator):

    @property
    def metric(self):
        return Metric('energy', None, '.0f')


class PerformanceEvaluator(Evaluator):

    # Overrides

    @property
    def plotters(self):
        return super().plotters + [self.__time_histogram_plotter, self.__time_scatter_plotter,
                                   self.__memory_histogram_plotter, self.__memory_scatter_plotter]

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(test_dir, cfg, index_columns)
        self._results[self.__memory_cols()] /= (1024 * 1024)
        self._global_stats: Optional[pd.DataFrame] = None
        self._time_unit: str = 'ms'

    def write_results(self):
        super().write_results()
        self.__write_total_times(path.join(self.evaluation_dir, 'total_times.csv'))
        self.__write_memory(path.join(self.evaluation_dir, 'memory.csv'))
        self.__write_global_stats(path.join(self.evaluation_dir, 'global_stats.csv'))

    # Private

    def __memory_cols(self) -> List:
        return [c for c in self._results.columns if 'memory' in c.lower()]

    def __parsing_cols(self) -> List:
        return [c for c in self._results.columns if 'parsing' in c.lower()]

    def __reasoning_cols(self) -> List:
        other_cols = self.__memory_cols() + self.__parsing_cols()
        return [c for c in self._results.columns if c not in other_cols]

    def __write_total_times(self, file_path: str) -> None:
        cols = self.__memory_cols()
        cols = [c for c in self._results.columns if c not in cols]
        totals = self.results_grouped_by_reasoner(cols).sum()
        totals.to_csv(file_path, float_format='%.2f')

    def __write_memory(self, file_path: str) -> None:
        totals = self.results_grouped_by_reasoner(self.__memory_cols()).sum()
        totals.to_csv(file_path, float_format='%.2f')

    def __write_global_stats(self, file_path: str) -> None:
        reasoners = list(self.reasoners())

        parsing_cols = self.__parsing_cols()
        reasoning_cols = self.__reasoning_cols()
        memory_cols = self.__memory_cols()

        parsing = self.results_grouped_by_reasoner(parsing_cols).sum().sum()
        reasoning = self.results_grouped_by_reasoner(reasoning_cols).sum().sum()

        memory = self.results_grouped_by_reasoner(memory_cols).sum()
        memory_min, memory_mean, memory_max = memory.min(), memory.mean(), memory.max()

        parsing = np.asarray([parsing[r] for r in reasoners])
        reasoning = np.asarray([reasoning[r] for r in reasoners])
        memory_min = np.asarray([memory_min[r] for r in reasoners])
        memory_mean = np.asarray([memory_mean[r] for r in reasoners])
        memory_max = np.asarray([memory_max[r] for r in reasoners])

        if np.append(parsing, reasoning).min() < 1000.0:
            time_unit = 'ms'
        else:
            parsing /= 1000.0
            reasoning /= 1000.0
            time_unit = 's'

        data = pd.DataFrame({
            'Reasoner': reasoners,
            'Total parsing time ({})'.format(time_unit): parsing,
            'Total reasoning time ({})'.format(time_unit): reasoning,
            'Total time ({})'.format(time_unit): parsing + reasoning,
            'Min memory peak (MiB)': memory_min,
            'Avg memory peak (MiB)': memory_mean,
            'Max memory peak (MiB)': memory_max
        }).set_index('Reasoner')

        data.to_csv(file_path, float_format='%.2f')

        self._global_stats = data
        self._time_unit = time_unit

    def __time_histogram_plotter(self, ax: plt.Axes) -> None:
        data = self._global_stats.iloc[:, :2]
        reasoners = list(data.index.values)

        values = data.values.transpose()
        data = dict(zip(['Parsing', 'Reasoning'], list(values)))

        metric = Metric('time', self._time_unit, '.0f')
        plotutils.draw_grouped_histograms(ax, data, metric, reasoners)
        ax.set_title('Total parsing and reasoning time')

    def __memory_histogram_plotter(self, ax: plt.Axes) -> None:
        metric = Metric('memory peak', 'MiB', '.2f')
        self.draw_min_max_avg(ax, self._global_stats, metric,
                              col_filter=lambda c: metric.name in c)

    def __time_scatter_plotter(self, ax: plt.Axes) -> None:
        metric = Metric('time', self._time_unit, '.0f')
        self.draw_scatter(ax, metric, lambda c: 'memory' not in c)

    def __memory_scatter_plotter(self, ax: plt.Axes) -> None:
        metric = Metric('memory peak', 'MiB', '.2f')
        self.draw_scatter(ax, metric, lambda c: 'memory' in c)
