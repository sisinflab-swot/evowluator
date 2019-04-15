import os
from abc import ABC, abstractmethod
from os import path
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evowluator.config import Paths
from evowluator.data.dataset import Dataset
from evowluator.pyutils import fileutils
from . import plotutils
from .evaluator import Evaluator


class SingleValueEvaluator(Evaluator, ABC):

    # Abstract

    @property
    @abstractmethod
    def _metric(self) -> str:
        pass

    @property
    @abstractmethod
    def _unit(self) -> str:
        pass

    # Overrides

    @property
    def plotters(self):
        return super().plotters + [self._histogram_plotter, self._scatter_plotter]

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(test_dir, cfg, index_columns)
        self._global_stats: Optional[pd.DataFrame] = None

    def _write_results(self):
        self._write_global_stats(path.join(self.evaluation_dir, 'global_stats.csv'))

    # Private

    def _metric_and_unit(self, capitalized: bool = False) -> str:
        res = '{} ({})'.format(self._metric, self._unit) if self._unit else self._metric
        return res[0].upper() + res[1:] if capitalized else res

    def _write_global_stats(self, file_path: str) -> None:
        results = np.asarray([self.results_for_reasoner(r).values for r in self.reasoners()])

        self._global_stats = pd.DataFrame({
            'Reasoner': list(self.reasoners()),
            'Min ' + self._metric_and_unit(): results.min(axis=1).flatten(),
            'Avg ' + self._metric_and_unit(): results.mean(axis=1).flatten(),
            'Max ' + self._metric_and_unit(): results.max(axis=1).flatten()
        }).set_index('Reasoner')

        self._global_stats.to_csv(file_path, float_format='%.2f')

    def _histogram_plotter(self, ax: plt.Axes) -> None:
        stats = self._global_stats
        reasoners = stats.index.values

        data = [stats.loc[r].values for r in reasoners]
        plotutils.draw_min_avg_max_histograms(ax, dict(zip(reasoners, data)),
                                              self._metric, self._unit)

    def _scatter_plotter(self, ax: plt.Axes) -> None:
        reasoners = list(self.reasoners())
        dataset = Dataset(os.path.join(Paths.DATA_DIR, self.dataset_name))

        scale, unit = fileutils.human_readable_scale_and_unit(dataset.get_max_ontology_size())
        ax.set_title('{} by ontology size'.format(self._metric.capitalize()))
        ax.set_xlabel('Ontology size ({})'.format(unit))
        ax.set_ylabel(self._metric_and_unit(capitalized=True))

        data = []

        for reasoner in reasoners:
            ontologies = dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                sort_by_size=True)
            results = self.results_for_reasoner(reasoner)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            ontologies = [o for o in ontologies if o.name in results.index]

            x = [o.size / scale for o in ontologies]
            y = [results.loc[o.name].values[0] for o in ontologies]

            data.append((x, y))

        plotutils.draw_scatter_plot(ax, dict(zip(reasoners, data)))


class EnergyEvaluator(SingleValueEvaluator):

    @property
    def _metric(self):
        return 'energy'

    @property
    def _unit(self):
        return ''


class PerformanceEvaluator(Evaluator):

    class GlobalStats(NamedTuple):
        data: pd.DataFrame
        time_unit: str

    # Overrides

    @property
    def plotters(self):
        return super().plotters + [self.__time_histogram_plotter, self.__time_scatter_plotter,
                                   self.__memory_histogram_plotter, self.__memory_scatter_plotter]

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(test_dir, cfg, index_columns)
        self._results[self.__memory_cols()] /= (1024 * 1024)
        self._global_stats: Optional[PerformanceEvaluator.GlobalStats] = None

    def _write_results(self):
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
        self._global_stats = self.GlobalStats(data, time_unit)

    def __time_histogram_plotter(self, ax: plt.Axes) -> None:
        data = self._global_stats.data.iloc[:, :2]
        reasoners = data.index.values

        values = [list(data.loc[r].values) for r in reasoners]
        data = dict(zip(reasoners, values))

        ax.set_title('Total parsing and reasoning time')
        ax.set_ylabel('Time ({})'.format(self._global_stats.time_unit))

        plotutils.draw_stacked_histograms(ax, data, ['Parsing', 'Reasoning'])

    def __time_scatter_plotter(self, ax: plt.Axes) -> None:
        reasoners = list(self.reasoners())
        dataset = Dataset(os.path.join(Paths.DATA_DIR, self.dataset_name))

        scale, unit = fileutils.human_readable_scale_and_unit(dataset.get_max_ontology_size())
        ax.set_title('Parsing and reasoning time by ontology size')
        ax.set_xlabel('Ontology size ({})'.format(unit))
        ax.set_ylabel('Time (ms)')

        data = []

        for reasoner in reasoners:
            ontologies = dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                sort_by_size=True)
            results = self.results_for_reasoner(reasoner)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            results = results[[c for c in results.columns if 'memory' not in c.lower()]]
            ontologies = [o for o in ontologies if o.name in results.index]

            x = [o.size / scale for o in ontologies]
            y = [results.loc[o.name].sum() for o in ontologies]

            data.append((x, y))

        plotutils.draw_scatter_plot(ax, dict(zip(reasoners, data)))

    def __memory_histogram_plotter(self, ax: plt.Axes) -> None:
        cols = [c for c in self._global_stats.data.columns if 'memory peak' in c]
        stats = self._global_stats.data[cols]
        reasoners = stats.index.values

        data = [stats.loc[r].values for r in reasoners]
        plotutils.draw_min_avg_max_histograms(ax, dict(zip(reasoners, data)), 'memory peak', 'MiB')

    def __memory_scatter_plotter(self, ax: plt.Axes) -> None:
        reasoners = list(self.reasoners())
        dataset = Dataset(os.path.join(Paths.DATA_DIR, self.dataset_name))

        scale, unit = fileutils.human_readable_scale_and_unit(dataset.get_max_ontology_size())
        ax.set_title('Memory peak by ontology size')
        ax.set_xlabel('Ontology size ({})'.format(unit))
        ax.set_ylabel('Memory peak (MiB)')

        data = []

        for reasoner in reasoners:
            ontologies = dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                sort_by_size=True)
            results = self.results_for_reasoner(reasoner)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            results = results[[c for c in results.columns if 'memory' in c.lower()]]
            ontologies = [o for o in ontologies if o.name in results.index]

            x = [o.size / scale for o in ontologies]
            y = [results.loc[o.name].sum() for o in ontologies]

            data.append((x, y))

        plotutils.draw_scatter_plot(ax, dict(zip(reasoners, data)))
