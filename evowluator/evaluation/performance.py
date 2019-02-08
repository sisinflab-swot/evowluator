import os
from abc import ABCMeta, abstractmethod
from os import path
from typing import List, NamedTuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evowluator.config import Paths
from evowluator.data.dataset import Dataset
from evowluator.pyutils import fileutils
from . import plotutils
from .evaluator import Evaluator


class SingleValueEvaluator(Evaluator):
    __metaclass__ = ABCMeta

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
        self._global_stats: pd.DataFrame = None

    def _write_results(self):
        self._write_global_stats(path.join(self.evaluation_dir, 'global_stats.csv'))

    # Private

    @property
    def _metric_and_unit(self) -> str:
        return '{} ({})'.format(self._metric, self._unit) if self._unit else self._metric

    def _write_global_stats(self, file_path: str) -> None:
        results = np.asarray([self.results_for_reasoner(r).values for r in self.reasoners()])

        self._global_stats = pd.DataFrame({
            'Reasoner': list(self.reasoners()),
            'Min ' + self._metric_and_unit: results.min(axis=1).flatten(),
            'Avg ' + self._metric_and_unit: results.mean(axis=1).flatten(),
            'Max ' + self._metric_and_unit: results.max(axis=1).flatten()
        }).set_index('Reasoner')

        self._global_stats.to_csv(file_path, float_format='%.2f')

    def _histogram_plotter(self, ax: plt.Axes) -> None:
        stats = self._global_stats
        reasoners = stats.index.values

        n_reasoners = len(reasoners)
        n_stats = len(stats.columns)

        width = 1.0 / (n_reasoners + 1)
        bar_width = 0.8 * width

        for i, reasoner in enumerate(reasoners):
            ax.bar([j + width * i for j in range(n_stats)],
                   stats.loc[reasoner].values,
                   bar_width, alpha=0.9, label=reasoner)

        plotutils.display_labels(ax)

        ax.set_title('Minimum, average and maximum {}'.format(self._metric))
        ax.set_ylabel(self._metric_and_unit.capitalize())

        ax.set_xticks([p + width * ((n_reasoners - 1) / 2) for p in range(n_stats)])
        ax.set_xticklabels(['Min', 'Avg', 'Max'])

        plotutils.set_scale(ax, 'log', axis='y')

        plotutils.display_grid(ax, axis='y')
        ax.legend(loc='upper left')

    def _scatter_plotter(self, ax: plt.Axes) -> None:
        reasoners = list(self.reasoners())
        dataset = Dataset(os.path.join(Paths.DATA_DIR, self.dataset_name))

        scale, unit = fileutils.human_readable_scale_and_unit(dataset.get_max_ontology_size())
        point_size = plotutils.configure_scatter_plot(ax, dataset.size)

        for i, reasoner in enumerate(reasoners):
            ontologies = dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                sort_by_size=True)
            results = self.results_for_reasoner(reasoner)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            ontologies = [o for o in ontologies if o.name in results.index]

            x = [o.size / scale for o in ontologies]
            y = [results.loc[o.name].values[0] for o in ontologies]

            ax.scatter(x, y, s=point_size, alpha=0.5, label=reasoner)

            weights = list(range(len(x), 0, -1))
            ax.plot(x, np.poly1d(np.polyfit(x, y, 1, w=weights))(x))

        ax.set_title('{} by ontology size'.format(self._metric.capitalize()))
        ax.set_xlabel('Ontology size ({})'.format(unit))
        ax.set_ylabel(self._metric_and_unit.capitalize())

        plotutils.display_grid(ax)
        ax.legend()


class MemoryEvaluator(SingleValueEvaluator):

    @property
    def _metric(self):
        return 'memory peak'

    @property
    def _unit(self):
        return 'MiB'

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(test_dir, cfg, index_columns)
        self._results /= (1024 * 1024)


class EnergyEvaluator(SingleValueEvaluator):

    @property
    def _metric(self):
        return 'energy'

    @property
    def _unit(self):
        return ''


class TimeEvaluator(Evaluator):

    class GlobalStats(NamedTuple):
        data: pd.DataFrame
        unit: str

    # Overrides

    @property
    def plotters(self):
        return super().plotters + [self.__histogram_plotter, self.__scatter_plotter]

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(test_dir, cfg, index_columns)
        self._global_stats: TimeEvaluator.GlobalStats = None

    def _write_results(self):
        self.__write_total_results(path.join(self.evaluation_dir, 'total_results.csv'))
        self.__write_global_stats(path.join(self.evaluation_dir, 'global_stats.csv'))

    # Private

    def __write_total_results(self, file_path: str) -> None:
        totals = self.results_grouped_by_reasoner().sum()
        totals.to_csv(file_path, float_format='%.2f')

    def __write_global_stats(self, file_path: str) -> None:
        reasoners = list(self.reasoners())

        parsing_cols = [c for c in self._results.columns if 'parsing' in c.lower()]
        reasoning_cols = [c for c in self._results.columns if c not in parsing_cols]

        parsing = self.results_grouped_by_reasoner(parsing_cols).sum().sum()
        reasoning = self.results_grouped_by_reasoner(reasoning_cols).sum().sum()

        parsing = np.asarray([parsing[r] for r in reasoners])
        reasoning = np.asarray([reasoning[r] for r in reasoners])

        if np.append(parsing, reasoning).min() < 1000.0:
            unit = 'ms'
        else:
            parsing /= 1000.0
            reasoning /= 1000.0
            unit = 's'

        data = pd.DataFrame({
            'Reasoner': reasoners,
            'Total parsing time ({})'.format(unit): parsing,
            'Total reasoning time ({})'.format(unit): reasoning,
            'Total time ({})'.format(unit): parsing + reasoning
        }).set_index('Reasoner')

        data.to_csv(file_path, float_format='%.2f')
        self._global_stats = self.GlobalStats(data, unit)

    def __histogram_plotter(self, ax: plt.Axes) -> None:

        data = self._global_stats.data

        reasoners = data.index.values
        parsing, reasoning = data.iloc[:, 0].values, data.iloc[:, 1].values

        pos = np.arange(len(reasoners))
        width = 0.35

        ax.bar(pos, parsing, width, alpha=0.9, label='Parsing')
        ax.bar(pos, reasoning, width, alpha=0.9, bottom=parsing, label='Reasoning')

        plotutils.display_labels(ax, center=True, fmt='{:.2f}')

        ax.set_title('Total parsing and reasoning time')
        ax.set_ylabel('Time ({})'.format(self._global_stats.unit))

        ax.set_xticks(pos)
        ax.set_xticklabels(reasoners)

        plotutils.display_grid(ax, axis='y')
        ax.legend(loc='upper left')

    def __scatter_plotter(self, ax: plt.Axes) -> None:
        reasoners = list(self.reasoners())
        dataset = Dataset(os.path.join(Paths.DATA_DIR, self.dataset_name))

        scale, unit = fileutils.human_readable_scale_and_unit(dataset.get_max_ontology_size())
        point_size = plotutils.configure_scatter_plot(ax, dataset.size)

        for reasoner in reasoners:
            ontologies = dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                sort_by_size=True)
            results = self.results_for_reasoner(reasoner)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            ontologies = [o for o in ontologies if o.name in results.index]

            x = [o.size / scale for o in ontologies]
            y = [results.loc[o.name].sum() for o in ontologies]

            ax.scatter(x, y, s=point_size, alpha=0.5, label=reasoner)

            weights = list(range(len(x), 0, -1))
            ax.plot(x, np.poly1d(np.polyfit(x, y, 1, w=weights))(x))

        ax.set_title('Parsing and reasoning time by ontology size')
        ax.set_xlabel('Ontology size ({})'.format(unit))
        ax.set_ylabel('Time (ms)')

        plotutils.display_grid(ax)
        ax.legend()
