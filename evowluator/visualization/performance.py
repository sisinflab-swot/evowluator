from os import path
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import Visualizer
from .metric import Metric
from .plot import GroupedHistogramPlot


class SingleValueVisualizer(Visualizer):

    # Override

    @property
    def metric(self) -> Metric:
        raise NotImplementedError()

    # Overrides

    def configure_plotters(self) -> None:
        super().configure_plotters()
        self.add_min_max_avg_plotter(self._summary, self.metric)
        self.add_scatter_plotter(self.metric)

    def write_results(self):
        super().write_results()
        self._write_summary(path.join(self.output_dir, 'summary.csv'))

    def __init__(self, results_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(results_dir, cfg, index_columns)
        self._summary: Optional[pd.DataFrame] = None

    # Private

    def _write_summary(self, file_path: str) -> None:
        results = np.array([self.results_for_reasoner(r).values for r in self._reasoners])
        metric_str = self.metric.to_string()

        self._summary = pd.DataFrame({
            'Reasoner': self._reasoners,
            'Min ' + metric_str: np.min(results, axis=1).flatten(),
            'Avg ' + metric_str: np.mean(results, axis=1).flatten(),
            'Max ' + metric_str: np.max(results, axis=1).flatten()
        }).set_index('Reasoner')

        self._summary.to_csv(file_path, float_format='%.2f')


class EnergyVisualizer(SingleValueVisualizer):

    @property
    def metric(self):
        return Metric('energy', None, '.2f')


class PerformanceVisualizer(Visualizer):

    # Overrides

    def __init__(self, results_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(results_dir, cfg, index_columns)
        self._results[self._memory_cols()] /= (1024 * 1024)
        self._summary: Optional[pd.DataFrame] = None
        self._time_unit: str = 'ms'

    def configure_plotters(self) -> None:
        super().configure_plotters()

        # Time histogram
        data = self._summary.iloc[:, :2]
        reasoners = list(data.index.values)
        data = data.values.transpose()

        self.add_plotter(GroupedHistogramPlot,
                         title='Total parsing and reasoning time',
                         data=dict(zip(['Parsing', 'Reasoning'], list(data))),
                         metric=Metric('time', self._time_unit, '.0f'),
                         groups=reasoners)

        # Time scatter
        metric = Metric('time', 'ms', '.0f')
        self.add_scatter_plotter(metric, col_filter=lambda c: 'memory' not in c)

        # Memory histogram
        metric = Metric('memory peak', 'MiB', '.2f')
        self.add_min_max_avg_plotter(self._summary, metric, col_filter=lambda c: metric.name in c)

        # Memory scatter
        metric = Metric('memory peak', 'MiB', '.2f')
        self.add_scatter_plotter(metric, col_filter=lambda c: 'memory' in c)

    def write_results(self):
        super().write_results()
        self._write_total_times(path.join(self.output_dir, 'total_times.csv'))
        self._write_memory(path.join(self.output_dir, 'memory.csv'))
        self._write_summary(path.join(self.output_dir, 'summary.csv'))

    # Private

    def _memory_cols(self) -> List:
        return [c for c in self._results.columns if 'memory' in c.lower()]

    def _parsing_cols(self) -> List:
        return [c for c in self._results.columns if 'parsing' in c.lower()]

    def _reasoning_cols(self) -> List:
        other_cols = self._memory_cols() + self._parsing_cols()
        return [c for c in self._results.columns if c not in other_cols]

    def _write_total_times(self, file_path: str) -> None:
        cols = self._memory_cols()
        cols = [c for c in self._results.columns if c not in cols]
        totals = self.results_grouped_by_reasoner(cols).sum()
        totals.to_csv(file_path, float_format='%.2f')

    def _write_memory(self, file_path: str) -> None:
        totals = self.results_grouped_by_reasoner(self._memory_cols()).sum()
        totals.to_csv(file_path, float_format='%.2f')

    def _write_summary(self, file_path: str) -> None:
        reasoners = self._reasoners
        parsing_cols = self._parsing_cols()
        reasoning_cols = self._reasoning_cols()
        memory_cols = self._memory_cols()

        parsing = self.results_grouped_by_reasoner(parsing_cols).sum().sum()
        reasoning = self.results_grouped_by_reasoner(reasoning_cols).sum().sum()

        memory = self.results_grouped_by_reasoner(memory_cols).sum()
        memory_min, memory_mean, memory_max = memory.min(), memory.mean(), memory.max()

        parsing = np.array([parsing[r] for r in reasoners])
        reasoning = np.array([reasoning[r] for r in reasoners])
        memory_min = np.array([memory_min[r] for r in reasoners])
        memory_mean = np.array([memory_mean[r] for r in reasoners])
        memory_max = np.array([memory_max[r] for r in reasoners])

        if np.min(np.append(parsing, reasoning)) < 1000.0:
            time_unit = 'ms'
        else:
            parsing /= 1000.0
            reasoning /= 1000.0
            time_unit = 's'

        time_unit = f' ({time_unit})'

        data = pd.DataFrame({
            'Reasoner': reasoners,
            'Total parsing time' + time_unit: parsing,
            'Total reasoning time' + time_unit: reasoning,
            'Total time' + time_unit: parsing + reasoning,
            'Min memory peak (MiB)': memory_min,
            'Avg memory peak (MiB)': memory_mean,
            'Max memory peak (MiB)': memory_max
        }).set_index('Reasoner')

        data.to_csv(file_path, float_format='%.2f')

        self._summary = data
        self._time_unit = time_unit
