from __future__ import annotations

from functools import cached_property
from os import path
from typing import List

import numpy as np
import pandas as pd

from .base import Visualizer
from .metric import Metric
from .plot import GroupedHistogramPlot
from ..data import csv


class PerformanceVisualizer(Visualizer):

    # Overrides

    def __init__(self, results_dir: str, cfg) -> None:
        super().__init__(results_dir, cfg)
        self._results[self._memory_cols] /= (1024 * 1024)
        self._summary: pd.DataFrame | None = None
        self._cumulative_time_metric = Metric('time', 'ms', '.0f')

    def configure_plotters(self) -> None:
        super().configure_plotters()

        time_metric = Metric('time', 'ms', '.0f')
        memory_metric = Metric('memory peak', 'MiB', '.2f')
        energy_metric = metric = Metric('energy', None, '.2f')

        # Time histogram
        data = self._summary.iloc[:, :2]
        reasoners = list(data.index.values)
        data = data.values.transpose()

        self.add_plotter(GroupedHistogramPlot,
                         title='Total parsing and reasoning time',
                         data=dict(zip(['Parsing', 'Reasoning'], list(data))),
                         metric=self._cumulative_time_metric,
                         groups=reasoners,
                         show_zero_labels=False)

        # Memory histogram
        self.add_min_max_avg_plotter(self._summary, memory_metric,
                                     col_filter=lambda c: 'memory' in c)

        if self._has_energy:
            # Energy histogram
            self.add_min_max_avg_plotter(self._summary, metric, col_filter=lambda c: 'energy' in c)

        # Time scatter
        self.add_scatter_plotter(time_metric, col_filter=lambda c: c not in ('memory', 'energy'))

        # Memory scatter
        self.add_scatter_plotter(memory_metric, col_filter=lambda c: c == 'memory')

        if self._has_energy:
            # Energy scatter
            self.add_scatter_plotter(energy_metric, col_filter=lambda c: c == 'energy')

    def write_results(self):
        super().write_results()
        self._write_total_times(path.join(self.output_dir, 'total_times.csv'))
        self._write_summary(path.join(self.output_dir, 'summary.csv'))

    # Private

    @cached_property
    def _memory_cols(self) -> List:
        return [c for c in self._results.columns if 'memory' in c.lower()]

    @cached_property
    def _energy_cols(self) -> List:
        return [c for c in self._results.columns if 'energy' in c.lower()]

    @cached_property
    def _parsing_cols(self) -> List:
        return [c for c in self._results.columns if 'parsing' in c.lower()]

    @cached_property
    def _reasoning_cols(self) -> List:
        other_cols = self._memory_cols + self._energy_cols + self._parsing_cols
        return [c for c in self._results.columns if c not in other_cols]

    @cached_property
    def _time_cols(self) -> List:
        return self._parsing_cols + self._reasoning_cols

    @property
    def _has_energy(self) -> bool:
        return True if self._energy_cols else False

    def _write_total_times(self, file_path: str) -> None:
        totals = self.results_grouped_by_reasoner(self._time_cols,
                                                  drop_missing=False).sum(min_count=1)
        csv.write(totals, file_path)

    def _write_summary(self, file_path: str) -> None:
        reasoners = self._reasoners

        parsing = self.results_grouped_by_reasoner(self._parsing_cols).sum().sum()
        reasoning = self.results_grouped_by_reasoner(self._reasoning_cols).sum().sum()

        parsing = np.array([parsing[r] for r in reasoners])
        reasoning = np.array([reasoning[r] for r in reasoners])

        if np.min(np.ma.masked_equal(np.append(parsing, reasoning), 0, copy=False)) >= 1000.0:
            parsing /= 1000.0
            reasoning /= 1000.0
            self._cumulative_time_metric.unit = 's'
            self._cumulative_time_metric.fmt = '.2f'

        time_unit = f' ({self._cumulative_time_metric.unit})'

        summary = pd.DataFrame({
            'reasoner': reasoners,
            'total parsing time' + time_unit: parsing,
            'total reasoning time' + time_unit: reasoning,
            'total time' + time_unit: parsing + reasoning
        }).set_index('reasoner')

        min_max_avg_metrics = (
            ('memory peak (MiB)', self._memory_cols),
            ('energy score', self._energy_cols)
        )

        for metric, cols in [(m, c) for m, c in min_max_avg_metrics if c]:
            res = self.results_grouped_by_reasoner(cols).sum()
            res_min, res_avg, res_max = res.min(), res.mean(), res.max()
            res_min = np.array([res_min[r] for r in reasoners])
            res_avg = np.array([res_avg[r] for r in reasoners])
            res_max = np.array([res_max[r] for r in reasoners])
            summary[f'min {metric}'] = res_min
            summary[f'avg {metric}'] = res_avg
            summary[f'max {metric}'] = res_max

        csv.write(summary, file_path)
        self._summary = summary
