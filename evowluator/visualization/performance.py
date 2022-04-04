from __future__ import annotations

from functools import cache
from os import path
from typing import List

import numpy as np
import pandas as pd

from .base import Visualizer
from .metric import Metric
from .plot import GroupedHistogramPlot
from ..config import ConfigKey
from ..data import csv
from ..reasoner.results import Field


class PerformanceVisualizer(Visualizer):

    # Overrides

    def __init__(self, results_dir: str, cfg) -> None:
        super().__init__(results_dir, cfg)
        self.fields = set(cfg[ConfigKey.FIELDS])
        self._summary: pd.DataFrame | None = None
        self._cumulative_time_metric = Metric('time', 'ms', '.2f')
        if self._has(Field.MEMORY):
            self._results[self._cols(Field.MEMORY)] /= (1024 * 1024)

    def configure_plotters(self) -> None:
        super().configure_plotters()

        time_metric = Metric('time', 'ms', '.0f')
        memory_metric = Metric('memory peak', 'MiB', '.2f')
        energy_metric = Metric('energy', None, '.2f')

        # Time histogram
        cols = [f.capitalize() for f in (Field.PARSING, Field.REASONING) if self._has(f)]
        if cols:
            data = self._summary.iloc[:, :len(cols)]
            reasoners = list(data.index.values)
            data = data.values.transpose()
            self.add_plotter(GroupedHistogramPlot,
                             title='Total time',
                             data=dict(zip(cols, list(data))),
                             metric=self._cumulative_time_metric,
                             groups=reasoners,
                             show_zero_labels=False)

        # Memory histogram
        if self._has(Field.MEMORY):
            self.add_min_max_avg_plotter(self._summary, memory_metric,
                                         col_filter=lambda c: Field.MEMORY in c)

        # Energy histogram
        if self._has(Field.ENERGY):
            self.add_min_max_avg_plotter(self._summary, energy_metric,
                                         col_filter=lambda c: Field.ENERGY in c)

        # Time scatter
        if self._has(Field.PARSING) or self._has(Field.REASONING):
            excluded = [Field.MEMORY, Field.ENERGY]
            excluded.extend(f for f in (Field.PARSING, Field.REASONING) if not self._has(f))
            self.add_scatter_plotter(time_metric, col_filter=lambda c: c not in excluded)

        # Memory scatter
        if self._has(Field.MEMORY):
            self.add_scatter_plotter(memory_metric, col_filter=lambda c: c == Field.MEMORY)

        # Energy scatter
        if self._has(Field.ENERGY):
            self.add_scatter_plotter(energy_metric, col_filter=lambda c: c == Field.ENERGY)

    def write_results(self):
        super().write_results()
        self._write_total_times(path.join(self.output_dir, 'total_times.csv'))
        self._write_summary(path.join(self.output_dir, 'summary.csv'))

    # Private

    def _has(self, field: Field) -> bool:
        return field in self.fields

    @cache
    def _cols(self, field: Field) -> List:
        if self._has(field):
            return [c for c in self._results.columns if field in c.lower()]
        return []

    def _write_total_times(self, file_path: str) -> None:
        cols = self._cols(Field.PARSING) + self._cols(Field.REASONING)
        if cols:
            totals = self.results_grouped_by_reasoner(cols, drop_missing=False).sum(min_count=1)
            csv.write(totals, file_path)

    def _write_summary(self, file_path: str) -> None:
        reasoners = self._reasoners
        summary = pd.DataFrame({'reasoner': reasoners}).set_index('reasoner')

        parsing = reasoning = None
        inf_time = float('inf')
        min_time = inf_time
        max_time = 0

        if self._has(Field.PARSING):
            parsing = self.results_grouped_by_reasoner(self._cols(Field.PARSING)).sum().sum()
            parsing = np.array([parsing[r] for r in reasoners])
            min_time = np.min(np.ma.masked_equal(parsing, 0))
            max_time = np.max(parsing)

        if self._has(Field.REASONING):
            reasoning = self.results_grouped_by_reasoner(self._cols(Field.REASONING)).sum().sum()
            reasoning = np.array([reasoning[r] for r in reasoners])
            min_time = min(np.min(np.ma.masked_equal(reasoning, 0)), min_time)
            max_time = max(np.max(reasoning), max_time)

        if min_time != inf_time and ((min_time >= 1000.0) or
                                     (min_time >= 100.0 and max_time >= 10000.0)):
            self._cumulative_time_metric.unit = 's'
            self._cumulative_time_metric.fmt = '.2f'
            if parsing is not None:
                parsing /= 1000.0
            if reasoning is not None:
                reasoning /= 1000.0

        if min_time != inf_time:
            time_unit = f' ({self._cumulative_time_metric.unit})'
            if parsing is not None:
                summary['total parsing time' + time_unit] = parsing
            if reasoning is not None:
                summary['total reasoning time' + time_unit] = reasoning
            if parsing is not None and reasoning is not None:
                summary['total time' + time_unit] = parsing + reasoning

        min_max_avg_metrics = (
            ('memory peak (MiB)', self._cols(Field.MEMORY)),
            ('energy score', self._cols(Field.ENERGY))
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
