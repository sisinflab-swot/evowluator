from __future__ import annotations

from functools import cached_property
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
        self.fields = list(cfg[ConfigKey.FIELDS])
        self._energy_probes = {p[ConfigKey.NAME] for p in cfg.get(ConfigKey.ENERGY_PROBES, [])}
        self._summary: pd.DataFrame | None = None
        self._cumulative_time_metric = Metric('time', 'ms', '.2f')
        if self._has_memory:
            self._results[self._memory_cols] /= (1024 * 1024)

    def configure_plotters(self) -> None:
        super().configure_plotters()

        time_metric = Metric('time', 'ms', '.0f')
        memory_metric = Metric('memory peak', 'MiB', '.2f')

        # Time histogram
        cols = [f.capitalize() for f in self._time_fields]
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
        if self._has_memory:
            self.add_min_max_avg_plotter(self._summary, memory_metric,
                                         col_filter=lambda c: Field.MEMORY in c)

        # Energy histogram
        for ef in self._energy_fields:
            self.add_min_max_avg_plotter(self._summary, Metric('energy', ef, '.2f'),
                                         col_filter=lambda c: any(f in c
                                                                  for f in self._energy_fields))

        # Time scatter
        if self._time_fields:
            self.add_scatter_plotter(time_metric, col_filter=lambda c: c in self._time_fields)

        # Memory scatter
        if self._has_memory:
            self.add_scatter_plotter(memory_metric, col_filter=lambda c: Field.MEMORY in c)

        # Energy scatter
        for ef in self._energy_fields:
            self.add_scatter_plotter(Metric('energy', ef, '.2f'), col_filter=lambda c: c == ef)

    def write_results(self):
        super().write_results()
        self._write_total_times(path.join(self.output_dir, 'total_times.csv'))
        self._write_summary(path.join(self.output_dir, 'summary.csv'))

    # Private

    @cached_property
    def _has_memory(self) -> bool:
        return True if Field.MEMORY in self.fields else False

    @cached_property
    def _time_fields(self) -> List[str]:
        excluded = [Field.MEMORY] + self._energy_fields
        return [f for f in self.fields if f not in excluded]

    @cached_property
    def _energy_fields(self) -> List[str]:
        return list(sorted(f for f in self.fields if f in self._energy_probes))

    @cached_property
    def _time_cols(self) -> List:
        return [c for c in self._results.columns if any(f in c for f in self._time_fields)]

    @cached_property
    def _parsing_cols(self) -> List:
        return [c for c in self._time_cols if Field.PARSING in c]

    @cached_property
    def _reasoning_cols(self) -> List:
        return [c for c in self._time_cols if c not in self._parsing_cols]

    @cached_property
    def _memory_cols(self) -> List:
        return [c for c in self._results.columns if Field.MEMORY in c] if self._has_memory else []

    @cached_property
    def _energy_cols(self) -> List:
        return [c for c in self._results.columns if any(f in c for f in self._energy_fields)]

    def _energy_probe_cols(self, probe: str) -> List:
        return [c for c in self._energy_cols if probe in c]

    def _time_field_cols(self, field: str) -> List:
        return [c for c in self._time_cols if field in c]

    def _write_total_times(self, file_path: str) -> None:
        if self._time_cols:
            totals = self.results_grouped_by_reasoner(self._time_cols,
                                                      drop_missing=False).sum(min_count=1)
            csv.write(totals, file_path)

    def _write_summary(self, file_path: str) -> None:
        reasoners = self._reasoners
        summary = pd.DataFrame({'reasoner': reasoners}).set_index('reasoner')

        times = {}
        inf_time = float('inf')
        min_time = inf_time
        max_time = 0

        for field in self._time_fields:
            data = self.results_grouped_by_reasoner(self._time_field_cols(field)).sum().sum()
            data = np.array([data[r] for r in reasoners])
            min_time = min(np.min(np.ma.masked_equal(data, 0)), min_time)
            max_time = max(np.max(data), max_time)
            times[field] = data

        if min_time != inf_time and ((min_time >= 1000.0) or
                                     (min_time >= 100.0 and max_time >= 10000.0)):
            self._cumulative_time_metric.unit = 's'
            self._cumulative_time_metric.fmt = '.2f'
            for field in times:
                times[field] /= 1000.0

        if min_time != inf_time:
            time_unit = f' ({self._cumulative_time_metric.unit})'
            for field, data in times.items():
                summary[field + time_unit] = data
            if len(times) > 1:
                summary['total time' + time_unit] = np.sum(list(times.values()))

        min_max_avg_metrics = [('memory peak (MiB)', self._memory_cols)]
        min_max_avg_metrics.extend((f'energy ({f})', self._energy_probe_cols(f))
                                   for f in self._energy_fields)

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
