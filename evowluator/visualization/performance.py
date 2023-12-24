from __future__ import annotations

from functools import cached_property
from os import path
from typing import List

import numpy as np
import pandas as pd

from pyutils.types.unit import MemoryUnit, TimeUnit
from .base import Visualizer
from .metric import Metric
from .plot import GroupedHistogramPlot
from ..config.key import ConfigKey
from ..data import csv


class PerformanceVisualizer(Visualizer):

    @property
    def time_unit(self) -> TimeUnit:
        return self._time_unit if self._time_unit else TimeUnit.MS

    @time_unit.setter
    def time_unit(self, unit: TimeUnit) -> None:
        if not hasattr(unit, 'multiplier'):
            unit = TimeUnit(unit)
        mult = unit.multiplier / self.time_unit.multiplier
        self._results[self._time_cols] /= mult
        self._time_unit = unit

    @property
    def memory_unit(self) -> MemoryUnit:
        return self._memory_unit if self._memory_unit else MemoryUnit.B

    @memory_unit.setter
    def memory_unit(self, unit: MemoryUnit) -> None:
        if not hasattr(unit, 'multiplier'):
            unit = MemoryUnit(unit)
        mult = unit.multiplier / self.memory_unit.multiplier
        self._results[self._memory_cols] /= mult
        self._memory_unit = unit

    @property
    def time_metric(self) -> Metric:
        unit = self.time_unit
        return Metric('time', unit, '.0f' if unit.multiplier < TimeUnit.S.multiplier else '.2f')

    @property
    def cumulative_time_metric(self) -> Metric:
        unit = self._cumulative_time_unit
        return Metric('time', unit, '.2f')

    @property
    def memory_metric(self) -> Metric:
        unit = self.memory_unit
        return Metric('memory', unit, '.0f' if unit == MemoryUnit.B else '.2f')

    def __init__(self, results_dir: str, cfg) -> None:
        super().__init__(results_dir, cfg)
        self.fields = list(cfg[ConfigKey.FIELDS])
        self.separate_fields = False
        self._energy_probes = {p[ConfigKey.NAME] for p in cfg.get(ConfigKey.ENERGY_PROBES, [])}
        self._summary: pd.DataFrame | None = None
        self._time_unit: TimeUnit | None = None
        self._cumulative_time_unit: TimeUnit | None = None
        self._memory_unit: MemoryUnit | None = None
        self._autoscale_results()

    def _autoscale_results(self) -> None:
        if self._has_memory:
            avg_mem = self._results[self._memory_cols].mean(axis=0).mean()
            self.memory_unit = MemoryUnit.B(avg_mem).readable().unit


    def configure_plotters(self) -> None:
        super().configure_plotters()

        # Time histogram
        cols = [f.capitalize() for f in self._time_fields]
        if cols:
            data = self._summary.iloc[:, :len(cols)]
            reasoners = list(data.index.values)
            data = data.values.transpose()
            self.add_plotter(GroupedHistogramPlot,
                             title='Total time',
                             data=dict(zip(cols, list(data))),
                             metric=self.cumulative_time_metric,
                             groups=reasoners,
                             show_zero_labels=False)

        # Memory histogram
        if self._has_memory:
            self.add_min_max_avg_plotter(self._summary, self.memory_metric,
                                         col_filter=lambda c: 'memory' in c)

        # Energy histogram
        for ef in self._energy_fields:
            self.add_min_max_avg_plotter(self._summary, Metric('energy', ef, '.2f'),
                                         col_filter=lambda c: any(f in c
                                                                  for f in self._energy_fields))

        # Time scatter
        if self._time_fields:
            self.add_scatter_plotter(self.time_metric, separate_cols=self.separate_fields,
                                     col_filter=lambda c: c in self._time_fields)

        # Memory scatter
        if self._has_memory:
            self.add_scatter_plotter(self.memory_metric, col_filter=lambda c: 'memory' in c)

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
        return True if 'memory' in self.fields else False

    @cached_property
    def _time_fields(self) -> List[str]:
        excluded = ['memory'] + self._energy_fields
        return [f for f in self.fields if f not in excluded]

    @cached_property
    def _energy_fields(self) -> List[str]:
        return list(sorted(f for f in self.fields if f in self._energy_probes))

    @cached_property
    def _time_cols(self) -> List:
        return [c for c in self._results.columns if any(f in c for f in self._time_fields)]

    @cached_property
    def _parsing_cols(self) -> List:
        return [c for c in self._time_cols if 'parsing' in c]

    @cached_property
    def _reasoning_cols(self) -> List:
        return [c for c in self._time_cols if c not in self._parsing_cols]

    @cached_property
    def _memory_cols(self) -> List:
        return [c for c in self._results.columns if 'memory' in c] if self._has_memory else []

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
            if self._time_unit:
                self._cumulative_time_unit = self._time_unit
            else:
                self._cumulative_time_unit = TimeUnit.S
                for field in times:
                    times[field] /= 1000.0

        if min_time != inf_time:
            time_unit = f' ({self._cumulative_time_unit})'
            for field, data in times.items():
                summary[field + time_unit] = data
            if len(times) > 1:
                summary['total time' + time_unit] = np.sum(list(times.values()), axis=0)

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
