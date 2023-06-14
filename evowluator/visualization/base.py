from __future__ import annotations

import os
from os import path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from pyutils.io import file
from pyutils.types.unit import MemoryUnit
from .metric import Metric
from .plot import Figure, LineStyle, MinMaxAvgHistogramPlot, ScatterPlot
from ..config import ConfigKey, Paths
from ..data import csv, json
from ..data.dataset import SortBy
from ..data.info import DatasetInfo
from ..data.syntax import Syntax
from ..evaluation.mode import EvaluationMode


class Visualizer:

    # Override

    def configure_plotters(self) -> None:
        pass

    def write_results(self) -> None:
        file.create_dir(self.output_dir)
        csv.write(self._results, path.join(self.output_dir, 'avg_results.csv'))

    # Public

    @classmethod
    def from_dir(cls, results_dir: str, reasoners: List[str] | None = None) -> Visualizer:
        from .correctness import CorrectnessVisualizer
        from .performance import PerformanceVisualizer

        cfg = json.load(os.path.join(results_dir, Paths.CONFIG_FILE_NAME))

        # Legacy, keep support for compatibility
        eval_name = cfg.get(ConfigKey.NAME)

        if eval_name:
            task, mode = eval_name.split(maxsplit=1)
        else:
            task = cfg[ConfigKey.TASK]
            mode = cfg[ConfigKey.MODE]
            eval_name = f'{task} {mode}'
            cfg[ConfigKey.NAME] = eval_name

        if reasoners:
            reasoner_cfg = {r[ConfigKey.NAME]: r for r in cfg[ConfigKey.REASONERS]}
            cfg[ConfigKey.REASONERS] = [reasoner_cfg[r] for r in reasoners]

        mode = EvaluationMode(mode)

        if mode == EvaluationMode.CORRECTNESS:
            return CorrectnessVisualizer(results_dir, cfg)
        elif mode == EvaluationMode.PERFORMANCE:
            return PerformanceVisualizer(results_dir, cfg)
        else:
            raise NotImplementedError(f'Visualizer not implemented for "{eval_name}"')

    @property
    def results_path(self) -> str:
        return os.path.join(self._results_dir, Paths.RESULTS_FILE_NAME)

    @property
    def output_dir(self) -> str:
        return path.join(self._results_dir, 'visualization')

    @property
    def config_path(self) -> str:
        return os.path.join(self._results_dir, Paths.CONFIG_FILE_NAME)

    def __init__(self, results_dir: str, cfg,
                 non_numeric_columns: bool | List[str] = False) -> None:
        self._results_dir = Paths.absolute(results_dir, Paths.RESULTS_DIR)
        self._dataset = DatasetInfo.from_dict(cfg[ConfigKey.DATASET])

        self._syntaxes_by_reasoner: Dict[str, Syntax] = {
            r[ConfigKey.NAME]: Syntax(r[ConfigKey.SYNTAX])
            for r in cfg[ConfigKey.REASONERS]
        }

        self._reasoners: List[str] = [r[ConfigKey.NAME] for r in cfg[ConfigKey.REASONERS]]
        self._results: pd.DataFrame = self.load_results(non_numeric_columns)

        self.colors: Dict[str, str] = {}
        self.markers: Dict[str, str] = {}
        self.line_styles: Dict[str, str] = {}
        self.figure = Figure(f'evOWLuator: {cfg[ConfigKey.NAME]} '
                             f'on "{self._dataset.name}" dataset '
                             f'({path.basename(self._results_dir)})')

    def ontologies(self) -> Iterable[str]:
        return self._results.index.values

    def results_for_reasoner(self, reasoner: str,
                             col_filter: Callable[[str], bool] | None = None,
                             drop_missing: bool = True) -> pd.DataFrame:
        results = self._results.dropna() if drop_missing else self._results

        needle = reasoner + ':'
        results = results[[f for f in self._results if f.startswith(needle)]]
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

    def plot_results(self, gui: bool = True, transparent_bg: bool = False,
                     plots: List[int] | None = None) -> None:
        self.configure_plotters()
        self.figure.draw(plots=plots)
        self.figure.save(path.join(self.output_dir, 'figure.pdf'), transparent=transparent_bg)

        if gui:
            self.figure.show()

    def load_results(self, non_numeric_columns: bool | List[str] = False) -> pd.DataFrame:
        res = csv.read(self.results_path)
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

        if not res.index.is_unique:
            res = res.groupby(level=res.index.names).mean()

        return res

    def set_colors(self, colors: List[str]) -> None:
        self.colors = {
            self._reasoners[idx]: color
            for idx, color in enumerate(colors[:len(self._reasoners)]) if color != 'auto'
        }
    
    def set_line_styles(self, styles: List[str]) -> None:
        self.line_styles = {
            self._reasoners[idx]: self._parse_line_style(style)
            for idx, style in enumerate(styles[:len(self._reasoners)]) if style != 'auto'
        }

    def set_markers(self, markers: List[str]) -> None:
        self.markers = {
            self._reasoners[idx]: marker
            for idx, marker in enumerate(markers[:len(self._reasoners)]) if marker != 'auto'
        }

    def add_plotter(self, plot_type: type, **kwargs) -> None:
        kwargs['colors'] = self.colors
        kwargs['markers'] = self.markers
        kwargs['line_styles'] = self.line_styles
        self.figure.add_plotter(plot_type, **kwargs)

    def add_scatter_plotter(self, metric: Metric, separate_cols: bool = False,
                            col_filter: Callable[[str], bool] | None = None) -> None:
        xunit = MemoryUnit.B(self._dataset.max_ontology_size()).readable().unit
        xmetric = Metric('ontology size', xunit, '.2f')

        data = {}

        for reasoner in self._reasoners:
            results = self.results_for_reasoner(reasoner, col_filter=col_filter, drop_missing=False)
            ontologies = self._dataset.get_ontologies(self._syntaxes_by_reasoner[reasoner],
                                                      names=results.index, sort_by=SortBy.SIZE)

            if isinstance(results.index, pd.MultiIndex):
                results = results.groupby(level=0).mean()

            x, y = [], []

            if separate_cols:
                for col in results.columns:
                    x, y = [], []
                    for onto in ontologies:
                        yi = results.loc[onto.name][col]
                        if not np.isnan(yi):
                            x.append(MemoryUnit.B(onto.size).to_value(xunit))
                            y.append(yi)
                    label = f'{reasoner}: {col}' if len(self._reasoners) > 1 else col.capitalize()
                    data[label] = (x, y)
            else:
                for onto in ontologies:
                    yi = results.loc[onto.name].sum(skipna=False)
                    if not np.isnan(yi):
                        x.append(MemoryUnit.B(onto.size).to_value(xunit))
                        y.append(yi)
                data[reasoner] = (x, y)

        self.add_plotter(ScatterPlot, data=data, xmetric=xmetric, ymetric=metric)

    def add_min_max_avg_plotter(self, data: pd.DataFrame, metric: Metric,
                                col_filter: Callable[[str], bool] | None = None) -> None:
        if col_filter:
            cols = [c for c in data.columns if col_filter(c)]
            data = data[cols]

        reasoners = data.index.values

        data = [data.loc[r].values for r in reasoners]
        data = dict(zip(reasoners, data))
        self.add_plotter(MinMaxAvgHistogramPlot, data=data, metric=metric)

    def _parse_line_style(self, style: str) -> LineStyle:
        valid = {
            'solid': '-',
            'loosely_dotted':           (0, (1, 10)),
            'dotted':                   (0, (1, 1)),
            'densely_dotted':           (0, (1, 1)),
            'loosely_dashed':           (0, (5, 10)),
            'dashed':                   (0, (5, 5)),
            'densely_dashed':           (0, (5, 1)),
            'loosely_dashdotted':       (0, (3, 10, 1, 10)),
            'dashdotted':               (0, (3, 5, 1, 5)),
            'densely_dashdotted':       (0, (3, 1, 1, 1)),
            'dashdotdotted':            (0, (3, 5, 1, 5, 1, 5)),
            'loosely_dashdotdotted':    (0, (3, 10, 1, 10, 1, 10)),
            'densely_dashdotdotted':    (0, (3, 1, 1, 1, 1, 1))
        }

        line_style = valid.get(style)
        
        if not line_style:
            raise ValueError(f'Invalid line style "{style}". '
                             f'Supported values: {", ".join(valid.keys())}')
        
        return line_style
