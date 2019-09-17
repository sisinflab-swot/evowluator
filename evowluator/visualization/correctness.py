from os import path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotutils
from .base import Visualizer
from .metric import Metric


class CorrectnessVisualizer(Visualizer):

    # Overrides

    @property
    def plotters(self):
        return super().plotters + [self._histogram_plotter]

    def __init__(self, results_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(results_dir, cfg, index_columns=index_columns, non_numeric_columns=True)
        self.reasoners = self.reasoners[1:]
        self._global_stats: Optional[pd.DataFrame] = None

    def write_results(self) -> None:
        super().write_results()
        self._write_global_stats(path.join(self.output_dir, 'global_stats.csv'))

    # Private

    def _write_global_stats(self, file_path: str) -> None:
        reasoners = self.reasoners
        results = [self.results_for_reasoner(r)['match'].value_counts(sort=False)
                   for r in reasoners]

        stats = ['same', 'different', 'timeout', 'error']
        correct, incorrect, timeout, error = [np.asarray([r.get(s, default=0) for r in results])
                                              for s in stats]

        self._global_stats = pd.DataFrame({
            'reasoner': reasoners,
            'correct': correct,
            'incorrect': incorrect,
            'timeout': timeout,
            'error': error,
            'ratio': correct / (correct + incorrect + timeout + error)
        }).set_index('reasoner')

        self._global_stats.to_csv(file_path, float_format='%.2f')

    def _histogram_plotter(self, ax: plt.Axes) -> None:
        cols = ['correct', 'incorrect', 'timeout', 'error']
        stats = self._global_stats[cols]
        reasoners = stats.index.values
        data = [stats.loc[r].values for r in reasoners]

        metric = Metric('occurrences', fmt='.0f')
        plotutils.draw_grouped_histograms(ax, dict(zip(reasoners, data)), metric, cols)
        ax.set_title('Correctness results')
