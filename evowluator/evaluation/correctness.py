from os import path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotutils
from .evaluator import Evaluator
from .metric import Metric


class CorrectnessEvaluator(Evaluator):

    # Overrides

    @property
    def plotters(self):
        return super().plotters + [self._histogram_plotter]

    def reasoners(self):
        return list(super().reasoners())[1:]

    def __init__(self, test_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(test_dir, cfg, index_columns=index_columns, non_numeric_columns=True)
        self._global_stats: Optional[pd.DataFrame] = None

    def write_results(self) -> None:
        super().write_results()
        self._write_global_stats(path.join(self.evaluation_dir, 'global_stats.csv'))

    # Private

    def _write_global_stats(self, file_path: str) -> None:
        results = [self.results_for_reasoner(r)['match'].value_counts(sort=False)
                   for r in self.reasoners()]

        stats = ['same', 'different', 'timeout', 'error']
        correct, incorrect, timeout, error = [np.asarray([r.get(s, default=0) for r in results])
                                              for s in stats]

        self._global_stats = pd.DataFrame({
            'reasoner': list(self.reasoners()),
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
