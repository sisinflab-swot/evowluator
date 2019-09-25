from os import path
from typing import List, Optional

import numpy as np
import pandas as pd
from pyutils.io import fileutils

from .base import Visualizer
from .metric import Metric
from .plot import GroupedHistogramPlot


class CorrectnessVisualizer(Visualizer):

    # Overrides

    def __init__(self, results_dir: str, cfg, index_columns: List[str] = None) -> None:
        super().__init__(results_dir, cfg, index_columns=index_columns, non_numeric_columns=True)
        self.reasoners = self.reasoners[1:]
        self._summary: Optional[pd.DataFrame] = None

    def configure_plotters(self) -> None:
        super().configure_plotters()

        # Histogram plotter
        cols = ['correct', 'incorrect', 'timeout', 'error']
        stats = self._summary[cols]
        reasoners = stats.index.values
        self.figure.add_plotter(GroupedHistogramPlot,
                                title='Correctness results',
                                data=dict(zip(reasoners, (stats.loc[r].values for r in reasoners))),
                                metric=Metric('occurrences', fmt='.0f'),
                                groups=cols)

    def write_results(self) -> None:
        fileutils.create_dir(self.output_dir)

        reasoners = self.reasoners
        results = [self.results_for_reasoner(r)['match'].value_counts(sort=False)
                   for r in reasoners]

        stats = ['same', 'different', 'timeout', 'error']
        correct, incorrect, timeout, error = [np.asarray([r.get(s, default=0) for r in results])
                                              for s in stats]

        self._summary = pd.DataFrame({
            'reasoner': reasoners,
            'correct': correct,
            'incorrect': incorrect,
            'timeout': timeout,
            'error': error,
            'ratio': correct / (correct + incorrect + timeout + error)
        }).set_index('reasoner')

        self._summary.to_csv(path.join(self.output_dir, 'summary.csv'), float_format='%.2f')
