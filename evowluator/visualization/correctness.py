from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from os import path
from random import shuffle
from typing import Dict, List

import numpy as np
import pandas as pd

from pyutils.io import file
from pyutils.types.stringutils import camel_case_split
from .base import Visualizer
from .metric import Metric
from .plot import GroupedHistogramPlot
from ..data import csv
from ..reasoner.base import Reasoner


class Status:
    OK = 'y'
    INCORRECT = 'n'
    TIMEOUT = 'timeout'
    ERROR = 'error'
    UNKNOWN = 'unknown'

    NOT_OK = [INCORRECT, TIMEOUT, ERROR]
    KNOWN = [OK] + NOT_OK
    ALL = KNOWN + [UNKNOWN]


class CorrectnessStrategy(ABC):

    __ALL: List[CorrectnessStrategy] = None

    @classmethod
    def all(cls) -> List[CorrectnessStrategy]:
        if cls.__ALL is None:
            cls.__ALL = [RandomMajorityStrategy(), OracleStrategy()]
        return cls.__ALL

    @classmethod
    def default(cls) -> CorrectnessStrategy:
        return cls.all()[0]

    @classmethod
    def with_name(cls, name: str | None, reasoners: List[str] | None = None) -> CorrectnessStrategy:
        if not name:
            return cls.default()

        try:
            return next(s for s in cls.all() if s.name == name)
        except StopIteration:
            pass

        try:
            idx = reasoners.index(name)
            strategy = CorrectnessStrategy.with_name('oracle')
            strategy.oracle_idx = idx
            return strategy
        except ValueError:
            raise ValueError(f'No correctness strategy named "{name}"')

    @cached_property
    def name(self) -> str:
        return '_'.join(list(t.lower() for t in camel_case_split(type(self).__name__))[:-1])

    @abstractmethod
    def evaluate(self, results: List) -> List:
        pass

    def evaluate_dict(self, results: Dict[Reasoner]) -> Dict[Reasoner]:
        return dict(zip(results.keys(), self.evaluate(list(results.values()))))

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.transform(self.evaluate, axis=1)


class OracleStrategy(CorrectnessStrategy):

    def __init__(self, oracle_idx: int = 0) -> None:
        self.oracle_idx = oracle_idx

    def evaluate(self, results: List) -> List:
        out = []

        ref_idx = self.oracle_idx
        ref_res = results[ref_idx]
        ref_ok = ref_res not in Status.ALL

        for idx, res in enumerate(results):
            if res not in Status.ALL:
                if idx == ref_idx:
                    res = Status.OK
                elif ref_ok:
                    res = Status.OK if res == ref_res else Status.INCORRECT
                else:
                    res = Status.UNKNOWN
            out.append(res)

        return out


class RandomMajorityStrategy(CorrectnessStrategy):

    def evaluate(self, results: List) -> List:
        out: List = [None] * len(results)
        groups = {}

        for idx, res in enumerate(results):
            if res in Status.ALL:
                out[idx] = res
            elif res in groups:
                groups[res].append(idx)
            else:
                groups[res] = [idx]

        if groups:
            groups = list(groups.values())
            shuffle(groups)
            correct = max((len(g), g) for g in groups)[1]

            for group in groups:
                for idx in group:
                    out[idx] = Status.OK if group is correct else Status.INCORRECT

        return out


class CorrectnessVisualizer(Visualizer):

    def __init__(self, results_dir: str, cfg) -> None:
        super().__init__(results_dir, cfg, non_numeric_columns=True)
        self.strategy: CorrectnessStrategy | None = None
        self.summary: pd.DataFrame | None = None

    def set_strategy(self, strategy: str) -> None:
        self.strategy = CorrectnessStrategy.with_name(strategy, self._reasoners)

    def configure_plotters(self) -> None:
        super().configure_plotters()

        # Histogram plotter
        cols = self.summary.columns[:-1]
        stats = self.summary[cols]
        reasoners = stats.index.values
        self.add_plotter(GroupedHistogramPlot,
                         title='Correctness results',
                         data=dict(zip(reasoners, (stats.loc[r].values for r in reasoners))),
                         metric=Metric('occurrences', fmt='.0f'),
                         groups=cols)

    def write_results(self) -> None:
        file.create_dir(self.output_dir)

        reasoners = self._reasoners
        res: pd.DataFrame = self.results_grouped_by_reasoner(drop_missing=False).first()[reasoners]
        res.fillna(Status.UNKNOWN, inplace=True)
        res = self.strategy.evaluate_dataframe(res)
        csv.write(res, path.join(self.output_dir, 'correct.csv'))

        results = [res[r].value_counts(sort=False) for r in reasoners]

        summary = pd.DataFrame({
            'reasoner': reasoners
        }).set_index('reasoner')
        self.summary = summary

        for s in Status.ALL:
            summary[s] = np.array([r.get(s, default=0) for r in results])

        summary['ratio'] = summary[Status.OK] / summary[Status.KNOWN].sum(axis=1)
        summary.rename(columns={
            Status.OK: 'correct',
            Status.INCORRECT: 'incorrect'
        }, inplace=True)

        csv.write(summary, path.join(self.output_dir, 'summary.csv'))
