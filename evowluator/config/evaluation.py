from __future__ import annotations

from os import path
from typing import List

from pyutils.io import file
from pyutils.proc.energy import EnergyProbe
from .paths import Paths
from ..data.dataset import Dataset
from ..data.syntax import Syntax
from ..evaluation.mode import EvaluationMode
from ..reasoner.base import Reasoner
from ..reasoner.task import ReasoningTask


class Evaluation:
    """Evaluation config namespace."""
    WORK_DIR = ''
    TEMP_DIR = ''
    TASK: ReasoningTask | None = None
    MODE = EvaluationMode.CORRECTNESS
    DATASET: Dataset | None = None
    REASONERS: List[Reasoner] = []
    SYNTAX: Syntax | None = None
    TIMEOUT: float | str = 1800.0
    ITERATIONS = 1
    ENERGY_PROBES: List[EnergyProbe] = []
    ENERGY_POLLING_INTERVALS = {
        'powertop': 1000,
        'powermetrics': 500
    }

    @classmethod
    def work_dir(cls) -> str:
        if not cls.WORK_DIR:
            cls.WORK_DIR = Paths.new_results_dir(cls.name())
            file.chmod(cls.WORK_DIR, 0o644, recursive=True, dir_mode=0o755)
        return cls.WORK_DIR

    @classmethod
    def temp_dir(cls) -> str:
        if not cls.TEMP_DIR:
            cls.TEMP_DIR = path.join(cls.work_dir(), 'temp')
            file.create_dir(cls.TEMP_DIR)
        return cls.TEMP_DIR

    @classmethod
    def task(cls) -> ReasoningTask:
        return cls.TASK

    @classmethod
    def mode(cls) -> EvaluationMode:
        return cls.MODE

    @classmethod
    def name(cls) -> str:
        return f'{cls.task()} {cls.mode()}'

    @classmethod
    def dataset(cls) -> Dataset:
        return cls.DATASET

    @classmethod
    def reasoners(cls) -> List[Reasoner]:
        return cls.REASONERS

    @classmethod
    def syntax(cls) -> Syntax | None:
        syntax = cls.SYNTAX
        if syntax and syntax not in cls.dataset().syntaxes:
            msg = f'"{syntax}" syntax not available for "{cls.dataset().name}" dataset.'
            raise ValueError(msg)
        return syntax

    @classmethod
    def timeout(cls) -> str | float:
        return cls.TIMEOUT

    @classmethod
    def iterations(cls) -> int:
        return cls.ITERATIONS

    @classmethod
    def energy_probes(cls) -> List[EnergyProbe]:
        return cls.ENERGY_PROBES

    @classmethod
    def fields(cls) -> List[str]:
        if cls.mode() == EvaluationMode.PERFORMANCE:
            return cls.task().performance_fields + [p.name for p in cls.energy_probes()]
        return ['output']
