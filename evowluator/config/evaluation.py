from __future__ import annotations

from datetime import datetime
from os import path
from typing import Dict, List

from pyutils.io import file
from pyutils.proc.energy import EnergyProbe
from pyutils.types.unit import MemoryUnit
from .key import ConfigKey
from .paths import Paths
from ..data import json
from ..data.dataset import Dataset, Ontology
from ..data.info import DatasetInfo
from ..data.syntax import Syntax
from ..data.sort_by import SortBy
from ..evaluation.mode import EvaluationMode
from ..reasoner.base import Reasoner
from ..reasoner.task import ReasoningTask
from ..util.math import evaluate_expression
from ..visualization.correctness import CorrectnessStrategy


class Evaluation:
    """Evaluation config namespace."""
    WORK_DIR = ''
    TEMP_DIR = ''
    TASK: ReasoningTask | None = None
    MODE = EvaluationMode.CORRECTNESS
    DATASET: Dataset | None = None
    REASONERS: List[Reasoner] = []
    TIMEOUT: float | str = 1800.0
    ITERATIONS = 1
    MAX_WORKERS: int | None = None
    CORRECTNESS_STRATEGY = CorrectnessStrategy.default()
    CORRECTNESS_RESULTS: str | None = None
    ENERGY_PROBES: List[EnergyProbe] = []
    ENERGY_PROBES_ATTRS = {
        'powertop': { 'interval': 1000 },
        'powermetrics': { 'interval': 500 },
        'random': { 'interval': 500, 'min': 0.1, 'max': 0.9 },
        'zero': {'interval': 500},
    }

    @classmethod
    def work_dir(cls) -> str:
        if not cls.WORK_DIR:
            cls.WORK_DIR = Paths.new_results_dir(cls.name())
            file.chmod(cls.WORK_DIR, 0o644, recursive=True, dir_mode=0o755)
        return cls.WORK_DIR

    @classmethod
    def config_file(cls) -> str:
        return path.join(Evaluation.work_dir(), Paths.CONFIG_FILE_NAME)

    @classmethod
    def results_file(cls) -> str:
        return path.join(Evaluation.work_dir(), Paths.RESULTS_FILE_NAME)

    @classmethod
    def log_file(cls) -> str:
        return path.join(Evaluation.work_dir(), Paths.LOG_FILE_NAME)

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

    _USABLE_REASONERS: List[Reasoner] = None
    @classmethod
    def usable_reasoners(cls) -> List[Reasoner]:
        if cls._USABLE_REASONERS is None:
            dataset = cls.dataset()
            cls._USABLE_REASONERS = [r for r in cls.reasoners() if dataset.syntaxes_for_reasoner(r)]
        return cls._USABLE_REASONERS

    _USED_SYNTAXES: List[Syntax] = None
    @classmethod
    def used_syntaxes(cls) -> List[Syntax]:
        if cls._USED_SYNTAXES is None:
            dataset = cls.dataset()
            syntaxes = {dataset.syntax_for_reasoner(r) for r in cls.usable_reasoners()}
            cls._USED_SYNTAXES = list(syntaxes)
        return cls._USED_SYNTAXES

    @classmethod
    def timeout(cls) -> str | float:
        return cls.TIMEOUT

    @classmethod
    def timeout_seconds(cls, inputs: List[Ontology]) -> float | None:
        timeout = Evaluation.timeout()
        try:
            timeout = float(timeout)
        except ValueError:
            # Not a float, could be an expression
            size_mb = MemoryUnit.B(sum(i.size for i in inputs)).to_value(MemoryUnit.MB)
            expression = timeout.replace('s', f'{size_mb:.2f}')
            timeout = evaluate_expression(expression)
        return timeout if timeout else None

    @classmethod
    def iterations(cls) -> int:
        return cls.ITERATIONS

    @classmethod
    def max_workers(cls) -> int | None:
        return cls.MAX_WORKERS

    @classmethod
    def correctness_strategy(cls) -> CorrectnessStrategy:
        return cls.CORRECTNESS_STRATEGY

    @classmethod
    def correctness_results(cls) -> str | None:
        return cls.CORRECTNESS_RESULTS

    @classmethod
    def energy_probes(cls) -> List[EnergyProbe]:
        return cls.ENERGY_PROBES

    _FIELDS: List[str] = None
    @classmethod
    def fields(cls) -> List[str]:
        if cls._FIELDS is None:
            if cls.mode() == EvaluationMode.PERFORMANCE:
                cls._FIELDS =  cls.task().performance_fields + [p.name for p in cls.energy_probes()]
            else:
                cls._FIELDS = ['output']
        return cls._FIELDS

    @classmethod
    def syntax_for_reasoner(cls, reasoner: Reasoner) -> Syntax:
        return cls.dataset().syntax_for_reasoner(reasoner)

    @classmethod
    def syntaxes_for_reasoner(cls, reasoner: Reasoner) -> List[Syntax]:
        return cls.dataset().syntaxes_for_reasoner(reasoner)

    @classmethod
    def output_path_for_reasoner(cls, reasoner: Reasoner) -> str:
        return path.join(cls.temp_dir(), reasoner.name.lower().replace(' ', '_'))

    @classmethod
    def new_config(cls) -> Dict:
        task = cls.task()
        data = cls.dataset()
        mode = cls.mode()

        cfg = {
            ConfigKey.TASK: task.name,
            ConfigKey.MODE: mode,
            ConfigKey.FIELDS: cls.fields(),
            ConfigKey.TIMEOUT: cls.timeout(),
            ConfigKey.ITERATIONS: cls.iterations(),
            ConfigKey.START: _cur_datetime_string(),
            ConfigKey.REASONERS: [{
                ConfigKey.NAME: r.name,
                ConfigKey.SYNTAX: data.syntax_for_reasoner(r)
            } for r in cls.usable_reasoners()],
            ConfigKey.DATASET: DatasetInfo.with_dataset(data).to_dict(cls.used_syntaxes()),
        }

        opt_cfg = {}

        if cls.max_workers():
            opt_cfg[ConfigKey.MAX_WORKERS] = cls.max_workers()

        if mode == EvaluationMode.PERFORMANCE:
            if cls.correctness_results():
                opt_cfg[ConfigKey.CORRECTNESS_STRATEGY] = cls.correctness_strategy().name
                opt_cfg[ConfigKey.CORRECTNESS_RESULTS] = cls.correctness_results()
            if cls.energy_probes():
                opt_cfg[ConfigKey.ENERGY_PROBES] = [
                    {ConfigKey.NAME: p.name, ConfigKey.POLLING_INTERVAL: p.interval}
                    for p in cls.energy_probes()
                ]
        else:
            opt_cfg[ConfigKey.CORRECTNESS_STRATEGY] = cls.correctness_strategy().name

        return _add_after(cfg, opt_cfg, ConfigKey.ITERATIONS)

    @classmethod
    def save_config(cls, cfg: Dict, end=False) -> None:
        if end:
            cfg = _add_after(cfg, { ConfigKey.END: _cur_datetime_string() }, ConfigKey.START)
        json.save(cfg, cls.config_file())

    @classmethod
    def load_config(cls) -> Dict:
        cfg = json.load(cls.config_file())
        del(cfg[ConfigKey.END])

        cls.TASK = ReasoningTask.with_name(cfg[ConfigKey.TASK])
        cls.MODE = EvaluationMode(cfg[ConfigKey.MODE])
        cls.REASONERS = Reasoner.with_names([c[ConfigKey.NAME] for c in cfg[ConfigKey.REASONERS]])
        cls.TIMEOUT = cfg[ConfigKey.TIMEOUT]
        cls.ITERATIONS = cfg[ConfigKey.ITERATIONS] if cls.MODE == EvaluationMode.PERFORMANCE else 1
        cls.MAX_WORKERS = cfg.get(ConfigKey.MAX_WORKERS)
        cls.CORRECTNESS_RESULTS = cfg.get(ConfigKey.CORRECTNESS_RESULTS)

        temp = cfg.get(ConfigKey.CORRECTNESS_STRATEGY)
        if temp:
            reasoners = [r.name for r in Evaluation.reasoners()]
            cls.CORRECTNESS_STRATEGY = CorrectnessStrategy.with_name(temp, reasoners)

        temp = cfg[ConfigKey.DATASET]
        cls.DATASET = Dataset(temp[ConfigKey.NAME])
        cls.DATASET.sort_by = SortBy(temp[ConfigKey.SORT_BY])
        cls.DATASET.preferred_syntax = temp.get(ConfigKey.SYNTAX)

        temp = cfg.get(ConfigKey.ENERGY_PROBES)
        if temp:
            def get_probe(probe_cfg: Dict) -> EnergyProbe:
                probe = EnergyProbe.with_name(probe_cfg[ConfigKey.NAME])
                probe.interval = probe_cfg[ConfigKey.POLLING_INTERVAL]
                return probe
            Evaluation.ENERGY_PROBES = [get_probe(n) for n in temp]
        return cfg


def _cur_datetime_string() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _add_after(dictionary: Dict, other: Dict, after_key) -> Dict:
    new_dictionary = {}
    if after_key in dictionary:
        for k, v in dictionary.items():
            new_dictionary[k] = v
            if k == after_key:
                new_dictionary.update(other)
    else:
        new_dictionary.update(dictionary)
        new_dictionary.update(other)
    return new_dictionary
