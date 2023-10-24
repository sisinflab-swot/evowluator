from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import traceback
from typing import List
from os import environ, path

from pyutils.io import file
from pyutils.proc.energy import EnergyProbe
from pyutils.types.strenum import StrEnum
from .evaluation.mode import EvaluationMode

EXE_NAME = environ.get('EVOWLUATOR_EXE', path.basename(sys.argv[0]))


class OnError(StrEnum):
    """Behavior in case of errors."""
    IGNORE = 'ignore'
    LOG = 'log'
    PAUSE = 'pause'
    ABORT = 'abort'


class Debug:
    """Debug config namespace."""
    ON_ERROR = OnError.LOG
    TRACE = False

    @classmethod
    def format(cls, e: Exception) -> str:
        return ''.join(traceback.format_exception(e)) if cls.TRACE else str(e)


class Paths:
    """Paths config namespace."""
    SRC_DIR = path.dirname(path.realpath(sys.argv[0]))
    ROOT_DIR = path.dirname(SRC_DIR)
    BIN_DIR = path.join(ROOT_DIR, 'bin')
    DATA_DIR = path.join(ROOT_DIR, 'data')
    LIB_DIR = path.join(ROOT_DIR, 'lib')
    RESULTS_DIR = path.join(ROOT_DIR, 'results')

    LOG_FILE_NAME = 'log.txt'
    RESULTS_FILE_NAME = 'results.csv'
    CONFIG_FILE_NAME = 'config.json'

    OWLTOOL = path.join(BIN_DIR, 'OWLTool', 'owltool.jar')

    @staticmethod
    def last_results_dir() -> str:
        try:
            return max((p for p in (path.join(Paths.RESULTS_DIR, d)
                                    for d in os.listdir(Paths.RESULTS_DIR))
                        if path.isdir(p)),
                       key=path.getmtime, default=None)
        except FileNotFoundError:
            raise FileNotFoundError('No available results')

    @staticmethod
    def new_results_dir(name: str) -> str:
        name = re.sub(r"[^\w\s]", '', name)
        name = re.sub(r"\s+", '_', name)
        prefix = time.strftime(f'{name}_%Y%m%d_%H%M%S_')
        file.create_dir(Paths.RESULTS_DIR)
        return tempfile.mkdtemp(dir=Paths.RESULTS_DIR, prefix=prefix)

    @staticmethod
    def absolute(rel_path: str, base: str = BIN_DIR) -> str:
        """Absolute path for the specified relative path."""
        rel_path = path.normpath(rel_path)
        return rel_path if path.isabs(rel_path) else path.join(base, rel_path)


class OWLTool:
    """OWLTool config namespace."""
    PATH = Paths.OWLTOOL
    VM_OPTS = ['-Xmx16g', '-DentityExpansionLimit=1000000000']


class ConfigKey:
    """Evaluation config dictionary keys."""
    DATASET = 'dataset'
    ENERGY_PROBES = 'energy_probes'
    FIELDS = 'fields'
    ITERATIONS = 'iterations'
    MODE = 'mode'
    NAME = 'name'
    ONTOLOGIES = 'ontologies'
    POLLING_INTERVAL = 'polling_interval'
    REASONERS = 'reasoners'
    SIZE = 'size'
    SYNTAX = 'syntax'
    TASK = 'task'
    TIMEOUT = 'timeout'
    START = 'start'
    END = 'end'


class Evaluation:
    """Evaluation config namespace."""
    MODE = EvaluationMode.CORRECTNESS
    TIMEOUT = 1800.0
    ITERATIONS = 1
    ENERGY_PROBES: List[EnergyProbe] = []
    ENERGY_POLLING_INTERVALS = {
        'powertop': 1000,
        'powermetrics': 500
    }


class User:
    """User config namespace."""
    BASE_PACKAGE = 'evowluator.user'
    PACKAGES = [f'{BASE_PACKAGE}.probes', f'{BASE_PACKAGE}.reasoners', f'{BASE_PACKAGE}.tasks']
