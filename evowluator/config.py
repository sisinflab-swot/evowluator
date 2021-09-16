import os
import sys
from os import environ, path


DEBUG = False
EXE_NAME = environ.get('EVOWLUATOR_EXE', path.basename(sys.argv[0]))


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
    def absolute(rel_path: str) -> str:
        """Absolute path for the specified relative path."""
        rel_path = path.normpath(rel_path)
        return rel_path if path.isabs(rel_path) else path.join(Paths.BIN_DIR, rel_path)


class OWLTool:
    """OWLTool config namespace."""
    PATH = Paths.OWLTOOL
    VM_OPTS = ['-Xmx16g', '-DentityExpansionLimit=1000000000']


class ConfigKey:
    """Evaluation config dictionary keys."""
    DATASET = 'dataset'
    NAME = 'name'
    ONTOLOGIES = 'ontologies'
    REASONERS = 'reasoners'
    SIZE = 'size'
    SYNTAX = 'syntax'


class Evaluation:
    """Evaluation config namespace."""
    TIMEOUT = 1200.0
    ITERATIONS = 5


class User:
    """User config namespace."""
    BASE_PACKAGE = 'evowluator.user'
    PACKAGES = [f'{BASE_PACKAGE}.probes', f'{BASE_PACKAGE}.reasoners']
