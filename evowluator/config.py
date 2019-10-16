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
    USER_DIR = path.join(SRC_DIR, 'user')
    REASONERS_DIR = path.join(USER_DIR, 'reasoners')
    PROBES_DIR = path.join(USER_DIR, 'probes')

    LOG_FILE_NAME = 'log.txt'
    RESULTS_FILE_NAME = 'results.csv'
    CONFIG_FILE_NAME = 'config.json'

    OWLTOOL = path.join(BIN_DIR, 'OWLTool', 'owltool.jar')


class OWLTool:
    """OWLTool config namespace."""
    PATH = Paths.OWLTOOL
    VM_OPTS = ['-Xmx16g', '-DentityExpansionLimit=1000000000']


class ConfigKey:
    """Evaluation config dictionary keys."""
    DATASET = 'dataset'
    REASONERS = 'reasoners'
    NAME = 'name'
    SYNTAX = 'syntax'


class Evaluation:
    """Evaluation config namespace."""
    TIMEOUT = 1200.0
    ITERATIONS = 5
