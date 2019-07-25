import sys
from os import path


DEBUG = False


class Paths:
    """Paths config namespace."""
    SRC_DIR = path.dirname(path.realpath(sys.argv[0]))
    ROOT_DIR = path.dirname(SRC_DIR)
    BIN_DIR = path.join(ROOT_DIR, 'bin')
    DATA_DIR = path.join(ROOT_DIR, 'data')
    RESULTS_DIR = path.join(ROOT_DIR, 'results')
    REASONERS_DIR = path.join(SRC_DIR, 'reasoner', 'reasoners')

    LOG_FILE_NAME = 'log.txt'
    RESULTS_FILE_NAME = 'results.csv'
    CONFIG_FILE_NAME = 'config.json'

    OWLTOOL = path.join(BIN_DIR, 'OWLTool', 'owltool.jar')


class OWLTool:
    """OWLTool config namespace."""
    PATH = Paths.OWLTOOL
    VM_OPTS = ['-Xmx16g', '-DentityExpansionLimit=1000000000']


class ConfigKey:
    """Test config dictionary keys."""
    DATASET = 'dataset'
    REASONERS = 'reasoners'
    NAME = 'name'
    SYNTAX = 'syntax'


class Test:
    """Test config namespace."""
    TIMEOUT = 1200.0
    DEFAULT_ITERATIONS = 5
