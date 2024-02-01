import os
import re
import sys
import tempfile
import time
from os import environ, path

from pyutils.io import file

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