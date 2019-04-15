import csv
from typing import List, Optional, TextIO


class _CSVIO:

    # Properties

    @property
    def file_path(self) -> str:
        return self._file_path

    # Public methods

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._file: Optional[TextIO] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        self.close()
        self._file = open(self.file_path, self._open_mode())
        self._on_open()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._on_close()

    # Must override

    def _open_mode(self) -> str:
        raise NotImplementedError

    def _on_open(self) -> None:
        raise NotImplementedError

    def _on_close(self) -> None:
        raise NotImplementedError


class CSVReader(_CSVIO):

    # Public methods

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.__csv_reader: csv.reader = None

    def __iter__(self):
        return (row for row in self.__csv_reader)

    def read_row(self) -> List[str]:
        return next(self.__csv_reader)

    # Overrides

    def _open_mode(self):
        return 'r'

    def _on_open(self):
        self.__csv_reader = csv.reader(self._file)

    def _on_close(self):
        self.__csv_reader = None


class CSVWriter(_CSVIO):

    # Public methods

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.__csv_writer: csv.writer = None

    def write_row(self, row: List[str]) -> None:
        self.__csv_writer.writerow(row)

    # Overrides

    def _open_mode(self):
        return 'w'

    def _on_open(self):
        self.__csv_writer = csv.writer(self._file)

    def _on_close(self):
        self.__csv_writer = None
