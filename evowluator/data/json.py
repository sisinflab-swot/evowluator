import json
import sys
from io import StringIO


def dump(data, file: StringIO = sys.stdout) -> None:
    json.dump(data, file, indent=2)
    file.write('\n')


def save(data, path: str) -> None:
    with open(path, mode='w') as out_file:
        json.dump(data, out_file, indent=2)


def load(path: str):
    with open(path, mode='r') as in_file:
        return json.load(in_file)
