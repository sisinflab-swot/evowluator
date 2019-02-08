import json


def save(data, path: str) -> None:
    with open(path, mode='w') as out_file:
        json.dump(data, out_file)


def load(path: str):
    with open(path, mode='r') as in_file:
        return json.load(in_file)
