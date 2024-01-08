from sys import stdout

from pyutils.io.pretty_printer import PrettyPrinter
from pyutils.proc.energy import EnergyProbe
from pyutils.types.unit import MemoryUnit
from ..data import json
from ..data.dataset import Dataset
from ..reasoner.base import Reasoner
from ..reasoner.task import ReasoningTask


def _general_text() -> None:
    log = PrettyPrinter(stdout)

    log.green('Datasets', underline='-')
    for d in Dataset.all():
        stats = d.stats()
        log.yellow(f'{d.name}')
        with log.indent:
            log.yellow('Size: ', endl=False)
            log(f'{stats.count} ontologies, {stats.size_readable}')
            log.yellow('Syntaxes: ', endl=False)
            log(', '.join(sorted(d.syntaxes)))
            log.spacer(2)

    log.spacer(2)
    log.green('Reasoners', underline='-')
    for r in Reasoner.all():
        log.yellow(r.name)
        with log.indent:
            log.yellow('Tasks: ', endl=False)
            log(', '.join(r.name for r in r.supported_tasks))
            log.yellow('Syntaxes: ', endl=False)
            log(', '.join(sorted(r.supported_syntaxes)))
            log.spacer(2)

    log.spacer(2)
    log.green('Reasoning tasks', underline='-')
    for t in ReasoningTask.all():
        log.yellow(t.name)

    log.spacer(2)
    log.green('Energy probes', underline='-')
    for p in EnergyProbe.all():
        log.yellow(p.name)


def _general_json() -> None:
    json.dump({
        'datasets': { d.name: d.to_json_dict(ontologies=False) for d in Dataset.all() },
        'reasoners': {
            r.name: {
                'tasks': [t.name for t in r.supported_tasks],
                'syntaxes': sorted(r.supported_syntaxes)
            }
            for r in Reasoner.all()
        },
        'reasoning_tasks': [t.name for t in ReasoningTask.all()],
        'energy_probes': [p.name for p in EnergyProbe.all()]
    })


def general(json_format: bool = False) -> None:
    if json_format:
        _general_json()
    else:
        _general_text()


def _dataset_text(data: Dataset) -> None:
    log = PrettyPrinter(stdout)
    log.green(f'{data.name} dataset', underline='-')
    stats = data.stats()
    log.yellow(f'Total size: ', endl=False)
    log(f'{stats.count} ontologies, {stats.size_readable}')
    log.yellow('Tasks')
    with log.indent:
        for t in data.reasoning_tasks:
            log(t.name)
    log.yellow('Syntaxes')
    with log.indent:
        for s in sorted(data.syntaxes):
            log.yellow(f'{s}: ', endl=False)
            syntax_stats = data.stats(s)
            log(syntax_stats.size_readable, endl=False)
            log('' if syntax_stats.count == stats.count else ' (incomplete)')

    log.spacer(2)
    log.green('Ontologies', underline='-')
    for e in data.get_entries():
        log.yellow(f'{e.name}')
        with log.indent:
            for s in sorted(data.syntaxes):
                s_size = e.size(s)
                if s_size:
                    log.yellow(f'{s}: ', endl=False)
                    log(MemoryUnit.B(s_size).readable())
        log.spacer(2)


def _dataset_json(data: Dataset) -> None:
    json.dump(data.to_json_dict(ontologies=True))

def dataset(name: str, json_format: bool = False) -> None:
    data = Dataset(name)
    if json_format:
        _dataset_json(data)
    else:
        _dataset_text(data)
