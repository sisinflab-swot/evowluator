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
        stats = d.cumulative_stats()
        log.yellow(f'{d.name}')
        with log.indent:
            log.yellow('Size: ', endl=False)
            log(f'{stats[0]} ontologies, {MemoryUnit.B(stats[1]).readable()}')
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
        'datasets': {
            d.name: {
                'count': stats[0],
                'size_bytes': stats[1],
                'size_readable': str(MemoryUnit.B(stats[1]).readable()),
                'syntaxes': sorted(d.syntaxes)
            }
            for d, stats in ((d, d.cumulative_stats()) for d in Dataset.all())
        },
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


def _dataset_text(name: str) -> None:
    log = PrettyPrinter(stdout)
    data = Dataset(name)

    log.green(f'{data.name} dataset', underline='-')
    stats = data.cumulative_stats()
    log.yellow(f'Total size: ', endl=False)
    log(f'{stats[0]} ontologies, {MemoryUnit.B(stats[1]).readable()}')
    log.yellow('Tasks')
    with log.indent:
        for t in data.reasoning_tasks:
            log(t.name)
    log.yellow('Syntaxes')
    with log.indent:
        for s in sorted(data.syntaxes):
            log.yellow(f'{s}: ', endl=False)
            log(MemoryUnit.B(data.cumulative_size((s,))).readable())

    log.spacer(2)
    log.green('Ontologies', underline='-')
    for e in data.get_entries():
        log.yellow(f'{e.name}')
        with log.indent:
            for s in sorted(data.syntaxes):
                log.yellow(f'{s}: ', endl=False)
                log(MemoryUnit.B(e.cumulative_size((s,))).readable())
        log.spacer(2)


def _dataset_json(name: str) -> None:
    data = Dataset(name)
    stats = data.cumulative_stats()
    syntaxes = sorted(data.syntaxes)
    json.dump({
        'count': stats[0],
        'size_bytes': stats[1],
        'size_readable': str(MemoryUnit.B(stats[1]).readable()),
        'tasks': sorted(r.name for r in data.reasoning_tasks),
        'syntaxes': {
            syntax: {
                'size_bytes': size,
                'size_readable': str(MemoryUnit.B(size).readable())
            }
            for syntax, size in ((s, data.cumulative_size((s,))) for s in syntaxes)
        },
        'ontologies': {
            entry.name: {
                'size_bytes': size,
                'size_readable': str(MemoryUnit.B(size).readable()),
                'syntaxes': {
                    syntax: {
                        'size_bytes': syntax_size,
                        'size_readable': str(MemoryUnit.B(syntax_size).readable())
                    }
                    for syntax, syntax_size in ((s, entry.cumulative_size((s,))) for s in syntaxes)
                }
            }
            for entry, size in ((e, e.cumulative_size()) for e in data.get_entries())
        }
    })

def dataset(name: str, json_format: bool = False) -> None:
    if json_format:
        _dataset_json(name)
    else:
        _dataset_text(name)
