from sys import stdout

from pyutils.io.file import readable_bytes
from pyutils.io.pretty_printer import PrettyPrinter
from pyutils.proc.bench import EnergyProbe
from ..data.dataset import Dataset
from ..reasoner.base import Reasoner
from ..reasoner.task import ReasoningTask


def general() -> None:
    log = PrettyPrinter(stdout)

    log.green('Datasets', underline='-')
    for d in Dataset.all():
        stats = d.cumulative_stats()
        log.yellow(f'{d.name}: ', endl=False)
        log(f'{stats[0]} ontologies, {readable_bytes(stats[1])}')

    log.spacer(2)
    log.green('Reasoning tasks', underline='-')
    for t in ReasoningTask.all():
        log.yellow(t.name)

    log.spacer(2)
    log.green('Reasoners', underline='-')
    for r in Reasoner.all():
        log.yellow(r.name)
        with log.indent:
            log.yellow('Tasks: ', endl=False)
            log(', '.join(r.name for r in r.supported_tasks))
            log.yellow('Syntaxes: ', endl=False)
            log(', '.join(r.supported_syntaxes))
            log.spacer(2)

    log.spacer(2)
    log.green('Energy probes', underline='-')
    for p in EnergyProbe.all():
        log.yellow(p.name)


def dataset(name: str) -> None:
    log = PrettyPrinter(stdout)
    data = Dataset.with_name(name)

    log.green(f'{data.name} dataset', underline='-')
    stats = data.cumulative_stats()
    log.yellow(f'Total size: ', endl=False)
    log(f'{stats[0]} ontologies, {readable_bytes(stats[1])}')
    log.yellow('Syntaxes')
    with log.indent:
        for s in sorted(data.syntaxes):
            log.yellow(f'{s}: ', endl=False)
            log(readable_bytes(data.cumulative_size((s,))))

    log.spacer(2)
    log.green('Ontologies', underline='-')
    for e in data.get_entries():
        log.yellow(f'{e.name}')
        with log.indent:
            for s in sorted(data.syntaxes):
                log.yellow(f'{s}: ', endl=False)
                log(readable_bytes(e.cumulative_size((s,))))
        log.spacer(2)
