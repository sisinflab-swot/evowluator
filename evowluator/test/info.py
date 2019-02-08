from evowluator.data.ontology import Ontology

from .base import Test
from .enum import TestName


class InfoTest(Test):
    """Dataset and reasoner info test."""

    @property
    def name(self):
        return TestName.Misc.INFO

    @property
    def default_reasoners(self):
        return self._loader.reasoners

    def setup(self):
        row = ['Ontology'] + ['Size ({})'.format(s) for s in Ontology.Syntax.ALL]
        self._csv_writer.write_row(row)

    def run(self, entry):
        self._csv_writer.write_row([entry.name] + [str(o.size) for o in entry.ontologies()])
