from evowluator.data.ontology import Ontology

from .base import Test


class InfoTest(Test):
    """Dataset and reasoner info test."""

    @property
    def name(self):
        return 'info'

    def setup(self):
        row = ['Ontology'] + ['Size ({})'.format(s) for s in Ontology.Syntax.ALL]
        self._csv_writer.write_row(row)

    def run(self, entry):
        self._csv_writer.write_row([entry.name] + [str(o.size) for o in entry.ontologies()])
