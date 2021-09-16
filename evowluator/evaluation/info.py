from .base import Evaluator
from ..data.syntax import Syntax


class InfoEvaluator(Evaluator):
    """Dataset and reasoner info evaluator."""

    @property
    def name(self):
        return 'info'

    def setup(self):
        row = ['Ontology'] + [f'Size ({s.value})' for s in Syntax]
        self._csv_writer.write_row(row)

    def run(self, entry):
        self._csv_writer.write_row([entry.name] + [str(o.size) for o in entry.ontologies()])
