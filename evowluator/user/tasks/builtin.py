from __future__ import annotations

import re

from pyutils.io import file
from pyutils.proc.task import Task
from ...reasoner.results import Output, Results
from ...reasoner.task import ReasoningTask
from ...config.evaluation import Evaluation, EvaluationMode
from ...util import owltool


class ClassificationTask(ReasoningTask):
    """Ontology classification reasoning task."""

    @property
    def expected_output_format(self) -> Output.Format:
        return Output.Format.ONTOLOGY

    def process_results(self, results: Results, task: Task) -> Results:
        super().process_results(results, task)

        if Evaluation.mode() != EvaluationMode.CORRECTNESS:
            return results

        if results.output.format == Output.Format.ONTOLOGY:
            owltool.print_taxonomy(results.output.path, results.output.path)
            results.output.format = Output.Format.TEXT

        return results


class ConsistencyTask(ReasoningTask):
    """Ontology consistency reasoning task."""

    @property
    def expected_output_format(self) -> Output.Format:
        return Output.Format.STRING

    def process_results(self, results: Results, task: Task) -> Results:
        super().process_results(results, task)

        if Evaluation.mode() != EvaluationMode.CORRECTNESS:
            return results

        if results.output.format == Output.Format.ONTOLOGY:
            # TODO: use owltool to detect collapsed taxonomies.
            raise ValueError('Unsupported output format.')
        elif results.output.format == Output.Format.TEXT:
            output = file.contents(results.output.path)
        else:
            output = results.output.data

        if re.search(r'(not |in)consistent', output, re.IGNORECASE):
            output = 'not consistent'
        elif re.search(r'consistent', output, re.IGNORECASE):
            output = 'consistent'
        else:
            output = 'unknown'

        results.output = Output(output, Output.Format.STRING)
        return results


class MatchmakingTask(ReasoningTask):
    """Matchmaking reasoning task."""

    @property
    def requires_additional_inputs(self) -> bool:
        return True
