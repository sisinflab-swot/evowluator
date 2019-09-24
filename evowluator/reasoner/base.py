from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from pyutils import exc
from pyutils.io import fileutils
from pyutils.proc.bench import Benchmark, EnergyProfiler, EnergyProbe
from pyutils.proc.task import Task

from evowluator.config import Paths
from evowluator.data.ontology import Ontology
from evowluator.evaluation.mode import EvaluationMode
from evowluator.util import owltool
from evowluator.util.strenum import StrEnum
from .results import MatchmakingResults, ReasoningResults, ResultsParser


class ReasoningTask(StrEnum):
    """Reasoning tasks."""
    CLASSIFICATION = 'classification'
    CONSISTENCY = 'consistency'
    MATCHMAKING = 'matchmaking'

    @classmethod
    def standard(cls) -> List[ReasoningTask]:
        return [cls.CLASSIFICATION, cls.CONSISTENCY]


class MetaArgs:
    """Meta-arguments."""
    INPUT = '<input_meta_arg>'
    OUTPUT = '<output_meta_arg>'
    REQUEST = '<request_meta_arg>'

    @staticmethod
    def replace(args: List[str], input_arg: str,
                output_arg: Optional[str] = None, request_arg: Optional[str] = None) -> List[str]:
        """Replace meta-args with actual ones."""
        replacements = [
            (MetaArgs.INPUT, input_arg),
            (MetaArgs.OUTPUT, output_arg),
            (MetaArgs.REQUEST, request_arg)
        ]

        for meta_arg, replacement in replacements:
            if replacement is not None:
                args = [arg.replace(meta_arg, replacement) for arg in args]

        return args


class OutputFormat(StrEnum):
    """Output format."""
    TEXT = 'text'
    ONTOLOGY = 'ontology'


class Reasoner(ABC):
    """Abstract reasoner interface."""

    # Override

    @classmethod
    def is_template(cls) -> bool:
        """
        If you return True, this class is ignored by the reasoner loader,
        allowing its use as a template.
        """
        return cls == Reasoner

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the reasoner."""
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        """Path of the reasoner executable."""
        pass

    @property
    def supported_syntaxes(self) -> List[Ontology.Syntax]:
        """OWL syntaxes supported by the reasoner."""
        return Ontology.Syntax.all()

    @property
    def supported_tasks(self) -> List[ReasoningTask]:
        """Reasoning tasks supported by the reasoner."""
        return ReasoningTask.standard()

    @property
    def preferred_syntax(self) -> Ontology.Syntax:
        """Default syntax used by the reasoner."""
        return self.supported_syntaxes[0]

    @property
    def classification_output_format(self) -> OutputFormat:
        """Output format of the classification task."""
        return OutputFormat.ONTOLOGY

    @property
    def results_parser(self) -> ResultsParser:
        """Results parser instance."""
        return ResultsParser()

    @abstractmethod
    def args(self, task: ReasoningTask, mode: EvaluationMode) -> List[str]:
        """Args to be passed to the reasoner executable for each task and evaluation mode."""
        pass

    # Public

    def __init__(self) -> None:
        self.energy_probe: Optional[EnergyProbe] = None

    def perform_task(self, task: ReasoningTask, input_file: Union[str, Tuple[str, str]],
                     output_file: Optional[str] = None, timeout: Optional[float] = None,
                     mode: EvaluationMode = EvaluationMode.CORRECTNESS) -> ReasoningResults:
        """Performs the specified reasoning task."""
        if task == ReasoningTask.CLASSIFICATION:
            return self.classify(input_file, output_file=output_file, timeout=timeout, mode=mode)
        elif task == ReasoningTask.CONSISTENCY:
            return self.consistency(input_file, timeout=timeout, mode=mode)
        else:
            return self.matchmaking(input_file[0], input_file[1], timeout=timeout, mode=mode)

    def classify(self,
                 input_file: str,
                 output_file: Optional[str] = None,
                 timeout: Optional[float] = None,
                 mode: EvaluationMode = EvaluationMode.CORRECTNESS) -> ReasoningResults:
        """Runs the classification reasoning task."""
        exc.raise_if_not_found(input_file, file_type=exc.FileType.FILE)

        classification_out = None
        use_owl_tool = self.classification_output_format == OutputFormat.ONTOLOGY

        if output_file:
            if use_owl_tool:
                classification_out = os.path.splitext(output_file)[0]
            else:
                classification_out = output_file
            fileutils.remove(output_file)
            fileutils.remove(classification_out)

        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CLASSIFICATION, mode=mode),
                                input_arg=input_file,
                                output_arg=classification_out)

        task = self._run(args=args, timeout=timeout, mode=mode)

        if mode == EvaluationMode.CORRECTNESS and use_owl_tool:
            owltool.print_tbox(classification_out, output_file)

        results = self.results_parser.parse_classification_results(task)
        return results.with_output(output_file, is_file=True)

    def consistency(self,
                    input_file: str,
                    timeout: Optional[float] = None,
                    mode: EvaluationMode = EvaluationMode.CORRECTNESS) -> ReasoningResults:
        """Checks if the given ontology is consistent."""
        exc.raise_if_not_found(input_file, file_type=exc.FileType.FILE)

        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CONSISTENCY, mode=mode),
                                input_arg=input_file)

        task = self._run(args, timeout=timeout, mode=mode)
        return self.results_parser.parse_consistency_results(task)

    def matchmaking(self,
                    resource_file: str,
                    request_file: str,
                    output_file: Optional[str] = None,
                    timeout: Optional[float] = None,
                    mode: EvaluationMode = EvaluationMode.CORRECTNESS) -> MatchmakingResults:
        """Runs abductions or contractions between all resource and request individuals."""
        exc.raise_if_not_found(resource_file, file_type=exc.FileType.FILE)
        exc.raise_if_not_found(request_file, file_type=exc.FileType.FILE)

        if output_file:
            fileutils.remove(output_file)

        args = MetaArgs.replace(args=self.args(task=ReasoningTask.MATCHMAKING, mode=mode),
                                input_arg=resource_file, request_arg=request_file,
                                output_arg=output_file)

        task = self._run(args, timeout=timeout, mode=mode)

        results = self.results_parser.parse_matchmaking_results(task)
        return results.with_output(output_file, is_file=True)

    # Protected

    def _absolute_path(self, path: str) -> str:
        """Absolute path for the specified relative path."""
        path = os.path.normpath(path)
        return path if os.path.isabs(path) else os.path.join(Paths.BIN_DIR, path)

    def _task(self, args: List[str]) -> Task:
        return Task(self._absolute_path(self.path), args=args)

    def _run(self, args: List[str], timeout: Optional[float], mode: EvaluationMode) -> Task:
        """Runs the reasoner."""
        exc.raise_if_not_found(self._absolute_path(self.path), file_type=exc.FileType.FILE)
        task = self._task(args)

        if mode == EvaluationMode.PERFORMANCE:
            task = Benchmark(task)
        elif mode == EvaluationMode.ENERGY:
            task = EnergyProfiler(task, self.energy_probe, sampling_interval=500)

        task.run(timeout=timeout)
        task.raise_if_failed(message='{} exited with code: {}'.format(self.name, task.exit_code))

        return task
