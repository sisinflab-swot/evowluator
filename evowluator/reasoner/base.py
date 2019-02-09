import os
from abc import ABC, abstractmethod
from typing import List, Optional

from evowluator.config import OWLTool, Paths
from evowluator.data.ontology import Ontology
from evowluator.pyutils import exc, fileutils
from evowluator.pyutils.proc import Benchmark, EnergyProfiler, Jar, OutputAction, Task
from evowluator.test.enum import TestMode
from .results import ConsistencyResults, MatchmakingResults, ReasoningStats, ResultsParser


class ReasoningTask:
    """Reasoning tasks namespace."""
    CLASSIFICATION = 'classification'
    CONSISTENCY = 'consistency'
    NON_STANDARD = 'non-standard'

    STANDARD = [CLASSIFICATION, CONSISTENCY]
    ALL = [CLASSIFICATION, CONSISTENCY, NON_STANDARD]


class MetaArgs:
    """Meta-arguments namespace."""
    INPUT = '<input_meta_arg>'
    OUTPUT = '<output_meta_arg>'
    REQUEST = '<request_meta_arg>'

    @staticmethod
    def replace(args: List[str], input_arg: str,
                output_arg: Optional[str] = None, request_arg: Optional[str] = None) -> List[str]:
        """Replace meta-args with actual ones."""
        replacements = {
            MetaArgs.INPUT: input_arg,
            MetaArgs.OUTPUT: output_arg,
            MetaArgs.REQUEST: request_arg
        }

        return [replacements.get(arg, arg) for arg in args]


class ClassificationOutputFormat:
    """Classification output format namespace."""
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
    def supported_syntaxes(self) -> List[str]:
        """OWL syntaxes supported by the reasoner."""
        return Ontology.Syntax.ALL

    @property
    def supported_tasks(self) -> List[str]:
        """Reasoning tasks supported by the reasoner."""
        return ReasoningTask.STANDARD

    @property
    def preferred_syntax(self) -> str:
        """Default syntax used by the reasoner."""
        return self.supported_syntaxes[0]

    @property
    def classification_output_format(self) -> str:
        """Output format of the classification task."""
        return ClassificationOutputFormat.TEXT

    @property
    def results_parser(self) -> ResultsParser:
        """Results parser instance."""
        return ResultsParser()

    @abstractmethod
    def args(self, task: str, mode: str) -> List[str]:
        """Args to be passed to the reasoner executable for each task and test mode."""
        pass

    # Public

    @property
    def absolute_path(self) -> str:
        """Absolute path of the reasoner executable."""
        path = os.path.normpath(self.path)
        return path if os.path.isabs(path) else os.path.join(Paths.BIN_DIR, path)

    def __init__(self) -> None:
        exc.raise_if_not_found(self.absolute_path, file_type=exc.FileType.FILE)

    def syntax_for_requested(self, syntax: str) -> str:
        """Returns 'syntax' if it is supported, otherwise returns the preferred syntax."""
        return syntax if syntax in self.supported_syntaxes else self.preferred_syntax

    def classify(self,
                 input_file: str,
                 output_file: Optional[str] = None,
                 timeout: Optional[float] = None,
                 mode: str = TestMode.CORRECTNESS) -> ReasoningStats:
        """Runs the classification reasoning task."""
        exc.raise_if_not_found(input_file, file_type=exc.FileType.FILE)

        classification_out = None
        use_owl_tool = self.classification_output_format == ClassificationOutputFormat.ONTOLOGY

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

        if mode == TestMode.CORRECTNESS and use_owl_tool:
            exc.raise_if_not_found(OWLTool.PATH, file_type=exc.FileType.FILE)
            args = ['print-tbox', '-o', output_file, classification_out]
            jar = Jar(OWLTool.PATH, jar_args=args,
                      vm_opts=OWLTool.VM_OPTS, output_action=OutputAction.DISCARD)
            jar.run()

        return self.results_parser.parse_classification_results(task)

    def consistency(self,
                    input_file: str,
                    timeout: Optional[float] = None,
                    mode: str = TestMode.CORRECTNESS) -> ConsistencyResults:
        """Checks if the given ontology is consistent."""
        exc.raise_if_not_found(input_file, file_type=exc.FileType.FILE)

        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CONSISTENCY, mode=mode),
                                input_arg=input_file)

        task = self._run(args, timeout=timeout, mode=mode)
        return self.results_parser.parse_consistency_results(task)

    def matchmaking(self,
                    resource_file: str,
                    request_file: str,
                    timeout: Optional[float] = None,
                    mode: str = TestMode.CORRECTNESS) -> MatchmakingResults:
        """Runs abductions or contractions between all resource and request individuals."""
        exc.raise_if_not_found(resource_file, file_type=exc.FileType.FILE)
        exc.raise_if_not_found(request_file, file_type=exc.FileType.FILE)

        args = MetaArgs.replace(args=self.args(task=ReasoningTask.NON_STANDARD, mode=mode),
                                input_arg=resource_file,
                                request_arg=request_file)

        task = self._run(args, timeout=timeout, mode=mode)
        return self.results_parser.parse_matchmaking_results(task)

    # Protected methods

    def _run(self, args: List[str], timeout: Optional[float], mode: str) -> Task:
        """Runs the reasoner."""
        task = Task(self.absolute_path, args=args)

        if mode == TestMode.MEMORY:
            task = Benchmark(task)
        elif mode == TestMode.ENERGY:
            task = EnergyProfiler(task, sampling_interval=500)

        task.run(timeout=timeout)
        return task
