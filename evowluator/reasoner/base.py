import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from pyutils import exc
from pyutils.io import fileutils
from pyutils.proc.bench import Benchmark, EnergyProfiler, PowermetricsProbe
from pyutils.proc.task import Jar, Task, OutputAction

from evowluator.config import OWLTool, Paths
from evowluator.data.ontology import Ontology
from evowluator.test.test_mode import TestMode
from .results import MatchmakingResults, ReasoningResults, ResultsParser


class ReasoningTask:
    """Reasoning tasks namespace."""
    CLASSIFICATION = 'classification'
    CONSISTENCY = 'consistency'
    MATCHMAKING = 'matchmaking'

    STANDARD = [CLASSIFICATION, CONSISTENCY]
    ALL = [CLASSIFICATION, CONSISTENCY, MATCHMAKING]


class MetaArgs:
    """Meta-arguments namespace."""
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

    def __init__(self) -> None:
        exc.raise_if_not_found(self._absolute_path(self.path), file_type=exc.FileType.FILE)

    def perform_task(self, task: str, input_file: Union[str, Tuple[str, str]],
                     output_file: Optional[str] = None, timeout: Optional[float] = None,
                     mode: str = TestMode.CORRECTNESS) -> ReasoningResults:
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
                 mode: str = TestMode.CORRECTNESS) -> ReasoningResults:
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

        results = self.results_parser.parse_classification_results(task)
        return results.with_output(output_file, is_file=True)

    def consistency(self,
                    input_file: str,
                    timeout: Optional[float] = None,
                    mode: str = TestMode.CORRECTNESS) -> ReasoningResults:
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
                    mode: str = TestMode.CORRECTNESS) -> MatchmakingResults:
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

    # Protected methods

    def _absolute_path(self, path: str) -> str:
        """Absolute path for the specified relative path."""
        path = os.path.normpath(path)
        return path if os.path.isabs(path) else os.path.join(Paths.BIN_DIR, path)

    def _run(self, args: List[str], timeout: Optional[float], mode: str) -> Task:
        """Runs the reasoner."""
        task = Task(self._absolute_path(self.path), args=args)

        if mode == TestMode.PERFORMANCE:
            task = Benchmark(task)
        elif mode == TestMode.ENERGY:
            task = EnergyProfiler(task, PowermetricsProbe(), sampling_interval=500)

        task.run(timeout=timeout)
        return task
