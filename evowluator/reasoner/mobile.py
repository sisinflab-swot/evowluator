import errno
from abc import ABC, abstractmethod
from typing import List, Optional

from evowluator.pyutils import exc
from evowluator.pyutils.proc import Task, find_executable
from evowluator.test.test_mode import TestMode
from .base import (
    ClassificationOutputFormat,
    MetaArgs,
    Reasoner,
    ReasoningStats,
    ReasoningTask
)


class MobileReasonerIOS(Reasoner, ABC):
    """iOS mobile reasoner wrapper."""

    # Overrides

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def project(self) -> str:
        """Xcode project path."""
        pass

    @property
    @abstractmethod
    def scheme(self) -> str:
        """Xcode scheme for the test."""
        pass

    @abstractmethod
    def test_name_for_task(self, test: str) -> str:
        """
        Override this method by returning the Xcode test name for the specified reasoning task.
        """
        pass

    # Public

    @property
    def path(self):
        return find_executable('xcodebuild')

    @property
    def classification_output_format(self):
        return ClassificationOutputFormat.TEXT

    def args(self, task: str, mode: str) -> List[str]:
        args = ['-project', self.project,
                '-scheme', self.scheme,
                '-destination', 'platform=iOS,name={}'.format(self._detect_connected_device()),
                '-only-testing:{}'.format(self.test_name_for_task(task)),
                'test-without-building',
                'RESOURCE={}'.format(MetaArgs.INPUT)]

        if task == ReasoningTask.MATCHMAKING:
            args.append('REQUEST={}'.format(MetaArgs.REQUEST))

        return args

    def classify(self,
                 input_file: str,
                 output_file: Optional[str] = None,
                 timeout: Optional[float] = None,
                 mode: str = TestMode.CORRECTNESS) -> ReasoningStats:
        exc.raise_if_not_found(input_file, file_type=exc.FileType.FILE)

        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CLASSIFICATION, mode=mode),
                                input_arg=input_file)
        task = self._run(args=args, timeout=timeout, mode=mode)
        return self.results_parser.parse_classification_results(task)

    # Protected

    def _run(self, args: List[str], timeout: Optional[float], mode: str) -> Task:
        task = Task(self.absolute_path, args=args)
        task.run(timeout=timeout)
        return task

    def _detect_connected_device(self) -> str:
        """Returns the name of a connected device."""
        task = Task('instruments', args=['-s', 'devices'])
        task.run()

        for line in task.stdout.splitlines():
            components = line.split(' (', 1)

            if len(components) == 2 and not components[1].endswith('(Simulator)'):
                return components[0]

        exc.raise_ioerror(errno.ENODEV, message='No connected devices.')
