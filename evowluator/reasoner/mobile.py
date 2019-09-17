import errno
import os
from abc import ABC, abstractmethod
from typing import List, Optional

from pyutils import exc
from pyutils.proc.task import Task
from pyutils.proc.util import find_executable

from evowluator.test.test_mode import TestMode
from .base import (
    ClassificationOutputFormat,
    MatchmakingResults,
    MetaArgs,
    Reasoner,
    ReasoningResults,
    ReasoningTask
)


class MobileReasoner(Reasoner, ABC):
    """Mobile reasoner wrapper."""

    # Overrides

    @classmethod
    def is_template(cls) -> bool:
        return cls == MobileReasoner

    @property
    def classification_output_format(self):
        return ClassificationOutputFormat.TEXT

    def classify(self,
                 input_file: str,
                 output_file: Optional[str] = None,
                 timeout: Optional[float] = None,
                 mode: str = TestMode.CORRECTNESS) -> ReasoningResults:
        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CLASSIFICATION, mode=mode),
                                input_arg=os.path.basename(input_file))
        task = self._run(args=args, timeout=timeout, mode=mode)
        return self.results_parser.parse_classification_results(task)

    def consistency(self,
                    input_file: str,
                    timeout: Optional[float] = None,
                    mode: str = TestMode.CORRECTNESS) -> ReasoningResults:
        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CONSISTENCY, mode=mode),
                                input_arg=os.path.basename(input_file))
        task = self._run(args, timeout=timeout, mode=mode)
        return self.results_parser.parse_consistency_results(task)

    def matchmaking(self,
                    resource_file: str,
                    request_file: str,
                    output_file: Optional[str] = None,
                    timeout: Optional[float] = None,
                    mode: str = TestMode.CORRECTNESS) -> MatchmakingResults:
        args = MetaArgs.replace(args=self.args(task=ReasoningTask.MATCHMAKING, mode=mode),
                                input_arg=os.path.basename(resource_file),
                                request_arg=os.path.basename(request_file))
        task = self._run(args, timeout=timeout, mode=mode)
        return self.results_parser.parse_matchmaking_results(task)

    def _run(self, args: List[str], timeout: Optional[float], mode: str) -> Task:
        exc.raise_if_not_found(self._absolute_path(self.path), file_type=exc.FileType.FILE)
        task = Task(self._absolute_path(self.path), args=args)
        task.run(timeout=timeout)
        return task


class AndroidReasoner(MobileReasoner, ABC):
    """Android mobile reasoner wrapper."""

    LAUNCHER_CLASSPATH = 'it.poliba.sisinflab.owl.evowluator/.Launcher'

    # Override

    @property
    @abstractmethod
    def target_package(self) -> str:
        """Main reasoner app package."""
        pass

    @property
    @abstractmethod
    def log_prefix(self) -> str:
        """Prefix appended to log messages, used to filter logcat output."""
        pass

    # Overrides

    @classmethod
    def is_template(cls) -> bool:
        return cls == AndroidReasoner

    @property
    def path(self):
        return find_executable('adb')

    def args(self, task: str, mode: str) -> List[str]:
        instrument_env = [('task', task), ('resource', MetaArgs.INPUT)]

        if task == ReasoningTask.MATCHMAKING:
            instrument_env.append(('request', MetaArgs.REQUEST))

        instrument_env = ' '.join(['{}:{}'.format(*env_kv) for env_kv in instrument_env])
        shell_cmds = [
            'logcat -c',
            'am instrument -w -e "target" "{}" -e "args" "{}" {}'.format(self.target_package,
                                                                         instrument_env,
                                                                         self.LAUNCHER_CLASSPATH),
            'am force-stop {}'.format(self.target_package),
            'am kill {}'.format(self.target_package),
            'logcat -d -s {}'.format(self.log_prefix)
        ]

        return ['shell', '-x', ';'.join(shell_cmds)]


class IOSReasoner(MobileReasoner, ABC):
    """iOS mobile reasoner wrapper."""

    # Override

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
    def test_name_for_task(self, task: str) -> str:
        """
        Override this method by returning the Xcode test name for the specified reasoning task.
        """
        pass

    # Overrides

    @classmethod
    def is_template(cls) -> bool:
        return cls == IOSReasoner

    @property
    def path(self):
        return find_executable('xcodebuild')

    def args(self, task: str, mode: str) -> List[str]:
        args = ['-project', self._absolute_path(self.project),
                '-scheme', self.scheme,
                '-destination', 'platform=iOS,name={}'.format(self._detect_connected_device()),
                '-only-testing:{}'.format(self.test_name_for_task(task)),
                'test-without-building',
                'RESOURCE={}'.format(MetaArgs.INPUT)]

        if task == ReasoningTask.MATCHMAKING:
            args.append('REQUEST={}'.format(MetaArgs.REQUEST))

        return args

    # Protected

    def _detect_connected_device(self) -> str:
        """Returns the name of a connected device."""
        task = Task('instruments', args=['-s', 'devices'])
        task.run()

        for line in task.stdout.splitlines():
            components = line.split(' (', 1)

            if len(components) == 2 and not components[1].endswith('(Simulator)'):
                return components[0]

        exc.raise_ioerror(errno.ENODEV, message='No connected devices.')
