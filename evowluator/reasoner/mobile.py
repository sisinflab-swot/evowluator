import errno
import os
from abc import ABC, abstractmethod
from typing import List, Optional

from pyutils import exc
from pyutils.proc.task import OutputAction, Task
from pyutils.proc.util import find_executable
from evowluator.config import Paths
from os import path

from evowluator.evaluation.mode import EvaluationMode
from .base import (
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

    def classify(self,
                 input_file: str,
                 output_file: Optional[str] = None,
                 timeout: Optional[float] = None,
                 mode: EvaluationMode = EvaluationMode.CORRECTNESS) -> ReasoningResults:
        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CLASSIFICATION, mode=mode),
                                input_arg=os.path.basename(input_file))
        task = self._run(args=args, timeout=timeout, mode=mode)
        return self.results_parser.parse_classification_results(task)

    def consistency(self,
                    input_file: str,
                    timeout: Optional[float] = None,
                    mode: EvaluationMode = EvaluationMode.CORRECTNESS) -> ReasoningResults:
        args = MetaArgs.replace(args=self.args(task=ReasoningTask.CONSISTENCY, mode=mode),
                                input_arg=os.path.basename(input_file))
        task = self._run(args, timeout=timeout, mode=mode)
        return self.results_parser.parse_consistency_results(task)

    def matchmaking(self,
                    resource_file: str,
                    request_file: str,
                    output_file: Optional[str] = None,
                    timeout: Optional[float] = None,
                    mode: EvaluationMode = EvaluationMode.CORRECTNESS) -> MatchmakingResults:
        args = MetaArgs.replace(args=self.args(task=ReasoningTask.MATCHMAKING, mode=mode),
                                input_arg=os.path.basename(resource_file),
                                request_arg=os.path.basename(request_file))
        task = self._run(args, timeout=timeout, mode=mode)
        return self.results_parser.parse_matchmaking_results(task)

    def _run(self, args: List[str], timeout: Optional[float], mode: EvaluationMode) -> Task:
        exc.raise_if_not_found(self._absolute_path(self.path), file_type=exc.FileType.FILE)
        task = Task(self._absolute_path(self.path), args=args)
        task.run(timeout=timeout)
        return task


class AndroidReasoner(MobileReasoner, ABC):
    """Android mobile reasoner wrapper."""

    PACKAGE = 'it.poliba.sisinflab.owl.evowluator'
    LAUNCHER_CLASSPATH = PACKAGE + '/.Launcher'
    LAUNCHER_DIR = path.join(Paths.LIB_DIR, 'android-launcher')
    APK_PATH = path.join(LAUNCHER_DIR, 'app', 'build', 'outputs',
                         'apk', 'release', 'app-release.apk')

    # Override
    @property
    @abstractmethod
    def target_package(self) -> str:
        """Main reasoner app package."""
        pass

    @property
    @abstractmethod
    def log_prefix(self) -> str:
        """Prefix prepended to log messages, used to filter logcat output."""
        pass

    # Overrides

    @classmethod
    def is_template(cls) -> bool:
        return cls == AndroidReasoner

    @property
    def path(self) -> str:
        return find_executable('adb')

    def args(self, task: ReasoningTask, mode: EvaluationMode) -> List[str]:
        instrument_env = [('task', task.value), ('resource', MetaArgs.INPUT)]

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

    def setup(self) -> None:
        if self._is_instrumentation_installed():
            return

        if not self._is_instrumentation_assembled():
            self._assemble_instrumentation()

        self._install_instrumentation()

    def teardown(self) -> None:
        self._uninstall_instrumentation()

    # Protected

    def _is_instrumentation_installed(self) -> bool:
        args = ['shell', 'cmd package list packages {}'.format(self.PACKAGE)]
        adb = Task.spawn(self.path, args=args)
        adb.raise_if_failed(message='Cannot start Android Debug Bridge')
        return True if adb.stdout else False

    def _is_instrumentation_assembled(self) -> bool:
        return os.path.isfile(self.APK_PATH)

    def _assemble_instrumentation(self) -> None:
        base_args = ['-p', self.LAUNCHER_DIR]
        tasks = ['wrapper']

        gradle = Task.spawn('gradle', args=base_args + tasks)
        gradle.raise_if_failed(message='Cannot generate Gradle wrapper')

        tasks = ['clean', 'assembleRelease']
        gradle = Task.spawn(path.join(self.LAUNCHER_DIR, 'gradlew'), args=base_args + tasks)
        gradle.raise_if_failed(message='Gradle build failed')

    def _install_instrumentation(self) -> None:
        adb = Task.spawn(self.path, args=['install', self.APK_PATH])
        adb.raise_if_failed(message='Cannot install instrumentation on device')

    def _uninstall_instrumentation(self) -> None:
        adb = Task.spawn(self.path, args=['uninstall', self.PACKAGE])
        adb.raise_if_failed(message='Cannot uninstall instrumentation from device')


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
    def test_name_for_task(self, task: ReasoningTask) -> str:
        """
        Override this method by returning the Xcode test name for the specified reasoning task.
        """
        pass

    # Overrides

    @classmethod
    def is_template(cls) -> bool:
        return cls == IOSReasoner

    @property
    def path(self) -> str:
        return find_executable('xcodebuild')

    def setup(self) -> None:
        args = self._common_args() + ['build-for-testing']
        Task.spawn(self.path, args=args, output_action=OutputAction.DISCARD)

    def args(self, task: ReasoningTask, mode: EvaluationMode) -> List[str]:
        args = self._common_args() + [
            '-only-testing:{}'.format(self.test_name_for_task(task)),
            'test-without-building',
            'RESOURCE={}'.format(MetaArgs.INPUT)
        ]

        if task == ReasoningTask.MATCHMAKING:
            args.append('REQUEST={}'.format(MetaArgs.REQUEST))

        return args

    # Protected

    def _common_args(self) -> List[str]:
        """Common arguments used in xcodebuild invocations."""
        return ['-project', self._absolute_path(self.project),
                '-scheme', self.scheme,
                '-destination', 'platform=iOS,name={}'.format(self._detect_connected_device())]

    def _detect_connected_device(self) -> str:
        """Returns the name of a connected device."""
        task = Task('instruments', args=['-s', 'devices'])
        task.run()

        for line in task.stdout.splitlines():
            components = line.split(' (', 1)

            if len(components) == 2 and not components[1].endswith('(Simulator)'):
                return components[0]

        exc.raise_ioerror(errno.ENODEV, message='No connected devices.')
