from __future__ import annotations

import errno
import os
from abc import ABC, abstractmethod
from os import path
from typing import List

from pyutils import exc
from pyutils.proc.task import OutputAction, Task
from pyutils.proc.util import find_executable
from .base import ReasoningTask, RemoteReasoner
from ..config import Paths
from ..evaluation.mode import EvaluationMode


class AndroidReasoner(RemoteReasoner, ABC):
    """Abstract Android reasoner interface."""

    PACKAGE = 'it.poliba.sisinflab.owl.evowluator'
    LAUNCHER_CLASSPATH = PACKAGE + '/.Launcher'
    LAUNCHER_DIR = path.join(Paths.LIB_DIR, 'android-launcher')
    APK_PATH = path.join(LAUNCHER_DIR, 'app', 'build', 'outputs',
                         'apk', 'release', 'app-release.apk')

    # Override
    @property
    @abstractmethod
    def target_package(self) -> str:
        """Package identifier of the reasoner app."""
        pass

    @property
    @abstractmethod
    def log_prefix(self) -> str:
        """String prepended to log messages, used to filter logcat output."""
        pass

    # Overrides

    @classmethod
    def is_template(cls) -> bool:
        return cls == AndroidReasoner

    @property
    def path(self) -> str:
        return find_executable('adb')

    def args(self, task: ReasoningTask, mode: EvaluationMode,
             inputs: List[str], output: str | None) -> List[str]:
        instrument_env = [('task', task.name), ('resource', inputs[0])]

        if task == ReasoningTask.MATCHMAKING:
            instrument_env.append(('request', inputs[1]))

        instrument_env = ' '.join([f'{env_kv[0]}:{env_kv[1]}' for env_kv in instrument_env])
        shell_cmds = [
            f'logcat -c',
            f'am instrument -w -e "target" "{self.target_package}" '
            f'-e "args" "{instrument_env}" {self.LAUNCHER_CLASSPATH}',
            f'am force-stop {self.target_package}',
            f'am kill {self.target_package}',
            f'logcat -d -s {self.log_prefix}'
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
        args = ['shell', f'cmd package list packages {self.PACKAGE}']
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


class IOSReasoner(RemoteReasoner, ABC):
    """Abstract iOS reasoner interface."""

    # Override

    @property
    @abstractmethod
    def project(self) -> str:
        """Xcode project path."""
        pass

    @property
    @abstractmethod
    def scheme(self) -> str:
        """Xcode scheme containing the tests."""
        pass

    @abstractmethod
    def test_name_for_task(self, task: ReasoningTask) -> str:
        """
        Returns the name of the test method exposing the specified reasoning task.

        :param task: Reasoning task.
        :return: Test method name.
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

    def args(self, task: ReasoningTask, mode: EvaluationMode,
             inputs: List[str], output: str | None) -> List[str]:
        inputs = [os.path.basename(f) for f in inputs]

        args = self._common_args() + [
            f'-only-testing:{self.test_name_for_task(task)}',
            f'test-without-building',
            f'RESOURCE={inputs[0]}'
        ]

        if task == ReasoningTask.MATCHMAKING:
            args.append(f'REQUEST={inputs[1]}')

        return args

    # Protected

    def _common_args(self) -> List[str]:
        """Common arguments used in xcodebuild invocations."""
        return ['-project', Paths.absolute(self.project),
                '-scheme', self.scheme,
                '-destination', f'platform=iOS,name={self._detect_connected_device()}']

    def _detect_connected_device(self) -> str:
        """Returns the name of a connected device."""
        for line in Task.spawn('instruments', args=['-s', 'devices']).stdout.splitlines():
            components = line.split(' (', 1)

            if len(components) == 2 and not components[1].endswith('(Simulator)'):
                return components[0]

        exc.raise_ioerror(errno.ENODEV, message='No connected devices.')
