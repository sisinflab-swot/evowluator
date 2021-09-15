import os
from importlib import import_module
from ..config import User


def import_user_modules() -> None:
    for package in User.PACKAGES:
        pkg = import_module(package)

        for file in os.listdir(pkg.__path__[0]):
            if file.endswith('.py') and not file.startswith('_'):
                import_module(f'.{file[:-3]}', package)
