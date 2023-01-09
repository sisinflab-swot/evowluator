import sys


def version_check():
    min_python = (3, 9)

    if sys.version_info[:2] < min_python:
        print('evOWLuator equires Python version {}.{} or newer.'.format(*min_python))
        sys.exit(1)


def main():
    from pyutils.io import echo
    from . import cli
    from .user import loader

    try:
        loader.import_user_modules()
        ret_val = cli.process_args()
    except KeyboardInterrupt:
        echo.error('Interrupted by user.')
        ret_val = 1
    except Exception as e:
        from . import config
        echo.error(config.Debug.format(e))
        ret_val = 1

    return ret_val


if __name__ == '__main__':
    version_check()
    sys.exit(main())
