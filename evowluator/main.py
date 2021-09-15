import sys

from pyutils.io import echo
from . import cli, config
from .user import loader


# Main


def main() -> int:
    try:
        loader.import_user_modules()
        ret_val = cli.process_args()
    except KeyboardInterrupt:
        echo.error('Interrupted by user.')
        ret_val = 1
    except Exception as e:
        if config.DEBUG:
            raise
        else:
            echo.error(str(e))
            ret_val = 1

    return ret_val


if __name__ == '__main__':
    sys.exit(main())
