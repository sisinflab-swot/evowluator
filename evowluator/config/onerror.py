from pyutils.types.strenum import StrEnum


class OnError(StrEnum):
    """Behavior in case of errors."""
    IGNORE = 'ignore'
    LOG = 'log'
    PAUSE = 'pause'
    ABORT = 'abort'
