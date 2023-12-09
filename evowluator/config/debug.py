import traceback

from .onerror import OnError


class Debug:
    """Debug config namespace."""
    ON_ERROR = OnError.LOG
    TRACE = False

    @classmethod
    def format(cls, e: Exception) -> str:
        return ''.join(traceback.format_exception(e)) if cls.TRACE else str(e)
