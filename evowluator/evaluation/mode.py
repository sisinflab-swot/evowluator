from ..util.strenum import StrEnum


class EvaluationMode(StrEnum):
    """Evaluation modes."""

    CORRECTNESS = 'correctness'
    """Correctness evaluation."""

    PERFORMANCE = 'performance'
    """Performance evaluation."""
