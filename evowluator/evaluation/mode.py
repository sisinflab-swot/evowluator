from evowluator.util.strenum import StrEnum


class EvaluationMode(StrEnum):
    """Evaluation modes."""

    CORRECTNESS = 'correctness'
    """Correctness evaluation."""

    PERFORMANCE = 'performance'
    """Performance evaluation."""

    ENERGY = 'energy'
    """Energy impact evaluation."""
