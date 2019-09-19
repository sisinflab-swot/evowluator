from evowluator.util.strenum import StrEnum


class EvaluationMode(StrEnum):
    """Evaluation modes."""
    CORRECTNESS = 'correctness'
    PERFORMANCE = 'performance'
    ENERGY = 'energy'
