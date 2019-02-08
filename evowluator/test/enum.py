class TestMode:
    """Namespace for test modes."""
    CORRECTNESS = 'correctness'
    TIME = 'time'
    MEMORY = 'memory'
    ENERGY = 'energy'

    ALL = [CORRECTNESS, TIME, MEMORY, ENERGY]


class TestName:
    """Namespace for test names."""

    class Classification:
        CORRECTNESS = 'classification correctness'
        TIME = 'classification time'
        MEMORY = 'classification memory'
        ENERGY = 'classification energy'

        ALL = [CORRECTNESS, TIME, MEMORY, ENERGY]

    class Consistency:
        CORRECTNESS = 'consistency correctness'
        TIME = 'consistency time'
        MEMORY = 'consistency memory'
        ENERGY = 'consistency energy'

        ALL = [CORRECTNESS, TIME, MEMORY, ENERGY]

    class Matchmaking:
        CORRECTNESS = 'matchmaking correctness'
        TIME = 'matchmaking time'
        MEMORY = 'matchmaking memory'
        ENERGY = 'matchmaking energy'

        ALL = [CORRECTNESS, TIME, MEMORY, ENERGY]

    class Misc:
        INFO = 'info'
        NOT_IMPLEMENTED = 'not implemented'

    TIME = [Classification.TIME, Consistency.TIME, Matchmaking.TIME]
    MEMORY = [Classification.MEMORY, Consistency.MEMORY, Matchmaking.MEMORY]
    ENERGY = [Classification.ENERGY, Consistency.ENERGY, Matchmaking.ENERGY]
