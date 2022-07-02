from abc import ABC, abstractmethod
from numpy.random import Generator, PCG64


class Evaluator(ABC):
    """Class wrapper to evaluate game positions"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, game):
        """
        Game -> float
        Should return a probability in [0,1] that the current will win the game
        """
        pass


class RandomEvaluator(Evaluator):
    """
    Returns a random evaluation in [0,1] for any position
    """

    def __init__(self):
        self.rng = Generator(PCG64())
        super().__init__()

    def evaluate(self, game):
        if game.outcome is not None:
            return -1 * game.outcome
        return self.rng.uniform(-0.1, 0.1)
