from abc import ABC, abstractmethod
import random


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
        pass

    def evaluate(self, game):
        return random.uniform()
