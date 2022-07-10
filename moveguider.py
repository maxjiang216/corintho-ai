import numpy as np
from abc import ABC, abstractmethod


class MoveGuider(ABC):
    """Class wrapper to evaluate move probabilities"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, game):
        """
        Game -> array
        Should return an array of size 96 of probabilities summing to 1
        """
        pass


class UniformGuider(MoveGuider):
    """
    Returns uniform probabilities for all legal moves
    """

    def __init__(self):
        pass

    def generate(self, game):
        n = len(game.get_legal_moves())
        if n == 0:
            return []
        return np.full(n, 1 / n)
