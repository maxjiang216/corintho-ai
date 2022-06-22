from abc import ABC, abstractmethod
import random


class Player(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self, game):
        pass


class RandomPlayer(Player):
    """
    Chooses uniformly between legal moves
    """

    def __init__(self):
        pass

    def get_move(self, game):
        legal_moves = game.get_legal_moves()
        move = random.choice(legal_moves)
        return move
