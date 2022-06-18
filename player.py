from abc import ABC, abstractmethod
from game import Game
import random


class Player(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self, game):
        pass


class RandomPlayer(Player):
    def __init__(self):
        pass

    def get_move(self, game):
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves)
