from abc import ABC, abstractmethod
import random


class Player(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self, game, legal_moves):
        pass

    @abstractmethod
    def receive_opp_move(self, move):
        pass


class RandomPlayer(Player):
    """
    Chooses uniformly between legal moves
    """

    def __init__(self):
        pass

    def get_move(self, game, legal_moves):
        move = random.choice(legal_moves)
        return move
