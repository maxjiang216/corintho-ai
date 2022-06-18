from abc import ABC, abstractmethod
from board import Board


class Player(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def getMove(self, board):
        pass
