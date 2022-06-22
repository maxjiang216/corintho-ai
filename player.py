from abc import ABC, abstractmethod
import random
from move import Move


def decode_move(move):
    if len(move) != 3:
        return False
    char_to_num = {'A': 2, 'C': 1, 'B': 0}
    if move[0] in ['A', 'B', 'C']:
        if not(move[1] in "0123" and move[2] in "0123"):
            return False
        return Move(True, int(move[1]), int(move[2]), ptype=char_to_num[move[0]])
    else:
        if not(move[0] in "0123" and move[1] in "0123"):
            return False
        dx = dy = 0
        if move[2] == 'U':
            dy = -1
        elif move[2] == 'D':
            dy = 1
        elif move[2] == 'L':
            dx = -1
        elif move[2] == 'R':
            dx = 1
        else:
            return False
        return Move(False, int(move[0]), int(move[1]), row2=int(move[0])+dy, col2=int(move[1])+dx)


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


class HumanPlayer(Player):
    """
    accepts user input to choose moves
    """

    def __init__(self):
        pass

    def get_move(self, game):
        move = decode_move(input(str(game)))
        print(move)
        legal = move and game.is_legal(move)
        while not legal:
            print("Illegal move!")
            move = move and decode_move(input(str(game)))
            legal = game.is_legal(move)
        return move
