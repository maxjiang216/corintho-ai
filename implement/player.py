from abc import ABC, abstractmethod
import numpy as np
from implement.move import Move


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
        self.rng = np.random.Generator(np.random.PCG64())
        super().__init__()

    def get_move(self, game, legal_moves):
        choices = []
        for i, legal in enumerate(legal_moves):
            if legal:
                choices.append(i)
        move = self.rng.choice(choices)
        return move

    def receive_opp_move(self, move):
        super().receive_opp_move(move)


class HumanPlayer(Player):
    """
    accepts user input to choose moves
    """

    def __init__(self):
        super().__init__()

    def get_move(self, game, legal_moves):
        move = self.decode_move(input(str(game)))
        legal = move and legal_moves[move]
        while not legal:
            print("Illegal move!")
            move = self.decode_move(input())
            legal = move and legal_moves[move]
        return move

    def receive_opp_move(self, move):
        super().receive_opp_move(move)

    @staticmethod
    def decode_move(move):
        if len(move) != 3:
            return False
        char_to_num = {"A": 2, "C": 1, "B": 0}
        # Place
        if move[0] in ["A", "B", "C"]:
            if not (move[1] in "0123" and move[2] in "0123"):
                return False
            return Move.encode_place(char_to_num[move[0]], int(move[1]), int(move[2]))
        # Move
        else:
            if not (move[0] in "0123" and move[1] in "0123"):
                return False
            dx = 0
            dy = 0
            if move[2] == "U":
                dy = -1
            elif move[2] == "D":
                dy = 1
            elif move[2] == "L":
                dx = -1
            elif move[2] == "R":
                dx = 1
            else:
                return False
            return Move.encode_move(
                int(move[0]),
                int(move[1]),
                int(move[0]) + dy,
                int(move[1]) + dx,
            )
