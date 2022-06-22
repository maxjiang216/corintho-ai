import numpy as np
from copy import deepcopy
from board import Board
from move import Move


class Game:
    """
    Represents Coritho game
    A board with player pieces, current player, and outcome
    """

    def __init__(self):
        self.board = Board()
        # Which player is playing
        self.to_play = 0
        # Number of pieces for each player
        self.pieces = [[4, 4, 4], [4, 4, 4]]
        # Outcome when game is done. 1 is first player win. None means the game is ongoing
        self.outcome = None

    def __str__(self):
        return str(self.board) + "\n" + str(self.pieces)

    def is_legal(self, move):
        """
        Move -> bool
        Checks if move is legal
        """
        lines = self.board.get_lines()
        # Place
        if move.mtype:
            if self.pieces[self.to_play][move.ptype] == 0 or not self.board.can_place(
                move.row1, move.col1, move.ptype
            ):
                return False
        # Move
        else:
            if not self.board.can_move(move.row1, move.col1, move.row2, move.col2):
                return False
        # Check that all lines are broken or extended
        temp_board = deepcopy(self.board)
        temp_board.do_move(move)
        new_lines = temp_board.get_lines()
        for count, line in enumerate(lines):
            if line == new_lines[count] == 1:
                return False
            elif line == 2 and new_lines[count] != 0:
                return False
        return True

    def get_legal_moves(self):
        """
        -> array
        Returns a list of all legal moves
        """
        moves = []
        # Place
        for ptype in range(3):
            if self.pieces[self.to_play][ptype] == 0:
                continue
            for row in range(4):
                for col in range(4):
                    moves.append(Move(True, row, col, ptype))
        # Move
        for row1 in range(4):
            for row2 in range(4):
                if abs(row1 - row2) > 1:
                    continue
                for col1 in range(4):
                    for col2 in range(4):
                        moves.append(Move(False, row1, col1, 0, row2, col2))

        legal_moves = []
        for move in moves:
            if self.is_legal(move):
                legal_moves.append(move)

        return legal_moves

    def do_move(self, move):
        """
        Move -> bool,float
        Does move
        Current player loses if move is illegal
        Returns if game is done and the outcome
        """
        # Illegal move
        if not self.is_legal(move):
            self.outcome = 1 - self.to_play
            return self.outcome
        # Place, remove piece from arsenal
        if move.mtype:
            self.pieces[self.to_play][move.ptype] -= 1
        self.board.do_move(move)
        # Previous player win
        self.to_play = 1 - self.to_play
        if len(self.get_legal_moves()) == 0:
            if np.max(self.board.get_lines()) > 0:
                self.outcome = 1 - self.to_play
            else:
                self.outcome = 0.5
            return self.outcome
        return None

    def get_canonical(self):
        """
        -> Game
        Get canonical form of board (from perspective of current player)
        """
        canonical_game = deepcopy(self)
        canonical_game.to_play = 0
        canonical_game.pieces = [
            self.pieces[self.to_play],
            self.pieces[1 - self.to_play],
        ]
        return canonical_game
