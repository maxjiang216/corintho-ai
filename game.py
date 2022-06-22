import numpy as np
import time
from copy import deepcopy
from board import Board
from move import Move


class Game:
    """
    Represents Coritho game
    A board with player pieces, current player, and outcome
    """

    GetLegalMovesTime = 0
    IsLegalTime = 0
    DeepCopyTime = 0
    DoMoveTime = 0

    def __init__(self):
        self.board = Board()
        # Which player is playing
        self.to_play = 0
        # Number of pieces for each player
        self.pieces = np.full(6, 4)
        # Outcome when game is done. 1 is first player win. None means the game is ongoing
        self.outcome = None

    def __str__(self):
        return (
            str(self.board) + "\n" + str(self.pieces) + "\n" + str(self.to_play) + "\n"
        )

    def is_legal(self, move, lines):
        """
        Move -> bool
        Checks if move is legal
        """
        t = time.time()
        # Place
        if move.mtype:
            if self.pieces[
                self.to_play * 3 + move.ptype
            ] == 0 or not self.board.can_place(move.row1, move.col1, move.ptype):
                Game.IsLegalTime += time.time() - t
                return False
        # Move
        else:
            if not self.board.can_move(move.row1, move.col1, move.row2, move.col2):
                Game.IsLegalTime += time.time() - t
                return False
        # Check that all lines are broken or extended
        if lines is None:
            Game.IsLegalTime += time.time() - t
            return True

        t2 = time.time()
        temp_board = deepcopy(self.board)
        Game.DeepCopyTime += time.time() - t2
        t3 = time.time()
        temp_board.do_move(move)
        Game.DoMoveTime += time.time() - t3
        new_lines = temp_board.get_lines()

        for count, line in enumerate(lines):
            if line == new_lines[count] == 1:
                Game.IsLegalTime += time.time() - t
                return False
            elif line == 2 and new_lines[count] != 0:
                Game.IsLegalTime += time.time() - t
                return False
        Game.IsLegalTime += time.time() - t
        return True

    def get_legal_moves(self):
        """
        -> array
        Returns a list of all legal moves
        """
        t = time.time()
        moves = []
        # Place
        for ptype in range(3):
            if self.pieces[self.to_play * 3 + ptype] == 0:
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
        lines = self.board.lines
        # More than 2 lines, no legal moves
        if len(lines) > 2:
            return []
        # Need to break / extend lines
        if len(lines) > 0:
            line_breakers = []
            if 
        for move in moves:
            if self.is_legal(move, lines):
                legal_moves.append(move)

        Game.GetLegalMovesTime += time.time() - t
        return legal_moves

    def do_move(self, move):
        """
        Move -> bool,float
        Does move
        Current player loses if move is illegal
        Returns if game is done and the outcome
        """
        # Illegal move
        # if not self.is_legal(move):
        #     self.outcome = 1 - self.to_play
        #     return self.outcome
        # Place, remove piece from arsenal
        if move.mtype:
            self.pieces[self.to_play * 3 + move.ptype] -= 1
        self.board.do_move(move)
        # Previous player win
        self.to_play = 1 - self.to_play
        if len(self.get_legal_moves()) == 0:
            if np.max(self.board.get_lines()) > 0:
                self.outcome = 2 * self.to_play - 1
            else:
                self.outcome = 0
            return self.outcome
        return None

    def get_canonical(self):
        """
        -> Game
        Get canonical form of board (from perspective of current player)
        """
        canonical_game = deepcopy(self)
        canonical_game.to_play = 0
        canonical_game.pieces = np.concatenate(
            (
                self.pieces[self.to_play * 3 : self.to_play * 3 + 3],
                self.pieces[(1 - self.to_play) * 3 : (1 - self.to_play) * 3 + 3],
            ),
        )
        return canonical_game
